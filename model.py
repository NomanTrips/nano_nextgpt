import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig
import bitsandbytes as bnb
from ImageBind import *
from ImageBind import data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

################################################################################
# bitsandbytes parameters
################################################################################
# Activate 4-bit precision base model loading
use_4bit = True
# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"
# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"
# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False

################################################################################
# QLoRA parameters
################################################################################
# LoRA attention dimension
lora_r = 64
# Alpha parameter for LoRA scaling
lora_alpha = 16
# Dropout probability for LoRA layers
lora_dropout = 0.1

# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules= ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
)

# Load the entire model on the GPU 0
device_map = {"": 0}

#llama_2_path = "./ckpt/bf16_13334" # uncomment if picking up training from finetuned or doing inference. should be finetuned .pth folder
llama_2_path =  "/home/brian/Desktop/Llama-2-7b-chat-hf/" # uncomment if training from scratch. should be llama 2 chat hf folder
tokenizer_path = "/home/brian/Desktop/Llama-2-7b-chat-hf/" # normally llama 2 chat hf path

# Load tokenizer and model with QLoRA configuration
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)
    
class NanoNextGPT(nn.Module):
    def __init__(self, config):
        super(NanoNextGPT, self).__init__()
        
        # Initialize Image Encoder
        self.image_encoder = self.init_image_encoder(config)
        self.image_encoder.to(device)
        
        # Initialize Image Input Projection Layer
        self.input_projection = self.init_input_projection(config)
        self.input_projection.to(device)
        
        # llama 2 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, use_fast=True)
        self.new_tokens = ["<Img>", "</Img>"]
        self.num_added_tokens = self.tokenizer.add_tokens(self.new_tokens)
        self.tokenizer.add_special_tokens({"pad_token":"<pad>"})
        self.tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

        # Initialize llama 2
        self.llm = self.init_llm(config)
        self.config = config
        
    def init_image_encoder(self, config):
        imagebind_ckpt_path = "./.checkpoints/"
        model, visual_hidden_size = imagebind_model.imagebind_huge(pretrained=True, store_path=imagebind_ckpt_path)
        model.eval()
        return model
    
    def init_input_projection(self, config):
        input_dim = config.image_embedding_dim
        return nn.Linear(input_dim, config.llama2_dim)
    
    def init_llm(self, config):
        model = AutoModelForCausalLM.from_pretrained(
            llama_2_path,
            quantization_config=bnb_config,
            device_map=device_map
        )
        model.config.use_cache = False
        model.config.pretraining_tp = 1
        model.config.pad_token_id = self.tokenizer.pad_token_id
        model.resize_token_embeddings(len(self.tokenizer)) # resize embedding layer bc of added tokens for <Img> and <pad> tokens
        if config.load_peft_model:
            model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, peft_config, 'nano_nextgpt')
            model.print_trainable_parameters()
        model.eval()
        return model

    def prepare_image_embed(self, image_paths, device):
        """
        Takes set of image paths corresponding to batch images and passes them through ImageBind --> Linear layer
        returns image embeddings mapped onto the LLM embedding space
        """
        inputs = {
            ModalityType.VISION: data.load_and_transform_vision_data(image_paths, device)
        }
        encoded_images = self.image_encoder(inputs)
        embedded_images = encoded_images[ModalityType.VISION]
        temp_embedding = self.input_projection(embedded_images).unsqueeze(1)  # Linear layer --> embedding
        temp_embedding = temp_embedding.to(self.llm.dtype) # quantization requires squeezing image bind data type down to quantized data type
        return temp_embedding

    def get_text_embedding(self, token_idx):
        """
        Take the text tokens indices idx (LongTensor of shape (b,t)) and return embeddings
        """
        if self.config.load_peft_model == True:
            emb = self.llm.model.model.embed_tokens(token_idx) # peft requires a deeper call to get at the embedding layer
        else:
            emb = self.llm.model.embed_tokens(token_idx)
        return emb

    def insert_image_embeddings(self, embeddings, input_ids, img_tensor):
        """
        Inserts image embeddings into the correct positions in the text embeddings.

        :param input_ids: Tensor of shape [b, s], where b is batch size and s is sequence length.
        :param embeddings: Tensor of embedded input_ids, shape [b, s, e] where e is the embedding size.
        :param img_tensor: Tensor of shape [b, s, f], where f is the feature size of image embeddings.
        :return: Tensor with image embeddings inserted at the correct positions.
        """
        token_id=32000 # id for <Img> the image start token
        # Step 1: Find the positions of token 32000 in input_ids
        positions = (input_ids == token_id).nonzero(as_tuple=True)[1]
    
        # Initialize a list to store the processed tensors
        processed_tensors = []
    
        # Step 2, 3, 4: Iterate over each example
        for i, pos in enumerate(positions):
            first_half = embeddings[i, :pos]
            second_half = embeddings[i, pos:]
    
            # Concatenating with img_tensor
            concatenated = torch.cat([first_half, img_tensor[i], second_half], dim=0)
            processed_tensors.append(concatenated)
    
        # Stack the processed tensors to get them back into a batch
        return torch.stack(processed_tensors)


    def forward(self, input_ids,  labels, attention_mask, image_paths):
        """
        Forward pass for the model.
        """
        # Prepare the image embedding
        img_embed = self.prepare_image_embed(image_paths, device=device)
        text_embed = self.get_text_embedding(input_ids)
        embeds = self.insert_image_embeddings(text_embed, input_ids, img_embed)
        if labels is not None:
            outputs = self.llm.forward(inputs_embeds=embeds, labels=labels, attention_mask=attention_mask)
        else:
            outputs = self.llm.forward(inputs_embeds=embeds)
        return outputs

class Config:
    def __init__(self):
        self.text_embedding_dim = 4096  # llama 2 embedding dimension size
        self.image_embedding_dim = 1024  # ImageBind embedding dim
        self.llama2_dim = 4096  # llama 2 feature dimension size
        self.load_peft_model = False
