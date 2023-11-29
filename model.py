import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, prepare_model_for_kbit_training, prepare_model_for_int8_training, LoraConfig, PeftModel
import bitsandbytes as bnb
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
lora_r = 64 #32
# Alpha parameter for LoRA scaling
lora_alpha = 16 # 32
# Dropout probability for LoRA layers
lora_dropout = 0.1

# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules= ['q_proj', 'k_proj', 'v_proj', 'o_proj'],#['q_proj', 'k_proj', 'v_proj', 'o_proj'], #["gate_proj", "q_proj", "v_proj", "o_proj", "down_proj", "up_proj", "k_proj", "lm_head"]
    # "q_proj", "v_proj"
    #modules_to_save = ['embed_tokens']
)

# Load the entire model on the GPU 0
device_map = {"": 0}

#llama_2_path = "/home/brian/Desktop/Llama-2-7b-chat-hf/" # base llama 2 from huggingface

llama_2_path =  "/home/brian/Desktop/Llama-2-7b-chat-hf/" # for training should be llama 2 base weights dir. inference on finetuned should be: "./ckpt/bf16_13334" or whereever the .pth is
tokenizer_path = "/home/brian/Desktop/Llama-2-7b-chat-hf/" # should normally be llama 2 base tokenizer location

# Load tokenizer and model with QLoRA configuration
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    #load_in_16bit=True,
    #load_in_8bit=True,
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

max_new_tokens = 12 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability

from ImageBind import *
from ImageBind import data

def find_all_linear_names(model, bits):
    cls = (
        bnb.nn.Linear4bit
        if bits == 4
        else (bnb.nn.Linear8bitLt if bits == 8 else torch.nn.Linear)
    )
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    return list(lora_module_names)
    
class NanoNextGPT(nn.Module):
    def __init__(self, config):
        super(NanoNextGPT, self).__init__()
        
        # Initialize Image Encoder
        self.image_encoder = self.init_image_encoder(config)
        self.image_encoder.to(device)
        
        # Initialize Image Input Projection Layer
        self.input_projection = self.init_input_projection(config)
        self.input_projection.to(device)

        #self.layer_norm = nn.LayerNorm(config.gpt2_dim, eps=1e-12).to(device)
        
        # Initialize the gpt2 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, use_fast=True)
        self.new_tokens = ["<Img>", "</Img>"]
        self.num_added_tokens = self.tokenizer.add_tokens(self.new_tokens)
        self.tokenizer.add_special_tokens({"pad_token":"<pad>"})
        #self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training
        #im_start = self.tokenizer.encode("<Img>", skip_special_tokens=False)
        #im_end = self.tokenizer.encode("</Img>", skip_special_tokens=False)
        #pad = self.tokenizer.encode("<pad>", skip_special_tokens=False)
        #print(f"start: {im_start}, end: {im_end}, pad: {pad}")

        # Initialize GPT-2 Language Model
        self.llm = self.init_llm(config)
        self.config = config
        
    def init_image_encoder(self, config):
        # Instantiate model
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
        model.resize_token_embeddings(len(self.tokenizer)) # resize embedding layer bc of added tokens for <Img> tokens
        if config.load_peft_model:
            model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, peft_config, 'nano_nextgpt')
            model.print_trainable_parameters()
        model.eval()
        return model

    def prepare_image_embed(self, image_paths, device):
        # Encode the images and fed them through a linear layer to map onto llama embedding dimension
        inputs = {
            ModalityType.VISION: data.load_and_transform_vision_data(image_paths, device)
        }
        encoded_images = self.image_encoder(inputs)
        embedded_images = encoded_images[ModalityType.VISION]
        temp_embedding = self.input_projection(embedded_images).unsqueeze(1)  # Linear layer --> embedding
        temp_embedding = temp_embedding.to(self.llm.dtype) # quantization
        return temp_embedding

    def get_text_embedding(self, token_idx):
        """
        Take the text tokens indices idx (LongTensor of shape (b,t)) and return embeddings
        """
        #device = token_idx.device
        if self.config.load_peft_model == True:
            emb = self.llm.model.model.embed_tokens(token_idx) # peft requires a deeper call to get at the embedding layer
        else:
            emb = self.llm.model.embed_tokens(token_idx)
        return emb

    def insert_image_embeddings(self, input_ids, text_embeddings, image_embeddings):
        print(f"input_ids.shape: ", input_ids.shape)
        print(f"text_embeddings.shape: ", text_embeddings.shape)
        print(f"image_embeddings.shape: ", image_embeddings.shape)
        """
        Inserts image embeddings into the correct positions in the text embeddings.
    
        :param input_ids: Tensor of shape [b, s], where b is batch size and s is sequence length.
        :param text_embeddings: Tensor of embedded input_ids, shape [b, s, e] where e is the embedding size.
        :param image_embeddings: Tensor of shape [b, s, f], where f is the feature size of image embeddings.
        :return: Tensor with image embeddings inserted at the correct positions.
        """
        batch_size, seq_len, _ = text_embeddings.shape
        img_start_token_id = 32000 # Your <Img> token ID
        img_end_token_id = 32001 # Your </Img> token ID
    
        # We will store the results here
        result = []
    
        for i in range(input_ids.shape[0]):
            start_positions = (input_ids[i] == img_start_token_id).nonzero(as_tuple=True)[0]
            end_positions = (input_ids[i] == img_end_token_id).nonzero(as_tuple=True)[0]
    
            if len(start_positions) != len(end_positions):
                raise ValueError("Mismatch in number of <Img> and </Img> tags")
    
            concatenated = [text_embeddings[i, :start_positions[0]]]
    
            for j, (start, end) in enumerate(zip(start_positions, end_positions)):
                # Add the image embedding
                concatenated.append(image_embeddings[i, j])
    
                # Add the next slice of text embeddings
                next_slice_start = end + 1 if end + 1 < text_embeddings.shape[1] else text_embeddings.shape[1]
                next_slice_end = start_positions[j + 1] if j + 1 < len(start_positions) else text_embeddings.shape[1]
                concatenated.append(text_embeddings[i, next_slice_start:next_slice_end])
            
            for ele in concatenated:
                print(ele.shape)
            concatenated_tensor = torch.cat(concatenated, dim=0)
            result.append(concatenated_tensor)
            
            # Debugging print statements
            print(f"Batch item {i}:")
            print(f"Start positions: {start_positions}")
            print(f"End positions: {end_positions}")
            print(f"Concatenated tensor shape: {concatenated_tensor.shape}")
    
        return torch.stack(result, dim=0)

    def process_tensors(self, embeddings, input_ids, new_tensor):
        token_id=32000
        # Step 1: Find the positions of token 32000 in input_ids
        positions = (input_ids == token_id).nonzero(as_tuple=True)[1]
    
        # Initialize a list to store the processed tensors
        processed_tensors = []
    
        # Step 2, 3, 4: Iterate over each example
        for i, pos in enumerate(positions):
            first_half = embeddings[i, :pos]
            second_half = embeddings[i, pos:]
    
            # Concatenating with new_tensor
            concatenated = torch.cat([first_half, new_tensor[i], second_half], dim=0)
            processed_tensors.append(concatenated)
    
        # Stack the processed tensors to get them back into a batch
        return torch.stack(processed_tensors)


    def forward(self, input_ids,  labels, attention_mask, image_paths):
        """
        Forward pass for the model.
        """
        # Prepare the image embedding
        img_embed = self.prepare_image_embed(image_paths, device=device)
        #img_start = (input_ids[0] == 32000).nonzero(as_tuple=True)[0]  # pos of <Img> in input ids
        #img_end = (input_ids[0] == 32001).nonzero(as_tuple=True)[0] # pos of </Img> in input ids
        text_embed = self.get_text_embedding(input_ids)
        #embed_before_img = text_embed[:, :img_start[0] + 1, :]
        #embed_after_img = text_embed[:, img_end[0]:, :]
        #embeds = self.insert_image_embeddings(input_ids, text_embed, img_embed)
        #print(f"input_ids.shape: ", input_ids.shape)
        #print(f"text_embed.shape: ", text_embed.shape)
        #print(f"img_embed.shape: ", img_embed.shape)
        embeds = self.process_tensors(text_embed, input_ids, img_embed)
        #print(f"embeds.shape: ", embeds.shape)
        #print(f"embeds.shape: ", embeds.shape)
        #embeds = torch.cat((embed_before_img, img_embed,  embed_after_img), dim=1)
        #print(f"input_ids[0]: ", input_ids[0])
        #print(f"labels[0]: ", labels[0])
        #print(f"attention_mask[0]: ", attention_mask[0])
        if labels is not None:
            outputs = self.llm.forward(inputs_embeds=embeds, labels=labels, attention_mask=attention_mask) # labels=labels,
        else:
            outputs = self.llm.forward(inputs_embeds=embeds)
        return outputs

class Config:
    def __init__(self):
        self.text_embedding_dim = 4096  # llama 2 embedding dimension size
        self.image_embedding_dim = 1024  # The dimensionality of the image embeddings (usually output from ResNet or similar)
        self.llama2_dim = 4096  # llama 2 feature dimension size
        self.load_peft_model = False
