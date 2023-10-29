import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
max_new_tokens = 12 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability


class NanoNextGPT(nn.Module):
    def __init__(self, config):
        super(NanoNextGPT, self).__init__()
        
        # Initialize Image Encoder
        self.image_encoder = self.init_image_encoder(config)
        self.image_encoder.to(device)
        
        # Initialize Image Input Projection Layer
        self.input_projection = self.init_input_projection(config)
        self.input_projection.to(device)
        
        # Initialize the gpt2 tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
        self.new_tokens = ["<Img>", "</Img>"]
        self.num_added_tokens = self.tokenizer.add_tokens(self.new_tokens)

        # Initialize GPT-2 Language Model
        self.llm = self.init_llm(config)
        
        # TODO - multimodal output
        # Initialize multi-modal output layers
        # self.image_output_projection = nn.Linear(config.gpt2_dim, config.image_embedding_dim)
        # self.image_output_projection.to(device)
        #self.image_decoder = SimpleImageDecoder(config.image_embedding_dim)
        # self.image_decoder.to(device)
        
    def init_image_encoder(self, config):
        # Use a pre-trained ResNet-18 model and remove the last fully-connected layer to get a feature vector
        resnet18 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        modules = list(resnet18.children())[:-2]
        return nn.Sequential(*modules)
    
    def init_input_projection(self, config):
        input_dim = config.image_embedding_dim
        return nn.Linear(input_dim, config.gpt2_dim)
    
    def init_llm(self, config):
        model = model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
        model.eval()
        model.resize_token_embeddings(len(self.tokenizer)) # resize embedding layer bc of added tokens for <Img> tokens
        model.to(device)
        return model

    def prepare_image_embed(self, image_tensor, pos_in_seq):
        # embed the image which is encoded and fed through a linear layer
        encoded_image = self.image_encoder(image_tensor)
        flattened_encoded = torch.flatten(encoded_image, 1) # Flatten the image embedding to feed into linear layer
        temp_embedding = self.input_projection(flattened_encoded).unsqueeze(1) # linear layer
        pos = torch.arange(pos_in_seq, pos_in_seq +1, dtype=torch.long, device=device) # shape (t)
        pos_emb = self.llm.transformer.wpe.weight[pos,:] # position embeddings of shape (t, n_embd)
        img_embedding = temp_embedding + pos_emb
        return img_embedding

    def get_text_embedding(self, token_idx, start_pos, end_pos):
        """
        Take the text tokens indices idx (LongTensor of shape (b,t)) and return
        token + position embeddings. start_pos and end_pos are where we are in the sequence.
        """
        device = token_idx.device
        pos = torch.arange(start_pos, end_pos, dtype=torch.long, device=device) # shape (t)  
        tok_emb = self.llm.transformer.wte.weight[token_idx,:] # token embeddings of shape (b, t, n_embd)
        pos_emb = self.llm.transformer.wpe.weight[pos,:] # position embeddings of shape (t, n_embd)
        emb = tok_emb + pos_emb
        return emb

    def forward(self, input_ids, labels, attention_mask, image_tensor):
        """
        Forward pass for the model.
        """
        # Prepare the image embedding
        img_embed = self.prepare_image_embed(image_tensor, 1)
        ids_before_img = input_ids[:, :1] # <Img>
        ids_after_img = input_ids[:, 1:] # </Img>caption: a picture of a cat...
        before_embed = self.get_text_embedding(ids_before_img, 0, 1)
        after_embeds = self.get_text_embedding(ids_after_img, 2, input_ids.size()[1] + 1) # +1 for img embed pos
        embeds = torch.cat((before_embed, img_embed,  after_embeds), dim=1) # <Img> image features </Img>caption: a picture of a cat...
        if labels is not None:
            outputs = self.llm.forward(inputs_embeds=embeds, labels=labels, attention_mask=attention_mask)
        else:
            outputs = self.llm.forward(inputs_embeds=embeds)
        return outputs

class SimpleImageDecoder(nn.Module):
    def __init__(self, input_dim):
        super(SimpleImageDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(input_dim, 128, kernel_size=4),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, kernel_size=4),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1, 1, 1)  # reshape tensor
        return self.decoder(x)

class Config:
    def __init__(self):
        self.text_embedding_dim = 1024  # The dimensionality of the text embeddings (usually the same as GPT-2 model dimension)
        self.image_embedding_dim = 2048  # The dimensionality of the image embeddings (usually output from ResNet or similar)
        self.gpt2_dim = 1024  # GPT-2 model dimension (this should match text_embedding_dim if you're using the same GPT-2 model for text)
