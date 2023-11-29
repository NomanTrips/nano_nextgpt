import json
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
from transformers import AutoTokenizer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
llama_2_path = "/home/brian/Desktop/Llama-2-7b-chat-hf/"

class TXPairDatasetInstruct(Dataset):
    def __init__(self, json_path, img_folder):
        # Load the JSON file containing the captions and image filenames
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        self.img_folder = img_folder

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        conversations = item['conversations']
        img_name = item['image']
        img_path = f"{self.img_folder}/{img_name}"
        
        return conversations, img_path

def collate_fn_instruct(batch):
    """
    Custom collate function to prepare a batch of data for training.
    
    Parameters:
    - batch (List[Tuple[str, Tensor]]): List of tuples containing captions and image tensors.
    
    Returns:
    - Dictionary containing tokenized input_ids, attention_mask, and image tensors.
    """
    # Initialize Llama 2 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(llama_2_path, trust_remote_code=True, use_fast=True)
    new_tokens = ["<Img>", "</Img>"]
    tokenizer.add_tokens(new_tokens)
    tokenizer.add_special_tokens({"pad_token":"<pad>"})
    tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training
    
    # Separate captions and images from the batch
    conversations, images = zip(*batch)
    
    # Tokenize the captions to create input_ids
    batch_input_ids = []
    batch_label_ids = []

    for conversation in conversations:
        input_ids = []
        label_ids = []

        for turn in conversation:
            # Tokenize the turn
            tokens = tokenizer(turn, add_special_tokens=False)['input_ids']

            if '[INST]' in turn:  # Human turn
                input_ids.extend(tokens)
                label_ids.extend([-100] * len(tokens))  # Masking the human input
                if '<Img>' in turn:
                    label_ids.append(-100) # for image embedding added to input later
            else:  # Assistant turn
                input_ids.extend(tokens)
                label_ids.extend(tokens)  # No masking for assistant's response

        # Shift the labels for loss calculation
        #label_ids =  label_ids[1:] +  [-100]

        batch_input_ids.append(input_ids)
        batch_label_ids.append(label_ids)
    

    batch_input_ids_tensors = [torch.tensor(ids).to(device) for ids in batch_input_ids]
    batch_label_ids_tensors = [torch.tensor(labels).to(device) for labels in batch_label_ids]
    #batch_input_ids_tensors
    #batch_label_ids_tensors
    
    pad_token_id = tokenizer.encode(tokenizer.pad_token)[1] # use this token for pad since llama2 has no pad token
    batch_input_ids = torch.nn.utils.rnn.pad_sequence(batch_input_ids_tensors, batch_first=True, padding_value=pad_token_id)
    batch_label_ids = torch.nn.utils.rnn.pad_sequence(batch_label_ids_tensors, batch_first=True, padding_value=pad_token_id)
    attention_mask = batch_input_ids.ne(pad_token_id).long()
    prepend_tensor = torch.ones((attention_mask.size(0), 1), device='cuda:0').long() # for image embedding
    updated_attention_mask = torch.cat((prepend_tensor, attention_mask), dim=1)
    return {
        "input_ids": batch_input_ids,
        "attention_mask": updated_attention_mask,
        "labels": batch_label_ids,
        "image_paths": images
    }
