import json
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import torchvision.transforms as transforms
from transformers import AutoTokenizer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
llama_2_path = "/home/brian/Desktop/Llama-2-7b-chat-hf/"

class TXPairDataset(Dataset):
    def __init__(self, json_path, img_folder):
        # Load the JSON file containing the captions and image filenames
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        self.img_folder = img_folder
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        caption = item['caption']
        img_name = item['image_name']
        img_path = f"{self.img_folder}/{img_name}"
        
        return caption, img_path

def collate_fn(batch):
    """
    Custom collate function to prepare a batch of data for training.
    
    Parameters:
    - batch (List[Tuple[str, Tensor]]): List of tuples containing captions and image tensors.
    
    Returns:
    - Dictionary containing tokenized input_ids, labels, attention_mask, and image tensors.
    """
    # Initialize Llama 2 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(llama_2_path, trust_remote_code=True)
    new_tokens = ["<Img>", "</Img>"]
    tokenizer.add_tokens(new_tokens)
    tokenizer.add_special_tokens({"pad_token":"<pad>"}) # use this token for pad since llama2 has no pad token
    tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training
    
    # Separate captions and images from the batch
    captions, images = zip(*batch)
    
    # Tokenize the captions to create input_ids
    batch_input_ids = []
    batch_label_ids = []
    for caption in captions:
        prompt = "<s>[INST] Provide a caption for this image: <Img> </Img>[/INST]"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs.input_ids
        len_prompt = len(input_ids[0])
        caption_tok = tokenizer(f"{caption}</s>", return_tensors="pt").to(device)
        caption_ids = caption_tok.input_ids
        input_ids = torch.cat((input_ids, caption_ids), dim=1)
        labels = input_ids.clone()
        labels_clone = labels.clone()
        b, s = labels_clone.shape
        minus_hundreds = torch.full((b, 1), -100).to(device)
        labels_clone = torch.cat((minus_hundreds, labels_clone), dim=1)
        labels_clone[:, :len_prompt] = -100 # don't calc loss on human prompt
        batch_input_ids.append(input_ids[0])
        batch_label_ids.append(labels_clone[0])
    


    pad_token_id = tokenizer.encode(tokenizer.pad_token)[1]
    batch_input_ids = torch.nn.utils.rnn.pad_sequence(batch_input_ids, batch_first=True, padding_value=pad_token_id)
    batch_label_ids = torch.nn.utils.rnn.pad_sequence(batch_label_ids, batch_first=True, padding_value=pad_token_id)
    attention_mask = batch_input_ids.ne(pad_token_id).long()
    prepend_tensor = torch.ones((attention_mask.size(0), 1), device='cuda:0').long()
    updated_attention_mask = torch.cat((prepend_tensor, attention_mask), dim=1)
    return {
        "input_ids": batch_input_ids,
        "attention_mask": updated_attention_mask,
        "labels": batch_label_ids,
        "image_paths": images
    }
