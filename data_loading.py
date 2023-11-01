import json
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import torchvision.transforms as transforms
from transformers import GPT2Tokenizer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        
        # Load and transform the image
        img_path = f"{self.img_folder}/{img_name}"
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image).to(device)
        
        return caption, image

def collate_fn(batch):
    """
    Custom collate function to prepare a batch of data for training.
    
    Parameters:
    - batch (List[Tuple[str, Tensor]]): List of tuples containing captions and image tensors.
    
    Returns:
    - Dictionary containing tokenized input_ids, attention_mask, and image tensors.
    """
    # Initialize GPT-2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
    tokenizer.add_tokens(["<Img>", "</Img>"])
    
    # Separate captions and images from the batch
    captions, image = zip(*batch)
    
    # Tokenize the captions to create input_ids
    batch_input_ids = []
    batch_label_ids = []
    for caption in captions:
        prompt = f"<Img></Img>caption: "
        input_ids = tokenizer.encode(prompt)
        label_ids = [-100] * (len(input_ids) + 1) # +1 for img embed (will be added to input ids in fwd)
        input_ids += tokenizer.encode(caption)
        label_ids += tokenizer.encode(caption)
        batch_input_ids.append(torch.LongTensor(input_ids).to(device))
        batch_label_ids.append(torch.LongTensor(label_ids).to(device))
    


    eos_token_id = tokenizer.encode(tokenizer.eos_token)[0] # use this token for pad since gpt2 has no pad token
    batch_input_ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(seq) for seq in batch_input_ids], batch_first=True, padding_value=eos_token_id)
    batch_label_ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(seq) for seq in batch_label_ids], batch_first=True, padding_value=eos_token_id)
    attention_mask = batch_input_ids.ne(eos_token_id).long()
    prepend_tensor = torch.ones((attention_mask.size(0), 1), device='cuda:0').long()
    updated_attention_mask = torch.cat((prepend_tensor, attention_mask), dim=1)
    return {
        "input_ids": batch_input_ids,
        "attention_mask": updated_attention_mask,
        "labels": batch_label_ids,
        "image": torch.stack(image)
    }
