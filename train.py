import argparse
parser = argparse.ArgumentParser(description='Training settings')
parser.add_argument('--load_pretrained', action='store_true', help='Load pretrained model')
args = parser.parse_args()

from model import NanoNextGPT, Config
from data_loading import *
import torch
import torch.nn as nn
from transformers import AdamW

def quick_validate(model, val_dataloader):
    model.eval()
    with torch.no_grad():
        # Get a single batch from val_dataloader
        batch = next(iter(val_dataloader))
        
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        image = batch['image']
        labels = batch['labels']
        
        outputs = model.forward(input_ids, labels, attention_mask, image)
        loss = outputs.loss
        val_loss = loss.item()
        
        # Assuming logits are against each token ID for language modeling
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        
        # Mask to ignore specific tokens like -100 in labels
        mask = (labels != -100)
        
        # Update accuracy calculations
        total_correct = ((predictions == labels) & mask).sum().item()
        total_count = mask.sum().item()
        
        val_accuracy = total_correct / total_count

    return val_loss, val_accuracy

def train_model(model, train_dataloader, val_dataloader, num_epochs=1):
    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=1e-4)

    # Calculate total steps and steps per quarter epoch
    total_steps = len(train_dataloader) * num_epochs
    steps_per_half_epoch = len(train_dataloader) // 2
    step = 0

    # Training loop converted to step-based
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for batch in train_dataloader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            image = batch['image']
            labels = batch['labels']
            
            outputs = model.forward(input_ids, labels, attention_mask, image)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

            # Periodic quick validation
            if step % steps_per_half_epoch == 0:
                val_loss, val_accuracy = quick_validate(model, val_dataloader)
                print(f"Step {step}/{total_steps} - Quick Val Loss: {val_loss} - Quick Val Accuracy: {val_accuracy}")

            # Periodic model saving
            if step % steps_per_half_epoch == 0:
                torch.save(model.state_dict(), f'C:/Users/Brian/Desktop/nano_nextgpt/ckpt/nano_nextgpt_med_step_{step}.pth')

            step += 1

        avg_train_loss = train_loss / len(train_dataloader)
        print(f"Step {step}/{total_steps} - Avg Train Loss: {avg_train_loss}")


# Note: The function assumes that you have already defined an optimizer and a loss criterion.
# For example:
train_dataset = TXPairDataset(json_path='C:/Users/Brian/Desktop/cc3m_mini/cc3m_mini_train.json', img_folder='C:/Users/Brian/Desktop/cc3m_mini/images/train')
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
val_dataset = TXPairDataset(json_path='C:/Users/Brian/Desktop/cc3m_mini/cc3m_mini_val.json', img_folder='C:/Users/Brian/Desktop/cc3m_mini/images/val')
val_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
config = Config()
model = NanoNextGPT(config)
if args.load_pretrained:
    model.load_state_dict(torch.load('C:/Users/Brian/Desktop/nano_nextgpt/ckpt/nano_nextgpt_med.pth'))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Example usage:
train_model(model, train_dataloader, val_dataloader, num_epochs=1)

