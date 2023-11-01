import argparse
parser = argparse.ArgumentParser(description='Training settings')
parser.add_argument('--load_pretrained', action='store_true', help='Load pretrained model')
args = parser.parse_args()

from model import NanoNextGPT, Config
from data_loading import *
import torch
import torch.nn as nn
from transformers import AdamW

import wandb
wandb.init(project='nano_nextgpt', name='base_llm_frzn')

def quick_validate(model, val_dataloader):
    model.eval()
    total_loss = 0  # Accumulate loss over all batches
    num_batches = len(val_dataloader)  # Get the number of batches
    running_gen_acc = 0.0
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            image = batch['image']
            labels = batch['labels']
            
            outputs = model.forward(input_ids, labels, attention_mask, image)
            loss = outputs.loss
            total_loss += loss.item()  # Accumulate loss
            
            chosen_tokens = torch.max(outputs.logits, dim=-1)[1][:, 1:-1]  # [B, S-1]
            labels = labels[:, 2:]
            gen_acc = (chosen_tokens.reshape(-1) == labels.reshape(-1)).to(torch.long)  # [B*S]
            valid_mask = (labels != -100) & (labels != 50256)
            valid_mask = valid_mask.reshape(-1)
            valid_tokens = gen_acc & valid_mask  # [B*S]
            gen_acc = valid_tokens.sum().item() / (valid_mask.sum().item() + 1.0)
            running_gen_acc += gen_acc            

        val_loss = total_loss / num_batches  # Compute average loss
        val_accuracy = running_gen_acc / num_batches

    return val_loss, val_accuracy


def train_model(model, train_dataloader, val_dataloader, num_epochs=2):
    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=1e-4)

    # Calculate total steps and steps per quarter epoch
    total_steps = len(train_dataloader) * num_epochs
    steps_per_quarter_epoch = len(train_dataloader) // 4
    step = 0

    # Training loop converted to step-based
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        running_train_loss = 0.0
        running_bow_acc = 0.0
        running_gen_acc = 0.0

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
            running_train_loss += loss.item()

            # calculate the token accuracy
            chosen_tokens = torch.max(outputs.logits, dim=-1)[1][:, 1:-1]  # [B, S-1]
            labels = labels[:, 2:]
            #print(f"chosen_tokens[0]: ", chosen_tokens[0])
            #print(f"labels[0]: ", labels[0])
            generated_text = model.tokenizer.decode(chosen_tokens[0], skip_special_tokens=True)
            #print("Generated text:", generated_text)
            filtered_labels = labels[0][labels[0] != -100]
            decoded_text = model.tokenizer.decode(filtered_labels, skip_special_tokens=True)
            #print("label text:", decoded_text)   
            gen_acc = (chosen_tokens.reshape(-1) == labels.reshape(-1)).to(torch.long)  # [B*S]
            valid_mask = (labels != -100) & (labels != 50256)
            valid_mask = valid_mask.reshape(-1)
            #print(f"valid_mask: ", valid_mask)
            valid_tokens = gen_acc & valid_mask  # [B*S]
            gen_acc = valid_tokens.sum().item() / (valid_mask.sum().item() + 1.0)
            running_gen_acc += gen_acc

            # Convert tensors to sets for bag-of-words comparison
            chosen_tokens_set = set(chosen_tokens.cpu().numpy().reshape(-1))
            labels_set = set(labels.cpu().numpy().reshape(-1))

            # Remove -100 (used for padding or ignored tokens)
            labels_set.discard(-100)
            labels_set.discard(50256)
            chosen_tokens_set.discard(50256)

            # Calculate bag-of-words accuracy
            common_tokens = chosen_tokens_set.intersection(labels_set)
            bow_acc = len(common_tokens) / len(labels_set) if len(labels_set) > 0 else 0.0
            running_bow_acc += bow_acc
            #print(f"Step {step} - gen_acc: {gen_acc} - bow_acc: {bow_acc}")
            # Periodic quick validation
            if step % 50 == 0:
                val_loss, val_accuracy = quick_validate(model, val_dataloader)
                avg_loss = running_train_loss / 50  # Compute the average training loss
                avg_gen_acc = running_gen_acc / 50
                avg_bow_acc = running_bow_acc / 50
                print(f"Step {step}/{total_steps} - Val Loss: {val_loss} - Val Acc: {val_accuracy} - Avg loss: {avg_loss} - avg_gen_acc: {avg_gen_acc} - avg_bow_acc: {avg_bow_acc}")
                running_train_loss = 0.0
                running_bow_acc = 0.0
                running_gen_acc = 0.0
                wandb.log({"step": step, "epoch": epoch, "val_loss": val_loss, "val_accuracy": val_accuracy, "avg_loss": avg_loss, "avg_gen_acc": avg_gen_acc, "avg_bow_acc": avg_bow_acc})

            # Periodic model saving
            if step % steps_per_quarter_epoch == 0:
                torch.save(model.state_dict(), f'./ckpt/gpt2-medium_step_{step}.pth')

            step += 1

        avg_train_loss = train_loss / len(train_dataloader)
        print(f"Step {step}/{total_steps} - Avg Train Loss: {avg_train_loss}")


# Note: The function assumes that you have already defined an optimizer and a loss criterion.
# For example:
train_dataset = TXPairDataset(json_path='./data/train_20k.json', img_folder='./data/train_20k')
train_dataloader = DataLoader(train_dataset, batch_size=20, shuffle=False, collate_fn=collate_fn)
val_dataset = TXPairDataset(json_path='./data/val_200.json', img_folder='./data/val_200')
val_dataloader = DataLoader(val_dataset, batch_size=20, shuffle=False, collate_fn=collate_fn)
config = Config()
model = NanoNextGPT(config)

# Freeze GPT-2
for param in model.llm.parameters():
    param.requires_grad = False

# Unfreeze the linear layer
for param in model.input_projection.parameters():
    param.requires_grad = True

for param in model.layer_norm.parameters():
    param.requires_grad = True

# Assuming you have the total number of original tokens and the resized embedding layer
original_num_tokens = 50257  # Original GPT-2 vocab size
new_num_tokens = 50259  # Example new vocab size

# Freeze original embeddings
model.llm.transformer.wte.weight[:original_num_tokens].requires_grad = False

# Unfreeze new embeddings
model.llm.transformer.wte.weight[original_num_tokens:new_num_tokens].requires_grad = True

wandb.watch(model)
if args.load_pretrained:
    model.load_state_dict(torch.load('./ckpt/nano_nextgpt_med.pth'))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Example usage:
train_model(model, train_dataloader, val_dataloader, num_epochs=1)

