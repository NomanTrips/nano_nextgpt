import torch
from torch.cuda.amp import autocast
from model import NanoNextGPT, Config
from data_loading import *
from data_loading_instruct import *
from torch.optim.lr_scheduler import OneCycleLR
import bitsandbytes as bnb
import argparse


parser = argparse.ArgumentParser(description='Training settings')
args = parser.parse_args()
parser.add_argument('--load_pretrained', action='store_true', help='Load pretrained model')
parser.add_argument('--train_instruct', action='store_true', help='Train the model on instruct dataset')

import wandb
wandb.init(project='nanonext', name='instruct_20k')

def train_model(model, train_dataloader, num_epochs=2):
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=1e-5, betas=(0.9, 0.995), is_paged=True)   
    warmup_ratio = 0.3
    if model.config.load_peft_model == True:
        model.llm.model.model.gradient_checkpointing_enable() # peft requires deeper call
    else:
        model.llm.model.gradient_checkpointing_enable()

    total_steps = len(train_dataloader) * num_epochs
    scheduler = OneCycleLR(optimizer, max_lr=1e-4, total_steps=total_steps, # 2e-3
                       pct_start=warmup_ratio, anneal_strategy='cos')
    step = 0

    # Training loop converted to step-based
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        running_train_loss = 0.0
        running_gen_acc = 0.0

        for batch in train_dataloader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            image_paths = batch['image_paths']
            labels = batch['labels']
            max_length = 1024  # so no cuda oom, adjust if gpu rich
            input_ids = input_ids[:, :max_length - 1]
            labels = labels[:, :max_length]
            attention_mask = attention_mask[:, :max_length]

            with autocast(dtype=torch.bfloat16): # bf16 more stable?
                outputs = model.forward(input_ids, labels,  attention_mask, image_paths)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1) # 0.3
            optimizer.step()
            scheduler.step()
            
    
            train_loss += loss.item()
            running_train_loss += loss.item()
            
            perplexity = torch.exp(loss)

            # calculate the token generation accuracy
            chosen_tokens = torch.max(outputs.logits, dim=-1)[1][:, :-1]  # [B, S]
            labels = labels[:, 1:]
            gen_acc = (chosen_tokens.reshape(-1) == labels.reshape(-1)).to(torch.long)  # [B*S]
            valid_mask = (labels != -100) & (labels != 32002) # exclude padding and human prompt from calc
            valid_mask = valid_mask.reshape(-1)
            valid_tokens = gen_acc & valid_mask  # [B*S]
            gen_acc = valid_tokens.sum().item() / (valid_mask.sum().item() + 1.0)
            running_gen_acc += gen_acc

            # Periodic quick validation
            if step >1 and step % 50 == 0:
                avg_loss = running_train_loss / 50
                avg_gen_acc = running_gen_acc / 50
                print(f"Step {step}/{total_steps} - Avg loss: {avg_loss} - avg_gen_acc: {avg_gen_acc} - perplexity: {perplexity}")
                running_train_loss = 0.0
                running_gen_acc = 0.0
                wandb.log({"step": step, "epoch": epoch, "avg_loss": avg_loss, "avg_gen_acc": avg_gen_acc, "perplexity": perplexity})
            if step > 1 and step % 1000 == 0:
                model.llm.save_pretrained(f"./adapters/instruct_77k_{step}")
                layers_to_save = ['input_projection']
                selected_state_dict = {k: v for k, v in model.state_dict().items() if k.split('.')[0] in layers_to_save}
                torch.save(selected_state_dict, f'./ckpt/instruct_77k_{step}.pth')

            step += 1

        avg_train_loss = train_loss / len(train_dataloader)
        print(f"Step {step}/{total_steps} - Avg Train Loss: {avg_train_loss}")
        if args.train_instruct:
            model.llm.save_pretrained(f"./adapters/instruct_77k_{step}")
        layers_to_save = ['input_projection']
        selected_state_dict = {k: v for k, v in model.state_dict().items() if k.split('.')[0] in layers_to_save}
        torch.save(selected_state_dict, f'./ckpt/instruct_77k_{step}.pth')

config = Config()

if args.train_instruct:
    train_dataset = TXPairDatasetInstruct(json_path='./data/instruct.json', img_folder='./data/instruct')
    train_dataloader = DataLoader(train_dataset, batch_size=6, shuffle=False, collate_fn=collate_fn_instruct) 
    config.load_peft_model = True
    model = NanoNextGPT(config)
    model.load_state_dict(torch.load('./ckpt/inputproj_trained.pth'), strict=False)
    for param in model.input_projection.parameters():
        param.requires_grad = True
    for param in model.image_encoder.parameters():
        param.requires_grad = False
else: # train 1 layer nn input proj to map image bind embedding space to LLM embedding space
    train_dataset = TXPairDataset(json_path='./data/train_20k.json', img_folder='./data/train_20k')
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
    config.load_peft_model = False
    model = NanoNextGPT(config)
    for param in model.llm.parameters():
        param.requires_grad = False
    for param in model.input_projection.parameters():
        param.requires_grad = True
    for param in model.image_encoder.parameters():
        param.requires_grad = False

wandb.watch(model)
train_model(model, train_dataloader, num_epochs=1)
	
