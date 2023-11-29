import argparse
parser = argparse.ArgumentParser(description='Training settings')
parser.add_argument('--load_pretrained', action='store_true', help='Load pretrained model')
parser.add_argument('--train_instruct', action='store_true', help='Train the model on instruct dataset')
from torch.cuda.amp import autocast
args = parser.parse_args()

from model import NanoNextGPT, Config
from data_loading import *
from data_loading_instruct import *
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR
#from transformers import AdamW
import bitsandbytes as bnb
import datetime

from datasets import load_dataset

import wandb
wandb.init(project='nanonext', name='instruct_20k')

def train_model(model, train_dataloader, num_epochs=2):
    # Initialize optimizer
    #optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.001) #1e-4
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=1e-5, betas=(0.9, 0.995), is_paged=True)   
    warmup_ratio = 0.3
    if model.config.load_peft_model == True:
        model.llm.model.model.gradient_checkpointing_enable() # peft requires deeper call
    else:
        model.llm.model.gradient_checkpointing_enable()

    total_steps = len(train_dataloader) * num_epochs
    warmup_steps = int(warmup_ratio * total_steps)
    scheduler = OneCycleLR(optimizer, max_lr=1e-4, total_steps=total_steps, # 2e-3
                       pct_start=warmup_ratio, anneal_strategy='cos')
    steps_per_quarter_epoch = len(train_dataloader) // 4
    step = 0

    # Training loop converted to step-based
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        running_train_loss = 0.0
        running_bow_acc = 0.0
        running_gen_acc = 0.0
        perplexity_running = 0.0

        for batch in train_dataloader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            image_paths = batch['image_paths']
            labels = batch['labels']
            max_length = 1024  # for example
            input_ids = input_ids[:, :max_length - 1]
            labels = labels[:, :max_length]
            attention_mask = attention_mask[:, :max_length]
            #with autocast(dtype=torch.bfloat16):
            #print(f"input_ids[0].shape: ", input_ids[0].shape)
            #print(f"labels[0].shape: ", labels[0].shape)
            #print(f"input_ids[0]: ", input_ids[0])
            #print(f"labels[0]: ", labels[0])
            #print(f"attention_mask[0]: ", attention_mask[0])
            with autocast(dtype=torch.bfloat16):
                outputs = model.forward(input_ids, labels,  attention_mask, image_paths)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1) # 0.3
            optimizer.step()
            scheduler.step()
            
    
            train_loss += loss.item()
            running_train_loss += loss.item()
            
            perplexity = torch.exp(loss)
            #perplexity_running += perplexity

            # calculate the token accuracy
            chosen_tokens = torch.max(outputs.logits, dim=-1)[1][:, :-1]  # [B, S]
            labels = labels[:, 1:]
            gen_acc = (chosen_tokens.reshape(-1) == labels.reshape(-1)).to(torch.long)  # [B*S]
            valid_mask = (labels != -100) & (labels != 32002)
            valid_mask = valid_mask.reshape(-1)
            valid_tokens = gen_acc & valid_mask  # [B*S]
            gen_acc = valid_tokens.sum().item() / (valid_mask.sum().item() + 1.0)
            running_gen_acc += gen_acc

            # Convert tensors to sets for bag-of-words comparison
            #chosen_tokens_set = set(chosen_tokens.cpu().numpy().reshape(-1))
            #labels_set = set(labels.cpu().numpy().reshape(-1))

            # Remove -100 (used for padding or ignored tokens)
            #labels_set.discard(-100)
            #labels_set.discard(32002)
            #chosen_tokens_set.discard(32002)

            # Calculate bag-of-words accuracy
            #common_tokens = chosen_tokens_set.intersection(labels_set)
            #bow_acc = len(common_tokens) / len(labels_set) if len(labels_set) > 0 else 0.0
            #running_bow_acc += bow_acc
            #print(f"Step {step} - gen_acc: {gen_acc} - bow_acc: {bow_acc}")
            # Periodic quick validation
            if step >1 and step % 50 == 0:
                #val_loss, val_accuracy = quick_validate(model, val_dataloader)
                avg_loss = running_train_loss / 50  # Compute the average training loss
                avg_gen_acc = running_gen_acc / 50
                #avg_bow_acc = running_bow_acc / 50
                #avg_perplexity =  perplexity_running / 50
                print(f"Step {step}/{total_steps} - Avg loss: {avg_loss} - avg_gen_acc: {avg_gen_acc} - perplexity: {perplexity}")
                #print(f"Step {step}/{total_steps} - Avg loss: {avg_loss} - perplexity: {perplexity}")
                running_train_loss = 0.0
                running_bow_acc = 0.0
                running_gen_acc = 0.0
                #perplexity_running = 0.0
                wandb.log({"step": step, "epoch": epoch, "avg_loss": avg_loss, "avg_gen_acc": avg_gen_acc, "perplexity": perplexity})
                #wandb.log({"step": step, "epoch": epoch, "avg_loss": avg_loss, "perplexity": perplexity})
            if step > 1 and step % 1000 == 0:
                now = datetime.datetime.now()
                formatted_date_time = now.strftime("%Y-%m-%d")
                model.llm.save_pretrained(f"./adapters/instruct_77k_{step}")
                layers_to_save = ['input_projection']
                selected_state_dict = {k: v for k, v in model.state_dict().items() if k.split('.')[0] in layers_to_save}
                torch.save(selected_state_dict, f'./ckpt/instruct_77k_{step}.pth')
            #if step > 1 and step % 6000 == 0:
            #	break

            # Periodic model saving
            #if step % steps_per_quarter_epoch == 0:
            #    torch.save(model.state_dict(), f'./ckpt/imagebind_step_{step}.pth')

            step += 1
            #avg_loss = train_loss / step
            #print(f"step: {step} train_loss: {avg_loss}")

        avg_train_loss = train_loss / len(train_dataloader)
        print(f"Step {step}/{total_steps} - Avg Train Loss: {avg_train_loss}")
        now = datetime.datetime.now()
        formatted_date_time = now.strftime("%Y-%m-%d")
        if args.train_instruct:
            model.llm.save_pretrained(f"./adapters/instruct_77k_{step}")
        layers_to_save = ['input_projection']
        selected_state_dict = {k: v for k, v in model.state_dict().items() if k.split('.')[0] in layers_to_save}
        torch.save(selected_state_dict, f'./ckpt/instruct_77k_{step}.pth')
        
        #model.tokenizer.save_pretrained("./adapters/nanonext-111023_tokenizer")


config = Config()



#if args.load_pretrained:
#    model.load_state_dict(torch.load('./ckpt/nano_nextgpt_med.pth'))

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
	#for name, param in model.llm.named_parameters():
	#	print(name)
	#	print(param.requires_grad)
	#model.llm.print_trainable_parameters()
	#print(f"get_nb_trainable_parameters: ", model.llm.get_nb_trainable_parameters())
	
else: # train input proj from image bind --> LLM
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

#first_batch = next(iter(train_dataloader))

# Accessing the first example of the first batch
#first_example_input_ids = first_batch['input_ids'][0]
#first_example_label_ids = first_batch['labels'][0]
#first_example_mask = first_batch['attention_mask'][0]

#print(f"input_ids.shape: ", first_example_input_ids.shape)
#print(f"label_ids.shape: ", first_example_label_ids.shape)
#print(f"first_example_mask.shape: ", first_example_mask.shape)
#print(f"input_ids: ", first_example_input_ids)
#print(f"label_ids: ", first_example_label_ids)
#labels_set = set(first_example_label_ids.cpu().numpy().reshape(-1))

# Remove -100 (used for padding or ignored tokens)
#labels_set.discard(-100)
#labels_set.discard(32002)
#prompt = "<s>[INST] Can you describe the colors of the train? [/INST]The train is blue and pink in color. </s>"
#test_input_ids = model.tokenizer(prompt, add_special_tokens=False)['input_ids']
#test_input_ids_text = model.tokenizer.decode(test_input_ids, skip_special_tokens=False)
#label_ids_text = model.tokenizer.decode(first_example_label_ids, skip_special_tokens=False)
#print("test_input_ids:", test_input_ids)
#print("test_input_ids_text:", test_input_ids_text)
#print(f"first_example_mask: ", first_example_mask)
#for input_token, label_token in zip(first_example_input_ids[:], first_example_label_ids[:]):
#	print(f"Input Token: {input_token} => Next Token: {label_token}")
#for idx, label in enumerate(first_example_label_ids):
#	if label != -100 and label != first_example_input_ids[idx]:
#		print(f"pos: {idx}, label: {label}")
	
