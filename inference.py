import torch
from model import NanoNextGPT, Config
from transformers import AutoTokenizer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
llama_2_path = "/home/brian/Desktop/Llama-2-7b-chat-hf/"

# Load the state dictionary from the file
state_dict = torch.load('./ckpt/bf16_13334.pth')

desired_keys = ["input_projection.weight", "input_projection.bias"]

selected_layers = {k: v for k, v in state_dict.items() if k in desired_keys}

print(f"selected_layers.len: ", len(selected_layers))
 
config = Config()
model = NanoNextGPT(config)
# Now, load the selected layers into the model
model.load_state_dict(selected_layers, strict=False)

# Initialize Llama 2 tokenizer
tokenizer = AutoTokenizer.from_pretrained(llama_2_path, trust_remote_code=True)
new_tokens = ["<Img>", "</Img>"]
tokenizer.add_tokens(new_tokens)
tokenizer.add_special_tokens({"pad_token":"<pad>"})
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

prompt = "<s>[INST] What animal is in the drawing? <Img> </Img> [/INST]"

# Load and transform the image
image_paths = []
image_paths.append("./data/val_200/GCC_train_002576676.jpg")

input_ids = tokenizer(prompt, add_special_tokens=False)['input_ids']
input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)

max_new_tokens = 48

print(f"input_ids.shape: ", input_ids.shape)
print(f"input_ids: ", input_ids)

# Loop to generate text
with torch.no_grad():
    for _ in range(max_new_tokens):
        outputs = model.forward(input_ids=input_ids, labels=None, attention_mask=None, image_paths=image_paths)
        logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)
        input_ids = torch.cat((input_ids, next_token), dim=1)

# Decode and print the generated text
generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=False)
print("Generated text:", generated_text)