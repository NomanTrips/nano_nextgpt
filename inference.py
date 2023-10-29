import torch
from PIL import Image
from transformers import GPT2Tokenizer
from model import NanoNextGPT, Config
import torchvision.transforms as transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

config = Config()
model = NanoNextGPT(config)
model.load_state_dict(torch.load('C:/Users/Brian/Desktop/nano_nextgpt/ckpt/nano_nextgpt_med.pth'))

# Initialize GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
tokenizer.add_tokens(["<Img>", "</Img>"])
prompt = "<Img></Img>caption: "

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# Load and transform the image
img_path = "C:/Users/Brian/Desktop/cc3m_mini/images/train/sandcastle.jpg"
image = Image.open(img_path).convert('RGB')
image = transform(image).to(device)
image = image.unsqueeze(0)

input_ids = tokenizer.encode(prompt)
input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)

max_new_tokens = 10

print(f"input_ids.shape: ", input_ids.shape)
print(f"image.shape: ", image.shape)

# Loop to generate text
with torch.no_grad():
    for _ in range(max_new_tokens):
        outputs = model.forward(input_ids=input_ids, labels=None, attention_mask=None, image_tensor=image)
        logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)
        input_ids = torch.cat((input_ids, next_token), dim=1)

# Decode and print the generated text
generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
print("Generated text:", generated_text)