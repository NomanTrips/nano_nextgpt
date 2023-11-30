import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, PeftModel
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
llama_2_path = "/home/brian/Desktop/Llama-2-7b-chat-hf/"


################################################################################
# QLoRA parameters
################################################################################

# LoRA attention dimension
lora_r = 64

# Alpha parameter for LoRA scaling
lora_alpha = 16

# Dropout probability for LoRA layers
lora_dropout = 0.1

################################################################################
# bitsandbytes parameters
################################################################################

# Activate 4-bit precision base model loading
use_4bit = True

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False

# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    #load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

#device_map = {"": 0}
device_map={"": "cpu"}

llama_model = AutoModelForCausalLM.from_pretrained(
	llama_2_path,
	quantization_config=bnb_config,
	device_map=device_map,
	torch_dtype=torch.float16
)
llama_model.config.use_cache = False
llama_model.config.pretraining_tp = 1

adapters_path = "./adapters/bf16_13334/nano_nextgpt" # set to finetuned adapter path
model = PeftModel.from_pretrained(llama_model, adapters_path, torch_dtype=torch.float16, device_map=device_map)
model = model.merge_and_unload()

model.save_pretrained("./ckpt/bf16_13334") # set to save path where you want the .pth