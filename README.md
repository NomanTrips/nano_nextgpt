# NextGPT Re-Implementation

## Overview
This project is a re-implementation of NextGPT focusing solely on image and text modalities. It has been trained on a single RTX 3090 in a matter of a few hours. The model is capable of performing simple visual question answering tasks and is intended primarily for learning and experimentation purposes.

## About NextGPT
NextGPT ([Official Website](https://next-gpt.github.io/)) operates by encoding images, text, audio, and video into a unified embedding space through a meta AI model called ImageBind. A linear layer is then used to map these embeddings to a larger LLM embedding space. This re-implementation, however, does not include the output aspect of NextGPT, which involves using the output of the LLM for various modalities like image generation.

## Limitations
As per insights from CogVLM ([Research Paper](https://arxiv.org/pdf/2311.03079.pdf)), the image embedding in this model occupies a minor portion of the input, limiting its depth of understanding. However, there's potential for deeper representation, making the image data more accessible to the model.

## Environment Setup
- **Operating System**: Ubuntu (required for bitsandbytes library)
- **GPU**: RTX 3090
- **Memory**: 48 GB (necessary for handling weights)

## Installation
pip install -r requirements.txt
- Download the Llama 2 chat HF weights and place them in a designated folder
- Alternatively, download the pretrained model weights
- Edit the 'llama_2_path' variable in model.py and train.py to point to this folder
- Set the tokenizer variable similarly
- Repeat these steps in inference.py if using it

## Usage:
python inference.py
- Edit the image file name and prompt in inference.py (Make this better on todo list)
Training:
- For stage 1: training the linear layer:
- python train.py
- For stage 2: instruction training the whole model:
python train.py --train_instruct

## Data:
- The data is basically from LLava which is just COCO 3m subsets with GPT-4 enhancing the textual portion and making it conversational. Basically any of the instruct.json data sets can be used. (set these in train.py at the bottom). DataLoading for stage 1 and DataLoadingInstruct for stage 2.
- https://huggingface.co/datasets/jubba/nano_nextgpt_instruct
- 'train_20k': this is for stage 1, training the linear layer to map from image bind embed to LLM embed
- 'instruct': stage 2: end to end conversational data involving Q/A on images

## Weights:
- https://huggingface.co/jubba/nano_nextgpt