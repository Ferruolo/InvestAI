import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW
from torch.utils.data import Dataset, DataLoader
import json
from train_core import ModelTrain
# from training.train_core import ModelTrain
import os

from transformers import GPT2LMHeadModel, GPT2TokenizerFast


# Filepaths
model_path = "./models/model_bin/code_generator"
train_path = "./data/train_data.json"
val_path = "./data/val_data.json"
max_length = 500
batch_size = 3
num_epochs = 15

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

# Load the tokenizer and model

# model_name = "'SIC98/GPT2-python-code-generator'"
# tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
# model = GPT2LMHeadModel.from_pretrained(model_name)

model_name = "Salesforce/codegen-350M-mono"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, pad_token_id=0)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})



# Define a custom dataset
class CodeGenDataset(Dataset):
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_text = self.inputs[idx]
        output_text = self.outputs[idx]
        encoded_input = tokenizer.encode_plus(input_text, padding="max_length", truncation=True,
                                              return_tensors="pt", max_length=max_length)
        encoded_output = tokenizer.encode_plus(output_text, padding="max_length", truncation=True,
                                               return_tensors="pt", max_length=max_length)
        return {"input_ids": encoded_input["input_ids"], "attention_mask": encoded_input["attention_mask"],
                "decoder_input_ids": encoded_output["input_ids"],
                "decoder_attention_mask": encoded_output["attention_mask"]}


with open(train_path, 'r') as f:
    train_data = json.load(f)

inputs = [a['question'] for a in train_data]  # List of input sequences
outputs = [a['program'] for a in train_data]  # List of corresponding output sequences

dataset = CodeGenDataset(inputs, outputs)

with open(val_path, 'r') as f:
    val_data = json.load(f)

val_inputs = [a['question'] for a in val_data]
val_outputs = [a['program'] for a in val_data]

val_dataset = CodeGenDataset(val_inputs, val_outputs)


def input_parser(item):
    return {
        "input_ids": item["input_ids"].to(device),
        "attention_mask": item["attention_mask"].to(device),
        # "decoder_input_ids": item["decoder_input_ids"].to(device),
        # "decoder_attention_mask": item["decoder_attention_mask"].to(device)
    }


def output_parser(output, item):
    return {
        "input": output.logits.view(-1, output.logits.shape[-1]),
        "target": item["decoder_input_ids"].to(device).view(-1)
    }


trainer = ModelTrain(tokenizer, model, dataset, model_path, device, val_dataset=val_dataset, batch_size=batch_size, num_epochs=num_epochs)
trainer.input_parser = input_parser
trainer.output_parser = output_parser

# Freeze all model parameters except the final layer
num_layers = 0
for param in model.parameters():
    num_layers += 1



for i in range(num_layers, num_layers - 60, -10):
    print(f"Training Layers {i - 10} through {i}")
    lay_num = 0
    for param in model.parameters():
        lay_num += 1
        if lay_num < i - 10 or lay_num > i:
            param.requires_grad = False
        else:
            param.requires_grad = True
    model.get_output_embeddings().weight.requires_grad = True



    trainer.train()
