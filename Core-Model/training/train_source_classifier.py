import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import json
from train_core import ModelTrain
import os
from transformers import BertTokenizer, BertModel
from torch import nn
from torch import functional as F

# Filepaths
model_path = "./models/model_bin/program_classifier"
train_path = "./data/train_data.json"
val_path = "./data/val_data.json"
model_name = 'bert-base-uncased'
max_length = 500
batch_size = 3
num_epochs = 10
num_labels = 4

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)
bert_model = BertModel.from_pretrained(model_name, num_labels=num_labels)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

for param in bert_model.parameters():
    param.requires_grad = False


class CustomBert(nn.Module):
    def __init__(self, bert_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bert = bert_model
        self.linear1 = nn.Linear(768, 256)
        self.linear2 = nn.Linear(256, num_labels)
        self.relu = nn.ReLU()

    def forward(self, **kwargs):
        out = self.bert(**kwargs)
        x = self.relu(out['pooler_output'])
        x = self.relu(self.linear1(x))
        return self.linear2(x)


model = CustomBert(bert_model)


# for param in model.parameters():
#     param.requires_grad = False
# model.get_output_embeddings().weight.requires_grad = True


class CodeGenDataset(Dataset):
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_text = self.inputs[idx]
        output_text = self.outputs[idx]
        encoded_input = tokenizer(input_text, return_tensors="pt", max_length=max_length, truncation=True,
                                  padding="max_length")
        encoded_output = torch.zeros((4,))
        encoded_output[output_text] = 1
        return {"input": encoded_input,
                "output": encoded_output}


with open(train_path, 'r') as f:
    train_data = json.load(f)

inputs = [a['question'] for a in train_data]  # List of input sequences
outputs = [a['class'] for a in train_data]  # List of corresponding output sequences

dataset = CodeGenDataset(inputs, outputs)

with open(val_path, 'r') as f:
    val_data = json.load(f)

val_inputs = [a['question'] for a in val_data]
val_outputs = [a['class'] for a in val_data]

val_dataset = CodeGenDataset(val_inputs, val_outputs)


def input_parser(item):
    input_data = item['input']
    input_data['input_ids'] = torch.reshape(input_data['input_ids'],
                                            (input_data['input_ids'].shape[0], input_data['input_ids'].shape[-1]))
    input_data['token_type_ids'] = torch.reshape(input_data['token_type_ids'], (
        input_data['token_type_ids'].shape[0], input_data['token_type_ids'].shape[-1]))
    input_data['attention_mask'] = torch.reshape(input_data['attention_mask'], (
        input_data['attention_mask'].shape[0], input_data['attention_mask'].shape[-1]))

    # assert(False)
    return {
        'input_ids': input_data['input_ids'].to(device),
        'token_type_ids': input_data['token_type_ids'].to(device),
        'attention_mask': input_data['attention_mask'].to(device)
    }


def output_parser(output, item):
    return {
        "input": output,
        "target": item["output"].to(device)
    }


# Create an instance of the ModelTrain class
trainer = ModelTrain(tokenizer=tokenizer, model=model, dataset=dataset, model_path=model_path, device=device,
                     val_dataset=val_dataset)
trainer.input_parser = input_parser
trainer.output_parser = output_parser

trainer.train()
