import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import json
from train_core import ModelTrain
import os
from transformers import BertTokenizer, BertModel, BertConfig
from torch import nn
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch import functional as F

# Filepaths
model_path = "./models/model_bin/program_classifier.pt"
train_path = "./data/ticker_dataset/train_data.json"
val_path = "./data/ticker_dataset/val_data.json"
model_name = 'dslim/bert-base-NER'
max_length = 20
batch_size = 100
num_epochs = 5
num_labels = 503

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)


# model_name = 'roberta-base'
# tokenizer = RobertaTokenizer.from_pretrained(model_name)
# bert_model = RobertaForSequenceClassification.from_pretrained(model_name)
#


bert_model = BertModel.from_pretrained(model_name, num_labels=num_labels)

tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# num_layers = 0
# for param in bert_model.parameters():
#     num_layers += 1
# lay_num = 0
# for param in bert_model.parameters():
#     lay_num += 1
#     if lay_num < num_layers:
#         param.requires_grad = False
#     else:
#         param.requires_grad = True
#
# for param in bert_model.parameters():
#     param.requires_grad = False


class CustomBert(nn.Module):
    def __init__(self, bert_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bert = bert_model
        # self.linear1 = nn.Linear(768, 768)
        # self.linear2 = nn.Linear(768, 635)
        self.linear3 = nn.Linear(768, num_labels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, **kwargs):
        out = self.bert(**kwargs)
        x = out['pooler_output']
        x = self.dropout(x)
        # x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        # x = self.softmax(x)
        return x


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
        encoded_output = torch.zeros((num_labels,))
        encoded_output[output_text] = 1
        return {"input": encoded_input,
                "output": encoded_output}


with open(train_path, 'r') as f:
    train_data = json.load(f)

train_limit = 10000

inputs = [a['SENTENCE'] for a in train_data[:train_limit]]  # List of input sequences
outputs = [a['TICKER'] for a in train_data[:train_limit]]  # List of corresponding output sequences
del train_data
dataset = CodeGenDataset(inputs, outputs)

with open(val_path, 'r') as f:
    val_data = json.load(f)
val_limit = 1000

val_inputs = [a['SENTENCE'] for a in val_data[:val_limit]]
val_outputs = [a['TICKER'] for a in val_data[:val_limit]]
del val_data
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
        # 'token_type_ids': input_data['token_type_ids'].to(device),
        'attention_mask': input_data['attention_mask'].to(device)
    }


def output_parser(output, item):
    # print(output.shape)
    # print(item['output'].shape)
    return {
        "input": output,
        "target": item["output"].to(device)
    }


# Create an instance of the ModelTrain class
trainer = ModelTrain(tokenizer=tokenizer, model=model, dataset=dataset, model_path=model_path, device=device,
                     val_dataset=val_dataset, batch_size=batch_size)
trainer.input_parser = input_parser
trainer.output_parser = output_parser

trainer.train()
