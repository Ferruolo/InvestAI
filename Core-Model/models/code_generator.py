from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class ProgramGenerator:
    def __init__(self):
        self.model_name = "Salesforce/codegen-16B-multi"
        self.model_path = "./models/model_bin/code_generator"
        self.max_length = 500
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, pad_token_id=0).to(self.device)
        self.model.eval()

    def forward(self, text):
        encoded_input = self.tokenizer.encode_plus(text, padding="max_length", truncation=True,
                                                   return_tensors="pt", max_length=self.max_length)

        output = self.model.generate(input_ids=encoded_input["input_ids"].to(self.device),
                                     attention_mask=encoded_input["attention_mask"].to(self.device),
                                     max_new_tokens=self.max_length)

        return self.tokenizer.decode(output[0].to('cpu'), skip_special_tokens=False)


