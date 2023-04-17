from torch.utils.data import TensorDataset, RandomSampler, DataLoader
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from torch.profiler import profile, record_function, ProfilerActivity


class ModelTrain:
    def __init__(self, tokenizer, model, dataset, model_path, device, num_epochs=5, lr=1e-1, batch_size=10,
                 val_dataset=None):
        self.device = device
        self.tokenizer = tokenizer
        self.model = model.to(self.device)
        self.train_data = dataset
        self.train_sampler = RandomSampler(self.train_data)
        self.train_dataloader = DataLoader(self.train_data, sampler=self.train_sampler, batch_size=batch_size)
        self.val_data = val_dataset
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size)
        self.input_parser = lambda i: {"x": i}
        self.output_parser = lambda a, b: {"input": a, "target": b}
        self.num_epochs = num_epochs
        self.lr = lr
        self.model_path = model_path
        self.batch_size = batch_size

    def set_input_parser(self, new_parser):
        self.input_parser = new_parser

    def profile_GPU(self, location):
        t = torch.cuda.get_device_properties(0).total_memory / 1e9
        r = torch.cuda.memory_reserved(0) / 1e9
        a = torch.cuda.memory_allocated(0) / 1e9

        print(f"{location} \t MEM RESERVED: {r:.02f} \t MEM ALLOCATED: {a:.02f}")

    def set_output_parser(self, new_parser):
        self.output_parser = new_parser

    def class_accuracy(self):
        num_correct = 0

        self.model.eval()
        with torch.no_grad():
            for batch in self.val_loader:
                y_pred = self.model(**self.input_parser(batch))
                # y_pred = y_pred.to(torch.float64)
                # y_pred = torch.softmax(y_pred, dim=-1)
                y_pred = y_pred.argmax(axis=1)
                y_true = batch['output'].to(self.device).argmax(axis=1)

                # print(y_pred)
                # print(y_true)
                # print(y_true == y_pred)
                # print((y_true == y_pred).sum().item())
                # assert (False)

                num_correct += (y_true == y_pred).sum().item()


        print(f"VAL ACCURACY: {(num_correct/len(self.val_data)) * 100:.02f}")
    def loss_accuracy(self, loss_fxn):
        val_loss = 0
        self.model.eval()
        with torch.no_grad():
            for batch in self.val_loader:
                outputs = self.model(**self.input_parser(batch))
                loss = loss_fxn(**self.output_parser(outputs, batch))
                val_loss += loss.item()
                val_loss /= len(self.val_loader)

        print(f"Validation loss: {val_loss}")

    def train(self):
        optimizer = Adam(self.model.parameters(), lr=self.lr)
        num_training_steps = self.num_epochs * len(self.train_dataloader)
        # lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
        #                                                num_training_steps=num_training_steps)
        scheduler = ExponentialLR(optimizer, gamma=0.9)
        loss_fn = torch.nn.CrossEntropyLoss()

        for epoch in range(self.num_epochs):
            print(f"------EPOCH NUMBER {epoch}")
            train_loss = 0
            self.model.train()
            for batch in tqdm(self.train_dataloader):
                model_inputs = self.input_parser(batch)

                optimizer.zero_grad()

                outputs = self.model(**model_inputs)

                model_outputs = self.output_parser(outputs, batch)

                loss = loss_fn(**model_outputs)

                train_loss += loss.item()

                loss.backward()

                optimizer.step()
                # loss = loss.detach()
                # del loss
                # del outputs
                # del model_outputs
                # del model_inputs
                torch.cuda.empty_cache()


            scheduler.step()
            train_loss /= len(self.train_dataloader)
            print(f"Train loss: {train_loss}")
            self.class_accuracy()

            torch.save(self.model.state_dict(), self.model_path)
            self.model.bert.save_pretrained('./models/model_bin/roberta_token_classifier')