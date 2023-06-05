
from torch.nn.utils.rnn import pad_sequence
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.cuda.amp import autocast
from torch import nn
from transformers import AdamW, get_cosine_schedule_with_warmup
from torch.utils.checkpoint import checkpoint
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
import csv
from torch import Tensor, nn, FloatTensor
from typing import List, Optional, Set, Tuple, Union
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from lion_pytorch import Lion
from transformers.modeling_outputs import ModelOutput
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
import unicodedata
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from transformers import AutoModelWithLMHead, AutoConfig
from transformers import T5Tokenizer
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, PreTrainedModel
from torch.optim import AdamW
import json
import random

import transformers

CUDA_DEBUGGING = True


class QADataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cash = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            self.tokenizer.sep_token_id = 5
            d = self.data[idx]
            instruction = d["instruction"]
            instruction = str.lower(instruction)
            in_ = d["context"]
            in_ = str.lower(in_)
            out = d["response"]
            out = str.lower(out)
            question = f"///instruction//{instruction}[SEP]///context//{in_}[SEP]"
            answer = f"{out}</s>"

            input = self.tokenizer.encode_plus(question, truncation=True,
                                               max_length=1024-128, padding=False, return_tensors="pt")  # Do not pad here
            target_ids = self.tokenizer.encode_plus(answer, truncation=True,
                                                    max_length=1024-128, padding=False, return_tensors="pt")["input_ids"]  # Do not pad here

            input_ids = input["input_ids"]
            attention_mask = input["attention_mask"]

            return {"input_ids": input_ids[0], "attention_mask": attention_mask[0], "labels": target_ids[0], "text": question}
        except Exception as e:
            print(e)
            return {"input_ids": torch.zeros(self.max_length).int(), "attention_mask": torch.zeros(self.max_length).int(), "labels": torch.zeros(self.max_length).int()}


class MyConv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        nf (`int`): The number of output features.
        nx (`int`): The number of input features.
    """

    def __init__(self,  w):
        super().__init__()
        size = w.weight.size()
        self.nf = size[1]
        nf = self.nf
        self.weight = nn.Parameter(torch.empty(size[0], nf))
        self.bias = nn.Parameter(torch.zeros(nf))
        self.b = nn.Parameter(torch.zeros(nf))
        self.weight.requires_grad = False
        self.bias.requires_grad = True
        self.u = nn.Parameter(torch.zeros((size[0], 6)))
        self.v = nn.Parameter(torch.zeros((6, size[1])))
        nn.init.normal_(self.weight, std=0.02)
        nn.init.normal_(self.u, std=0.01)
        nn.init.normal_(self.v, std=0.01)
        nn.init.normal_(self.b, std=0.01)

    def setup(self, w):
        self.weight = w.weight
        self.bias = w.bias
        self.weight.requires_grad = False
        self.u.requires_grad = True
        self.v.requires_grad = True
        self.b.requires_grad = True
        self.bias.requires_grad = False
        return self

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        z = torch.addmm(self.bias, x.view(-1, x.size(-1)),
                        self.weight)
        y = torch.addmm(self.b, x.view(-1, x.size(-1)),
                        self.u@self.v)
        x = (z.detach()+y).view(size_out)

        return x


class LoraLayer(GPT2Block):
    @staticmethod
    def set(layer: nn.Module):
        size = layer.attn.c_attn.weight.size()
        layer.attn.c_attn = MyConv1D(
            layer.attn.c_attn).setup(layer.attn.c_attn)

        size = layer.attn.c_proj.weight.size()
        layer.attn.c_proj = MyConv1D(
            layer.attn.c_proj).setup(layer.attn.c_proj)

        size = layer.mlp.c_fc .weight.size()
        layer.mlp.c_fc = MyConv1D(layer.mlp.c_fc).setup(layer.mlp.c_fc)

        size = layer.mlp.c_proj .weight.size()
        layer.mlp.c_proj = MyConv1D(layer.mlp.c_proj).setup(layer.mlp.c_proj)

        return layer


class LoraManagerbase(AutoModelWithLMHead):
    def __init__(self, config, rank):
        super().__init__(config=config)

    @classmethod
    def SetUp(self, model, rank, device):
        model.base_model.h = nn.ModuleList(
            [LoraLayer.set(layer.to(device))
             for layer in model.base_model.h]
        )

        torch.cuda.empty_cache()
        model.base_model.ln_f = model.base_model.ln_f .to(device)
        model.base_model.wte = model.base_model.wte.to(device)
        model.base_model.wpe = model.base_model.wpe.to(device)
        model.base_model.drop = model.base_model.drop
        model.base_model.ln_f.requires_grad = True
        model.base_model.wte.requires_grad = True
        model.base_model.wpe.requires_grad = True
        return model

    @ classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, device, *model_args, **kwargs):
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        rank = kwargs.pop("rank")
        model = super(LoraManagerbase, cls).from_pretrained(
            pretrained_model_name_or_path, config=config, *model_args, **kwargs)
        model = LoraManagerbase. SetUp(model, rank, device)

        model.rank = rank
        return model

    @ classmethod
    def from_config(cls, config, *model_args, **kwargs):
        rank = kwargs.pop("rank")
        model = super(LoraManagerbase, cls).from_config(
            config, *model_args, **kwargs)
        model.__class__ = cls
        model.rank = rank
        return model


class LoraTrainer:

    def __init__(self,   rank):
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            'rinna/japanese-gpt2-small', use_fast=False)
        # データを整形
        with open('databricks-dolly-15k-translated.json', 'r', encoding="utf-8") as f:
            data = json.load(f)

        # unicodeエスケープを解除
        da = []
        for d in data:
            p = {}

            for k in d:
                p[k] = unicodedata.normalize('NFKC', d[k])
            da.append(p)
        random.shuffle(da)
        # 保存
        with open('databricks-dolly-15k-translated.json', 'w', encoding="utf-8") as f:
            json.dump(da, f, ensure_ascii=False, indent=4)
        self.tokenizer = tokenizer

        # data を128の倍数に切り捨て
        data = data[:len(data)//128*128]
        # 重複削除
        dataset = QADataset(random.sample(data, len(data)),
                            self.tokenizer, max_length=1024)

# corate
        def collate_fn(batch):
            input_ids = [item['input_ids'] for item in batch]
            attention_mask = [item['attention_mask'] for item in batch]
            labels = [item['labels'] for item in batch]

            max_len = max(max(len(ids) for ids in input_ids),
                          max(len(lbl) for lbl in labels))

            padded_input_ids = []
            padded_attention_mask = []
            padded_labels = []

            for ids, mask, lbl in zip(input_ids, attention_mask, labels):
                pad_input_ids = torch.cat(
                    (ids, torch.tensor([tokenizer.pad_token_id] * (max_len - len(ids)))), dim=0)
                pad_attention_mask = torch.cat(
                    (mask, torch.tensor([0] * (max_len - len(mask)))), dim=0)
                pad_labels = torch.cat(
                    (lbl, torch.tensor([tokenizer.pad_token_id] * (max_len - len(lbl)))), dim=0)

                padded_input_ids.append(pad_input_ids)
                padded_attention_mask.append(pad_attention_mask)
                padded_labels.append(pad_labels)

            return {
                'input_ids': torch.stack(padded_input_ids).long(),
                'attention_mask': torch.stack(padded_attention_mask).long(),
                'labels': torch.stack(padded_labels).long(),
            }

        self.dataloader = DataLoader(
            dataset, batch_size=1,   shuffle=True, collate_fn=collate_fn)

        torch.cuda.empty_cache()
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = GPT2LMHeadModel.from_pretrained(
            'rinna/japanese-gpt2-medium').to(self.device)
        token = tokenizer.encode("こんにちは", return_tensors="pt").to(self.device)
        res = self.model.generate(token, max_length=1024, do_sample=True,  top_k=500, top_p=0.8,
                                  num_return_sequences=3, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        print(tokenizer.decode(res[0]))
        print("------------------------")
        print(tokenizer.decode(res[1]))
        print("------------------------")
        print(tokenizer.decode(res[2]))
        print("------------------------")

        print()
        print()

        self.    model = LoraManagerbase.from_pretrained(
            'rinna/japanese-gpt2-medium', self. device, rank=rank).to(self.device).bfloat16()
        self.model.save_pretrained("test")

        res = self.model.generate(token, max_length=1024, do_sample=True,  top_k=500, top_p=0.8,
                                  num_return_sequences=3, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        print(tokenizer.decode(res[0]))
        print("------------------------")
        print(tokenizer.decode(res[1]))
        print("------------------------")
        print(tokenizer.decode(res[2]))
        print("------------------------")

    def train(self, epochs, criterion):
        # save
        epoch = 0
        torch.cuda.empty_cache()

        step = 0

        low_path_loss = 0
      #  self.lora_manager.save_pretrained(f"model_epoch_{epoch}")

        b_Step = 0
        b_Step = b_Step+1

        # re shuffle to self.dataloader
        torch.cuda.empty_cache()
        loss_sum = 0
        a_rate = 16
        self.optimizer = Lion(
            self.model.parameters(), lr=1e-4, weight_decay=1e-4)
        # dataloaderは訓練データのDataLoaderです

        self.optimizer.zero_grad()
        h_ = len(self.model.base_model.h)
        self.model.train()
        for epoch in range(epochs):
            tq = tqdm(self.dataloader)

            for data in tq:
                # 学習の処理
                inputs, labels = data["input_ids"].to(
                    self.device), data["labels"].to(self.device)
                attention_mask = data["attention_mask"].to(self.device)
                inputs = inputs
                labels = labels
                attention_mask = attention_mask
                # with autocast(device_type="cuda"):
                past_key_values = None
                torch.cuda.empty_cache()

                lora_outputs = self. model(
                    inputs, attention_mask=attention_mask, labels=labels)

                torch.cuda.empty_cache()

                logits = lora_outputs["logits"]
            #    loss = criterion(logits, labels, inputs,
             #                    attention_mask).mean()/a_rate
                loss = masked_cross_entropy(
                    logits, labels,  attention_mask)/a_rate

                torch.cuda.empty_cache()
               # loss = lora_outputs["loss"]/a_rate
                loss.backward()
                loss_sum += loss
                torch.cuda.empty_cache()
                if (step+1) % a_rate == 0:

                    torch.cuda.empty_cache()
                    self.optimizer.step()
                    torch.cuda.empty_cache()
                    self.optimizer.zero_grad()

                if (step+1) % (4*64) == 0:
                    torch.cuda.empty_cache()

                    inputs = inputs[0].squeeze(0).unsqueeze(0)
                    # paddingを削除
                    inputs = inputs[inputs != 3].unsqueeze(0)

                    with torch.no_grad():
                        token = self.model.generate(
                            inputs, max_length=1024, do_sample=True,  top_k=500, num_return_sequences=3, no_repeat_ngram_size=3,  num_beams=3, pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id)
                    print("--------------------")
                    print(self.tokenizer.decode(token[0]))
                    print("--------------------")
                    print(self.tokenizer.decode(token[1]))
                    print("--------------------")
                    print(self.tokenizer.decode(token[2]))
                    print("--------------------")

                tq.set_description(
                    f"step {step} loss: {loss.item()*a_rate:.5f}")
                # save to csv
                path = "loss.csv"
                if not os.path.exists(path):
                    with open(path, 'w') as f:
                        f.write(f"{loss*a_rate }\n")
                else:
                    with open(path, 'a') as f:
                        f.write(f"{loss*a_rate }\n")

                step += 1
                torch.cuda.empty_cache()

            print(f"epoch {epoch} loss: {loss_sum/step :.5f}")
            # save
            self.model.save_pretrained(f"model_train_epoch_{epoch}")
            torch.cuda.empty_cache()


def masked_cross_entropy(logits, target, attention_mask):
    loss_fct = nn.CrossEntropyLoss(
        reduction='mean', ignore_index=3, label_smoothing=0.1)  # 'none'で各要素の損失を計算
    loss = loss_fct(logits.view(-1, logits.size(-1)), target.view(-1))

    return loss.mean()


def main():

    criterion = int()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    epochs = 100

    trainer = LoraTrainer(4)

    trainer.train(epochs, criterion)
    print("Training completed")


if __name__ == "__main__":
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        'rinna/japanese-gpt2-medium', use_fast=False)

    main()
