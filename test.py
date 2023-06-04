
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
        for d in tqdm(range(len(self.data))):
            self.cash.append(self.get(d))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.cash[idx]

    def get(self, idx):
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

            # self.tokenizer.pad_token_idv

            input = self.tokenizer.encode_plus(question, truncation=True,
                                               max_length=1024-128, padding="max_length", return_tensors="pt")
            target_ids = self. tokenizer.encode_plus(answer, truncation=True,
                                                     max_length=1024-128, padding="max_length", return_tensors="pt")["input_ids"].squeeze(0)

            input_ids = input["input_ids"] .squeeze(0)
            attention_mask = input["attention_mask"].squeeze(0)
            # eosを削除

            return {"input_ids": input_ids[0], "attention_mask": attention_mask, "labels": target_ids[0], "text": question}
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
        self.u = nn.Parameter(torch.zeros((size[0], 4)))
        self.v = nn.Parameter(torch.zeros((4, size[1])))
        nn.init.normal_(self.weight, std=0.02)
        nn.init.normal_(self.u, std=0.5)
        nn.init.normal_(self.v, std=0.5)
        nn.init.normal_(self.b, std=1)

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
        x = torch.addmm(self.bias.detach()+self.b, x.view(-1, x.size(-1)),
                        self.weight.detach()+self.u@self.v)
        x = x.view(size_out)

        self.u.data.clamp_(-1, 1)
        self.v.data.clamp_(-1, 1)
        self.b.data.clamp_(-10, 10)

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

        torch.cuda.empty_cache()
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = GPT2LMHeadModel.from_pretrained(
            'rinna/japanese-gpt2-small').to(self.device)
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
            'rinna/japanese-gpt2-small', self. device, rank=rank).to(self.device).bfloat16()
        self.model.save_pretrained("test")

        res = self.model.generate(token, max_length=1024, do_sample=True,  top_k=500, top_p=0.8,
                                  num_return_sequences=3, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        print(tokenizer.decode(res[0]))
        print("------------------------")
        print(tokenizer.decode(res[1]))
        print("------------------------")
        print(tokenizer.decode(res[2]))
        print("------------------------")

        self.tokenizer = tokenizer

        self.tokenizer.sep_token = self.tokenizer.eos_token

        # データを整形
        with open('rinnna_loader/databricks-dolly-15k-translated14801_14900.json', 'r', encoding="utf-8") as f:
            data = json.load(f)

        # unicodeエスケープを解除
        da = []
        for d in data:
            p = {}

            for k in d:
                p[k] = unicodedata.normalize('NFKC', d[k])
            da.append(p)
        # 保存
        with open('databricks-dolly-15k-translated.json', 'w', encoding="utf-8") as f:
            json.dump(da, f, ensure_ascii=False, indent=4)

        # data を128の倍数に切り捨て
        data = data[:len(data)//128*128]

        # 重複削除
        dataset = QADataset(random.sample(data, len(data)),
                            self.tokenizer, max_length=1024)

        self.dataloader = DataLoader(dataset, batch_size=2,   shuffle=True)

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
        a_rate = 4
        self.optimizer = Lion(
            self.model.parameters(), lr=6e-4, weight_decay=1e-3)
        # dataloaderは訓練データのDataLoaderです
        scheduler = lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.8, patience=10)

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
                    scheduler.step(loss_sum)

                if (step+1) % (4*64) == 0 or step == 0:

                    inputs = inputs[0].squeeze(0).unsqueeze(0)
                    # paddingを削除
                    inputs = inputs[inputs != 3].unsqueeze(0)

                    with torch.no_grad():
                        token = self.model.generate(
                            inputs, max_length=1024, do_sample=True,  top_k=50, top_p=0.8, num_return_sequences=3, no_repeat_ngram_size=4,  num_beams=3, pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id)
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
    loss_fct = nn.CrossEntropyLoss(reduction='none')  # 'none'で各要素の損失を計算
    loss = loss_fct(logits.view(-1, logits.size(-1)), target.view(-1))
    mask = attention_mask.view(-1)
    loss = loss * mask  # アテンションマスクを適用
    return loss.sum() / mask.sum()  # マスクを適用した部分だけで平均を取る


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=-100, reduce=True, attn_mask=None):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)

    # Apply attention mask
    if attn_mask is not None:
        # Ensure the mask is of type bool and expand dimensions
        attn_mask = attn_mask.unsqueeze(-1).bool()
        nll_loss.masked_fill_(~attn_mask, 0.0)
        smooth_loss.masked_fill_(~attn_mask, 0.0)

    nll_loss = nll_loss.sum()  # mean()?累積したロスをバッチサイズで割る必要がある場合
    smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss


class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, input, target):
        diff = input - target
        abs_diff = torch.abs_(diff)
        loss = ((abs_diff < self.delta) * 0.5 * diff**2) \
            + ((abs_diff >= self.delta) *
               self.delta * (abs_diff - 0.5 * self.delta))
        return loss


class myloss_fn(nn.Module):
    def __init__(self, smoothing):
        super(myloss_fn, self).__init__()
        self.smoothing = smoothing
        self.criterion = HuberLoss()

    def forward(self, input, target):
        log_prob = -F.log_softmax(input, dim=-1)
        weight = (input.new_ones(input.size()) *
                  self.smoothing / (input.size(-1) - 1.)).detach()
        weight = weight.scatter_(-1, target.unsqueeze(-1),
                                 (1. - self.smoothing)).detach()
        weight = -F.log_softmax(weight, dim=-1)

       # loss = ((weight * -log_prob)*(weight * -log_prob)).sum(dim=-1)
        torch.cuda.empty_cache()

        loss = self.criterion(log_prob, weight)
        return loss


class MyLoss(nn.Module):
    def __init__(self, label_smoothing=0.2):
        super().__init__()
        self.pad_id = 3
        self.l = myloss_fn(label_smoothing)

    def forward(self, input, target, attention_mask=None, target2=None):
        batch_size, sequence_length, num_classes = input.size()

        # Create a mask from the target  ensor
        mask_target = (target == self.pad_id).float()
        mask_target2 = (target != self.pad_id).float()

        loss = self.l(input.view(-1, num_classes), target.view(-1)
                      )
        loss = loss.mean(1).view(1, -1)
        loss = loss * (1 - mask_target)

        loss = loss.sum(-1)/mask_target2.sum(-1)

        return loss


def main():

    criterion = MyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    epochs = 100

    trainer = LoraTrainer(4)

    trainer.train(epochs, criterion)
    print("Training completed")


if __name__ == "__main__":
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        'rinna/japanese-gpt2-medium', use_fast=False)

    main()
