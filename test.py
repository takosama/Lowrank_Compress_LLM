
from transformers import AdamW, get_cosine_schedule_with_warmup
from torch.utils.checkpoint import checkpoint
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
import csv
from torch import Tensor, nn, FloatTensor
from typing import Tuple
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
            answer = f"///response//{out}</s>"

            # self.tokenizer.pad_token_idv

            input_ids = self.tokenizer.encode(question, truncation=True,
                                              max_length=1024, padding="max_length", return_tensors="pt", add_special_tokens=True)
            target_ids = self. tokenizer.encode(answer, truncation=True,
                                                max_length=1024, padding="max_length", return_tensors="pt", add_special_tokens=True)
            # 4 padを0にする
            attention_mask = (input_ids != 3).long()
            # 次元を落とす
            attention_mask = attention_mask.squeeze(0)
            if input_ids[0].shape[0] != 256 or attention_mask.shape[0] != 256 or target_ids[0].shape[0] != 256:
                pass
            return {"input_ids": input_ids[0], "attention_mask": attention_mask, "labels": target_ids[0], "text": question}
        except Exception as e:
            print(e)
            return {"input_ids": torch.zeros(self.max_length).int(), "attention_mask": torch.zeros(self.max_length).int(), "labels": torch.zeros(self.max_length).int()}


class MyModule(nn.Module):

    def __init__(self):
        self.rank = 512
        super().__init__()

    def from_Conv1D(self, w):
        self.w = w

        U, S, V = torch.svd(w.weight.data)
        S = torch.diag(S[:self.rank])
        U = U[:, :self.rank]
        V = V[:, :self.rank]
        self.U = nn.Parameter(U.bfloat16())
        self.V = nn.Parameter(V.bfloat16())
        self.S = nn.Parameter(S.bfloat16())
        self.U.requires_grad = False
        self.V.requires_grad = False
        self.S.requires_grad = False

        self.w.weight = nn.Parameter(torch.zeros(1))
        size = self.w.weight.size()
        self.uw = torch.zeros((size[0], 16))
        self.uv = torch.zeros((16, size[1]))
        self.uw = nn.Parameter(self.uw)
        self.uv = nn.Parameter(self.uv)
        self.uw.requires_grad = False
        self.uv.requires_grad = False

        return self

    def forward(self, x):
        self.w.weight = nn.Parameter(
            (self.U @ self.S @ self.V.t()).detach()+self.uw@self.uv)
        ret = self.w(x)
        return ret


class LoraLayer(GPT2Block):
    @staticmethod
    def set(layer: nn.Module):
        size = layer.attn.c_attn.weight.size()
        layer.attn.c_attn = MyModule().from_Conv1D(layer.attn.c_attn)
        layer.attn.c_proj = MyModule().from_Conv1D(layer.attn.c_proj)

        layer.mlp.c_fc = MyModule().from_Conv1D(layer.mlp.c_fc)
        layer.mlp.c_proj = MyModule().from_Conv1D(layer.mlp.c_proj)

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
            self.model.parameters(), lr=6e-4, weight_decay=5e-5)
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

                loss = lora_outputs["logits"]
                loss = criterion(loss, labels, inputs, attention_mask)/a_rate
                torch.cuda.empty_cache()
                loss = loss.mean()
                loss.backward()

                torch.cuda.empty_cache()
                if (step+1) % a_rate == 0:
                    torch.cuda.empty_cache()
                    self.optimizer.step()
                    torch.cuda.empty_cache()
                    self.optimizer.zero_grad()

                if (step+1) % (4*64) == 0 or step == 0:

                    inputs = inputs[0].squeeze(0).unsqueeze(0)
                    # paddingを削除
                    inputs = inputs[inputs != 3].unsqueeze(0)

                    with torch.no_grad():
                        token = self.model.generate(
                            inputs, max_length=1024, do_sample=True,  top_k=500, top_p=0.8, num_return_sequences=3, pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id)
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


class BatchLabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing):
        super(BatchLabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = (input.new_ones(input.size()) *
                  self.smoothing / (input.size(-1) - 1.)).detach()
        weight.scatter_(-1, target.unsqueeze(-1),
                        (1. - self.smoothing)).detach()
        loss = (-weight * log_prob).sum(dim=-1)
        return loss


class MyLoss(nn.Module):
    def __init__(self, label_smoothing=0.2, mask_penalty=0.5):
        super().__init__()
        self.pad_id = 3
        self.mask_penalty = mask_penalty
        self.l = BatchLabelSmoothingCrossEntropy(label_smoothing)

    def forward(self, input, target, attention_mask=None, target2=None):
        batch_size, sequence_length, num_classes = input.size()

        # Create a mask from the target tensor
        mask_target = (target == self.pad_id).float()
        mask_target3 = (target == 2).float()
        mask_target2 = (target != self.pad_id).float()

        loss = self.l(input.view(-1, num_classes), target.view(-1)
                      ).view(batch_size, sequence_length)
        # target2 とtargetをone hotにする

        # Zero out the loss where the target is a pad token
        # Note the "1 - mask_target"
        loss = loss * (1 - mask_target)
        loss = loss * (1 - mask_target3)

        # Create a mask from the input tensor (apply softmax to get probabilities)
        input_probs = torch.softmax(input, dim=1)
        mask_input = (input_probs.argmax(dim=-1) == self.pad_id).float()
        mask_input2 = (input_probs.argmax(dim=-1) == 2).float()
        # Add a penalty for each mask in the input (predictions)
        loss = loss.sum(-1)/mask_target2.sum(-1) + \
            self.mask_penalty * \
            mask_input.sum(-1)+self.mask_penalty * mask_input2.sum(-1)

        return loss*100


def main():

    criterion = MyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    epochs = 100

    trainer = LoraTrainer(4)

    trainer.train(epochs, criterion)
    print("Training completed")


if __name__ == "__main__":
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        'rinna/japanese-gpt2-small', use_fast=False)

    main()
