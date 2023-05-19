
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
from lib import QADataset, Swish
from transformers import AutoModelWithLMHead, AutoConfig
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, PreTrainedModel
from torch.optim import AdamW
import json
import random

import transformers

CUDA_DEBUGGING = True


class MyConv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        nf (`int`): The number of output features.
        nx (`int`): The number of input features.
    """

    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        rank = 16
        self.u = nn.Parameter(torch.zeros(
            nf, rank))
        self.v = nn.Parameter(torch.zeros(
            rank, nx))
        self.wu = nn.Parameter(torch.zeros(
            nf, rank))
        self.wv = nn.Parameter(torch.zeros(
            rank, nx))
        self.b = nn.Parameter(torch.zeros(nf))

    def from_Conv1D(self, conv1d: nn.Module):
        self.nf = conv1d.nf
        rank = 16
        # rank分解を行う

        # Perform SVD
        u, s, v = torch.svd(conv1d.weight)

        # Keep only the top 'rank' components
        self.wu = nn.Parameter(u[:, : rank] * s[: rank].sqrt())
        self.wv = nn.Parameter((v[:, : rank] * s[: rank].sqrt()).t())
        self.wu.requires_grad = False
        self.wv.requires_grad = False

        self.u = nn.Parameter(torch.zeros(self.wu.size()))
        self.v = nn.Parameter(torch.zeros(self.wv.size()))

        self.b = conv1d.bias
        self.act = Swish()
        return self

    def f(self, x):
        size_out = x.size()[:-1] + (self.nf,)

        x = torch.addmm(self.b, x.view(-1, x.size(-1)),
                        (self.act(self.u)@self.act(self.v)).detach()+self.wu@self.wv)
        x = x.view(size_out)
        torch.cuda.empty_cache()
        return x

    def forward(self, x):
        output = checkpoint(self.f,   x)
        return output


class LoraLayer(GPT2Block):
    def __init__(self, layer: nn.Module,  model: GPT2LMHeadModel, rank, i, device):
        config = model
        super().__init__(config)
        del config

        ind, oud = self.attn.c_proj.weight.shape
        self.attn.c_proj = MyConv1D(oud, ind).from_Conv1D(
            layer.attn.c_proj).to(device).bfloat16()
        self.attn.c_attn = MyConv1D(oud, ind).from_Conv1D(
            layer.attn.c_attn).to(device).bfloat16()
        self.mlp.c_proj = MyConv1D(oud, ind).from_Conv1D(
            layer.mlp.c_proj).to(device).bfloat16()
        self.mlp.c_fc = MyConv1D(oud, ind).from_Conv1D(
            layer.mlp.c_fc).to(device).bfloat16()

        pass


class LoraManagerbase(AutoModelWithLMHead):
    def __init__(self,     config, rank):
        super().__init__(config=config)

    @classmethod
    def SetUp(self, model, rank, device):

        model.base_model.h = nn.ModuleList(
            LoraLayer(model.base_model.h[i], model.config, rank, i, device).cuda().bfloat16() for i in range(len(model.base_model.h.cuda()))
        )

        torch.cuda.empty_cache()
        model.base_model.ln_f = model.base_model.ln_f.bfloat16() .to(device)
        model.base_model.wte = model.base_model.wte.bfloat16().to(device)
        model.base_model.wpe = model.base_model.wpe.bfloat16().to(device)
        model.base_model.drop = model.base_model.drop

        return model

    def forward(self, *args: any, **kwds: any) -> any:
        r = super().forward(*args, **kwds)
        print(r)
        return r

    @ classmethod
    def Set_Train_Layer(cls, model, l):
        on = [23, 22, 21, 20, 19, 18, 17, 16, 15,
              14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        components = ['mlp.c_fc', 'mlp.c_proj',
                      'attn.c_attn', 'attn.c_proj']
        params = ['u', 'v', "b"]

        def init_weights(model, layer, components, params):
            for component in components:
                component_parts = component.split('.')
                target = model.base_model.h[layer]
                for part in component_parts:
                    target = getattr(target, part)
                for param in params:
                    torch.nn.init.normal_(getattr(target, param), 0, 0.5)

                    getattr(target, param).requires_grad = True

        def init_weights2(model, layer, components, params):
            for component in components:
                component_parts = component.split('.')
                target = model.base_model.h[layer]
                for part in component_parts:
                    target = getattr(target, part)
                for param in params:
                    getattr(target, param).requires_grad = False

        for i in on:
            init_weights(model, i, components, params)

        for i in range(len(model.base_model.h)):
            if i not in on:
                init_weights2(model, i, components, params)

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
        torch.cuda.empty_cache()
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        self.    model = LoraManagerbase.from_pretrained(
            "rinna/japanese-gpt2-medium", self. device, rank=rank)
        self.    model.save_pretrained("test")
        torch.cuda.empty_cache()

        self.rank = rank

        self.tokenizer = AutoTokenizer.from_pretrained(
            "rinna/japanese-gpt2-medium", use_fast=False)
        self.tokenizer.sep_token = self.tokenizer.eos_token

        # データを整形
        with open('/home/rintya/rinnna_loader/databricks-dolly-15k-translated14801_14900.json', 'r', encoding="utf-8") as f:
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

        self.dataloader = DataLoader(dataset, batch_size=1,   shuffle=True)

    def train(self, epochs, criterion):
        # save
        epoch = 0
        torch.cuda.empty_cache()

        step = 0

        low_path_loss = 0
      #  self.lora_manager.save_pretrained(f"model_epoch_{epoch}")
        scaler = torch.cuda.amp.GradScaler()

        b_Step = 0
        for epoch in range(epochs):
            b_Step = b_Step+1

            # re shuffle to self.dataloader
            tq = tqdm(self.dataloader)
            torch.cuda.empty_cache()
            loss_sum = 0
            self.optimizer = Lion(
                self.model.parameters(), lr=1e-5, weight_decay=1e-4)

            self.optimizer.zero_grad()
            h_ = len(self.model.base_model.h)
            LoraManagerbase.Set_Train_Layer(self.model, h_-1)
            self.model.train()
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
                lora_outputs = self.model(
                    inputs, attention_mask=attention_mask, labels=labels)
                torch.cuda.empty_cache()

                # loss = lora_outputs["loss"]
                loss = lora_outputs["logits"]
                loss = criterion(loss, labels, attention_mask)
                torch.cuda.empty_cache()
                loss.backward()

                torch.cuda.empty_cache()
                if (step+1) % 8 == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 1)
                    torch.cuda.empty_cache()
                    self.optimizer.step()
                    torch.cuda.empty_cache()
                    self.optimizer.zero_grad()

                if (step+1) % (4*64) == 0 or step == 0:

                    inputs = inputs[0].squeeze(0).unsqueeze(0)
                    # paddingを削除
                    inputs = inputs[inputs != 3].unsqueeze(0)
                    # eosを削除
                    inputs = inputs[inputs != 2].unsqueeze(0)
                    with torch.no_grad():
                        token = self.model.generate(
                            inputs, max_length=1024, do_sample=True,  min_length=100, top_p=0.95, top_k=500, pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id)
                    print(self.tokenizer.decode(token[0]))

                tq.set_description(f"step {step} loss: {loss.item():.5f}")
                # save to csv
                path = "loss.csv"
                if not os.path.exists(path):
                    with open(path, 'w') as f:
                        f.write(f"{loss}\n")
                else:
                    with open(path, 'a') as f:
                        f.write(f"{loss}\n")

                step += 1
                torch.cuda.empty_cache()

            print(f"epoch {epoch} loss: {loss_sum/step:.5f}")
            # save
            self.model.save_pretrained(f"model_train_epoch_{epoch}")
            torch.cuda.empty_cache()


class BatchLabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(BatchLabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * \
            self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1)
        return loss


class MyLoss(nn.Module):
    def __init__(self, label_smoothing=0.1, mask_penalty=0.1):
        super().__init__()
        self.pad_id = 3
        self.mask_penalty = mask_penalty
        self.l = BatchLabelSmoothingCrossEntropy(label_smoothing)

    def forward(self, input, target, attention_mask=None):
        batch_size, sequence_length, num_classes = input.size()

        # Create a mask from the target tensor
        mask_target = (target == self.pad_id).float()
        mask_target2 = (target != self.pad_id).float()

        loss = self.l(input.view(-1, num_classes), target.view(-1)
                      ).view(batch_size, sequence_length)

        # Zero out the loss where the target is a pad token
        # Note the "1 - mask_target"
        loss = loss * (1 - mask_target)

        # Create a mask from the input tensor (apply softmax to get probabilities)
        input_probs = F.softmax(input, dim=-1)
        mask_input = (input_probs.argmax(dim=-1) == self.pad_id).float()

        # Add a penalty for each mask in the input (predictions)
        loss = loss.sum()/mask_target2.sum() + self.mask_penalty * torch.sum(mask_input)

        return loss


def main():

    criterion = MyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    epochs = 100

    trainer = LoraTrainer(4)

    trainer.train(epochs, criterion)
    print("Training completed")


if __name__ == "__main__":
    # debug cuda
    torch.backends.cudnn.deterministic = True
    torch.autograd.set_detect_anomaly(True)
    main()
