import torch
import rotary_embedding_torch as rpt
from rotary_embedding_torch import RotaryEmbedding
import pandas
import tokenizers
import random
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.utils.data import Dataset, DataLoader
import datasets
import os
import torch.optim as optim
from bitsandbytes.optim import AdamW8bit
from torch.utils.tensorboard import SummaryWriter
import transformer_engine.pytorch as te
import transformer_engine
class Hooker:
    def __init__(self):
        self.activations = {}
        self.flag = False
    def get_act(self, name):
        def hook_act(model, input, output):
                if self.flag == True:
                    self.activations[name] = output.detach().to(torch.float32).cpu()
        return hook_act
    def clear(self):
        self.activations.clear()
        self.flag = False
    def start_logging(self):
        self.flag = True
class KazryPyTorch(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.where(x>=0.0, x, torch.exp(x)*x)
def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
class DecoderLayer(nn.Module):
    def __init__(self, dim=512, activation=KazryPyTorch, heads=8, hidden_mlp_dim=1536):
        super().__init__()
        self.qkv_projects = te.Linear(dim, dim*3, bias=False)
        self.q_norm = te.RMSNorm(dim//heads)
        self.k_norm = te.RMSNorm(dim//heads)
        self.pre_att_norm = te.RMSNorm(dim)
        self.out_project = te.Linear(dim, dim, bias=False)
        self.mlp_up = te.Linear(dim, hidden_mlp_dim, bias=False)
        self.mlp_gate = nn.Sequential(
            te.Linear(dim, hidden_mlp_dim, bias=False),
            activation()
        )
        self.mlp_down = te.Linear(hidden_mlp_dim, dim, bias=False)
        self.pre_mlp_norm = te.RMSNorm(dim)
        self.post_att_norm = te.RMSNorm(dim)
        self.post_mlp_norm = te.RMSNorm(dim)
        self.dim = dim
        self.head_dim = dim // heads
        self.heads = heads
        self.heads_qkv = self.heads * 3
        self.rope = RotaryEmbedding(dim=self.head_dim//2)
    def forward(self, x):
        B, N, C = x.shape
        res_x = x
        x = self.pre_att_norm(x)
        q, k, v = torch.split(self.qkv_projects(x).reshape(B, N, self.heads_qkv, self.head_dim).transpose(1, 2), self.heads, dim=1)
        q, k = self.q_norm(q), self.k_norm(k)
        q, k = self.rope.rotate_queries_or_keys(t=q, seq_dim=-2), self.rope.rotate_queries_or_keys(t=k, seq_dim=-2)
        x = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, N, C)
        x = self.out_project(x)
        x = self.post_att_norm(x)
        x = x + res_x
        res_x = x
        x = self.pre_mlp_norm(x)
        x = self.mlp_gate(x) * self.mlp_up(x)
        x = self.mlp_down(x)
        return self.post_mlp_norm(x) + res_x
class Model(nn.Module):
    def __init__(self, dim=512, activation=KazryPyTorch, heads=8, hidden_mlp_dim=1536, vocab_size=32768, layers=16, layers_for_registration=[0, 7, 15], hooker=Hooker()):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)
        self.layers_list = nn.ModuleList()
        for i in range(layers):
            layer = DecoderLayer(dim=dim, activation=activation, heads=heads, hidden_mlp_dim=hidden_mlp_dim)
            if i in layers_for_registration:
                layer.register_forward_hook(hooker.get_act(f"decoder_layer_{i}"))
            for layer_name in layer.modules():
                if isinstance(layer_name, te.Linear):
                    nn.init.normal_(layer_name.weight, std=0.02)
            self.layers_list.append(layer)
        self.unemb = nn.Sequential(
            nn.RMSNorm(dim),
            nn.Linear(dim, vocab_size, bias=False)
        )
        nn.init.normal_(self.emb.weight, std=0.02)
        nn.init.normal_(self.unemb[-1].weight, std=0.02)
        self.layers = layers
    def forward(self, x):
        x = self.emb(x)
        for layer in self.layers_list:
            x = layer(x)
        return self.unemb(x)
class MyDataset(Dataset):
    def __init__(self, path, window):
        super().__init__()
        self.path = path
        self.context_window = window
        with open(file=self.path, mode="rb") as file:
            self.file = file.read()
        self.tokens = np.frombuffer(self.file, dtype=np.uint16)
        self.tokens = torch.from_numpy(self.tokens)
        del self.file
    def __len__(self):
        return len(self.tokens)//self.context_window - 1
    def __getitem__(self, index):
        input_to_model = self.tokens[index*self.context_window:index*self.context_window+self.context_window]
        target = self.tokens[index*self.context_window+1:index*self.context_window+self.context_window+1]
        return input_to_model.long(), target.long()
def train(
    learning_rate: float = 5e-3,
    batch_size: int = 12,
    weight_decay: float = 1e-3,
    epochs: int = 1,
    shuffle: bool = True,
    num_workers: int = 1,
    train_dataset_path: str | None = None,
    val_dataset_path: str | None = None,
    checkpoint_dir: str | None = None,
    log_dir: str | None = None,
    logging_interval_batches: int = 500,
    saving_interval_batches: int = 10000,
    validation_interval_batches: int = 100,
    activation = KazryPyTorch,
    dim: int = 512,
    heads: int = 8,
    layers: int = 16,
    vocab_size: int = 32768,
    window: int = 512,
    accumulation_steps: int = 16,
    batches_for_val: int = 64,
    batches_for_train: int = 100000 #use -1 for disable
):
    layer_0_grads = []
    layer_7_grads = []
    layer_15_grads = []
    os.makedirs(checkpoint_dir, exist_ok=True)
    dataset_train = MyDataset(train_dataset_path, window)
    dataset_val = MyDataset(val_dataset_path, window)
    dataloader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, drop_last=True, persistent_workers=True)
    dataloader_val = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, drop_last=True, persistent_workers=True)
    hooker = Hooker()
    model = Model(dim=dim, activation=activation, heads=heads, hidden_mlp_dim=dim*3, vocab_size=vocab_size, layers=layers, hooker=hooker)
    model = model.to("cuda", torch.bfloat16)
    model = torch.compile(model=model, fullgraph=False, dynamic=False, mode="max-autotune-no-cudagraphs")
    optimizer = AdamW8bit(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    fp8_recipe = transformer_engine.common.recipe.DelayedScaling(
    margin=1,
    amax_history_len=64,
    interval=1,
    fp8_format=transformer_engine.common.recipe.Format.HYBRID
    )
    if batches_for_train == -1:
        sch = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=len(dataloader_train)*epochs//accumulation_steps, eta_min=learning_rate/10)
    else:
        sch = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=batches_for_train//accumulation_steps, eta_min=learning_rate/10)
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter(log_dir=log_dir)
    optimizer.zero_grad()
    for epoch in range(epochs):
        losses_train = []
        losses_val = [0]
        start_time = time.perf_counter()
        for batch_index, (input_to_model, target) in enumerate(dataloader_train):
            if (batch_index+1) % logging_interval_batches == 0:
                hooker.start_logging()
            target = target.to("cuda")
            input_to_model = input_to_model.to("cuda")
            with te.autocast(enabled=True, recipe=fp8_recipe):
                out = model(input_to_model)
                loss = criterion(out.view(-1, vocab_size), target.view(-1)) / accumulation_steps
                losses_train.append(loss.item() * accumulation_steps)
            loss.backward()
            if (batch_index+1) % logging_interval_batches == 0:
                for name, param in model.named_parameters():
                    if "weight" in name and param.grad is not None:
                        grad_tensor = param.grad.detach().to(torch.float32).cpu() * accumulation_steps
                        if "layers_list.0." in name:
                            layer_0_grads.append(grad_tensor.flatten())
                        elif "layers_list.7." in name:
                            layer_7_grads.append(grad_tensor.flatten())
                        elif "layers_list.15." in name:
                            layer_15_grads.append(grad_tensor.flatten())
            if (batch_index+1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                sch.step()
                optimizer.zero_grad()
            if (batch_index+1) % logging_interval_batches == 0:
                end_time = time.perf_counter()
                if losses_val == [0]:
                    print(f"mean train loss: {sum(losses_train)/len(losses_train)}, mean val loss: {sum(losses_train)/len(losses_train)}, mean time {(end_time-start_time)}")
                    writer.add_scalar("losses/val loss", sum(losses_train)/len(losses_train), global_step=batch_index)
                else:
                    print(f"mean train loss: {sum(losses_train)/len(losses_train)}, mean val loss: {sum(losses_val)/len(losses_val)}, mean time {(end_time-start_time)}")
                    writer.add_scalar("losses/val loss", sum(losses_val)/len(losses_val), global_step=batch_index)
                writer.add_scalar("losses/train loss", sum(losses_train)/len(losses_train), global_step=batch_index)
                writer.add_scalar("losses/val loss", sum(losses_val)/len(losses_val), global_step=batch_index)
                writer.add_scalar("time/time", (end_time-start_time), global_step=batch_index)
                writer.add_histogram("histogramm/layer 1", hooker.activations["decoder_layer_0"], global_step=batch_index)
                writer.add_histogram("histogramm/layer 8", hooker.activations["decoder_layer_7"], global_step=batch_index)
                writer.add_histogram("histogramm/layer 16", hooker.activations["decoder_layer_15"], global_step=batch_index)
                writer.add_histogram("histogramm/layer 1 backward", torch.cat(layer_0_grads), global_step=batch_index)
                writer.add_histogram("histogramm/layer 8 backward", torch.cat(layer_7_grads), global_step=batch_index)
                writer.add_histogram("histogramm/layer 16 backward", torch.cat(layer_15_grads), global_step=batch_index)
                losses_train.clear()
                layer_0_grads.clear()
                layer_7_grads.clear()
                layer_15_grads.clear()
                hooker.clear()
            if (batch_index+1) % validation_interval_batches == 0:
                losses_val = []
                model.eval()
                for batch_index_val, (input_to_model, target) in enumerate(dataloader_val):
                    target = target.to("cuda")
                    input_to_model = input_to_model.to("cuda")
                    with te.autocast(enabled=True, recipe=fp8_recipe), torch.no_grad():
                        out = model(input_to_model)
                        loss = criterion(out.view(-1, vocab_size), target.view(-1))
                        losses_val.append(loss.item())
                    if (batch_index_val+1) % batches_for_val == 0:
                        break
                model.train()
            if (batch_index+1) % saving_interval_batches == 0:
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"model_{batch_index}.pth"))
            if batches_for_train != -1 and (batch_index+1) % batches_for_train == 0:
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, "model_final.pth"))
                break
            start_time = time.perf_counter()
if __name__ == "__main__":
    seed_everything()
    train(train_dataset_path="train.bin", val_dataset_path=r"val.bin",
          checkpoint_dir="kazry/checkpoints", log_dir="kazry/logs")