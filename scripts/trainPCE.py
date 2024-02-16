import torch
import wandb
import os
from torch import random
from torch.utils.data.dataloader import DataLoader, Dataset
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import random_split
import numpy as np
from tqdm import tqdm
from additive_parts.PCE.cloud import PointCloudProcessor
import json
import argparse
import math
from datetime import datetime
from collections import OrderedDict
from additive_parts.utils.preprocess import point_cloudify
import random
import pdb

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


class CloudDataset(Dataset):
    def __init__(self, dict_path, savio):
        with open(dict_path, "r") as r:
            dictionary = json.load(r, object_pairs_hook=OrderedDict)
        self.items = list(dictionary.items())
        self.on_savio = savio

    def __getitem__(self, index):
        entry = self.items[index]
        if self.on_savio:
            return torch.tensor(
                point_cloudify("/global/scratch/users/ethantqiu/" + entry[0], 2048),
                dtype=torch.float,
            ), torch.tensor(float(entry[1]) / 100, dtype=torch.float)
        else:
            return torch.tensor(
                point_cloudify("" + entry[0], 2048), dtype=torch.float
            ), torch.tensor(float(entry[1]) / 100, dtype=torch.float)

    def __len__(self):
        return len(self.items)


def bmse_loss(inputs, targets, noise_sigma=8.0):
    return bmc_loss(inputs, targets, noise_sigma**2)


def bmc_loss(pred, target, noise_var):
    logits = -0.5 * (pred - target.T).pow(2) / noise_var
    loss = F.cross_entropy(logits, torch.arange(pred.shape[0]).to(DEVICE))
    loss = loss * (2 * noise_var)
    return loss


class BMCLoss(nn.Module):
    def __init__(self, init_noise_sigma):
        super(BMCLoss, self).__init__()
        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma))

    def forward(self, pred, target):
        noise_var = self.noise_sigma**2
        return bmc_loss(pred, target, noise_var)


parser = argparse.ArgumentParser()
parser.add_argument("name", help="name of wandb experiment")
parser.add_argument("num", help="amount of data to use", type=int)
parser.add_argument("epoch", help="num of epochs", type=int)
parser.add_argument("lr", help="learning rate", type=float)
parser.add_argument("batch_size", help="batch size", type=int)
parser.add_argument("attn_repeat", help="num times of repeat", type=int)
parser.add_argument("num_layer", help="num of attn layer", type=int)
parser.add_argument("loss", help="type of loss for training")
parser.add_argument("-savio", action="store_true", default=False)

args = parser.parse_args()

EPOCH = args.epoch
LR = args.lr
BATCH_SIZE = args.batch_size
NUM_ATTN = args.attn_repeat
NUM_LAYER = args.num_layer
DEVICE = "cuda"
REGRESSION = True
if args.loss == "mse":
    LOSS = torch.nn.MSELoss()
elif args.loss == "balanced":
    LOSS = BMCLoss(8.0)
elif args.loss == "l1":
    LOSS = torch.nn.L1Loss()


print("emptying cache")
torch.cuda.empty_cache()
print(torch.cuda.memory_summary(device=None, abbreviated=False))


model = PointCloudProcessor(
    4, NUM_ATTN, NUM_LAYER, 64, True, [1024, 128], device=DEVICE, regression=REGRESSION
)
model_type = "Point-cloud-transformer"


print(torch.cuda.memory_summary(device=None, abbreviated=False))
print(
    "Model parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad)
)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
if args.loss == "balanced":
    optimizer.add_param_group(
        {"params": LOSS.noise_sigma, "lr": LR, "name": "noise_sigma"}
    )
wandb.init(
    # set the wandb project where this run will be logged
    project="percentile24",
    # track hyperparameters and run metadata
    config={
        "learning_rate": LR,
        "architecture": model_type,
        "epochs": EPOCH,
        "batch_size": BATCH_SIZE,
        "loss": LOSS,
        "num_attn": NUM_ATTN,
        "num_layer": NUM_LAYER,
        "Model parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
    },
    name=args.name,
)

save_path = "save"
os.makedirs(save_path, exist_ok=True)

if args.savio:
    train_dataset = CloudDataset(
        f"/global/scratch/users/ethantqiu/data/{args.num}.json", args.savio
    )
    test_dataset = CloudDataset(
        "/global/scratch/users/ethantqiu/data/benchmark.json", args.savio
    )
else:
    train_dataset = CloudDataset(f"data/{args.num}.json", args.savio)
    test_dataset = CloudDataset("data/benchmark.json", args.savio)
trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=50, shuffle=False)


def train(model, optimizer: torch.optim.Optimizer, criterion, epochs):
    model.to(DEVICE)
    losses = [[], []]
    for epoch in range(epochs):
        model.train()
        for idx, (data, labels) in enumerate(tqdm(trainloader)):
            output = model(data.to(DEVICE))
            loss = criterion(
                output.float(),
                labels.reshape((output.shape[0], -1)).to(DEVICE),
            ).float()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses[0].append(loss.cpu().detach().numpy())
            del loss
        with torch.no_grad():
            for idx, (data, labels) in enumerate(tqdm(testloader)):
                output = model(data.to(DEVICE))
                loss = torch.nn.MSELoss()(
                    output.float(),
                    labels.reshape((output.shape[0], -1)).to(DEVICE),
                ).float()
                losses[1].append(loss.cpu().detach().numpy())
                del loss
        wandb.log(
            {
                "epoch": epoch,
                "train_loss": np.average(losses[0]),
                "validation_loss": np.average(losses[1]),
            }
        )

        file_path = os.path.join(save_path, f"{args.name}_{epoch}.pt")
        torch.save(
            model.state_dict(),
            file_path,
        )
        wandb.log_model(file_path, name=f"{args.name}_{epoch}")


# wandb.watch(model, log_freq=100, log_graph=True)
train(model, optimizer, LOSS, EPOCH)
