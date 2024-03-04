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
from additive_parts.PCE.pointcloudnet import PointTransformerCls
from additive_parts.meshcnn.MeshCNN.models.networks import define_classifier
import json
import argparse
import math
from datetime import datetime
from collections import OrderedDict
from additive_parts.utils.preprocess import point_cloudify
from additive_parts.meshcnn.MeshCNN.models.layers.mesh import Mesh
from additive_parts.meshcnn.MeshCNN.util.util import is_mesh_file, pad

import trimesh

import random
import pdb

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


class MeshDataset(Dataset):
    def __init__(self, dict_path, args):
        with open(dict_path, "r") as r:
            dictionary = json.load(r, object_pairs_hook=OrderedDict)
        self.items = list(dictionary.items())
        self.on_savio = args.savio
        self.opt = args

    def __getitem__(self, index):
        entry = self.items[index]
        if self.on_savio:
            return Mesh(
                "/global/scratch/users/ethantqiu/" + entry[0],
                args,
                False,
                "/global/scratch/users/ethantqiu/exp",
            ), torch.tensor(float(entry[1]) / 100, dtype=torch.float)
        else:
            m = trimesh.load_mesh("" + entry[0])
            m.fill_holes()
            null_face = np.argwhere(m.area_faces <= 0)
            m.faces = np.delete(m.faces, null_face, axis=0)
            trimesh.exchange.export.export_mesh(m, "" + entry[0] + ".obj")
            mesh = Mesh(
                "" + entry[0] + ".obj",
                args,
                False,
                None,
            )
            label = torch.tensor(float(entry[1]) / 100, dtype=torch.float)
            meta = {"mesh": mesh, "label": label}
            # get edge features
            edge_features = mesh.extract_features()
            print(edge_features.shape)
            edge_features = pad(edge_features, self.opt.ninput_edges)
            meta["edge_features"] = (edge_features - np.mean(edge_features)) / np.std(
                edge_features
            )
            meta["file"] = entry
            return meta

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
parser.add_argument("-norm", help="normalization type", default="group")
parser.add_argument("-num_groups", help="num of groups", default=1, type=int)
parser.add_argument("-pool_res", help="num of groups", default=16, type=int)
parser.add_argument("-fc_n", help="num of groups", default=32, type=int)
parser.add_argument("-resblocks", help="num of blocks", default=3, type=int)
parser.add_argument("-nf0", help="num of conv", default=16, type=int)
parser.add_argument("-num_aug", help="num of augs", default=4, type=int)
parser.add_argument("-conv", help="conv_hyper_param", default=16, type=int)
parser.add_argument("-ninput_edges", help="conv_hyper_param", default=16384, type=int)
parser.add_argument("-loss", help="type of loss for training", default="l1")
parser.add_argument("-savio", action="store_true", default=False)

args = parser.parse_args()

EPOCH = args.epoch
LR = args.lr
BATCH_SIZE = args.batch_size
# NUM_ATTN = args.attn_repeat
# NUM_LAYER = args.num_layer
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

net_config = {
    "batch_size": args.batch_size,
}
args.pool_res = [args.pool_res]
model = define_classifier(
    args.conv, [args.nf0], 8, 1, args, [], "mconvnet", "normal", 0.01
)
model_type = "MeshCNN"


print(torch.cuda.memory_summary(device=None, abbreviated=False))
print(
    "Model parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad)
)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

wandb.init(
    # set the wandb project where this run will be logged
    project="PAPER",
    # track hyperparameters and run metadata
    config={
        "learning_rate": LR,
        "architecture": model_type,
        "epochs_choice": EPOCH,
        "batch_size": BATCH_SIZE,
        "loss": LOSS,
        "num_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        **net_config,
    },
    name=args.name,
    dir="global/scratch/users/ethantqiu/wandb",
)


if args.savio:
    train_dataset = MeshDataset(
        f"/global/scratch/users/ethantqiu/data/{args.num}.json", args
    )
    test_dataset = MeshDataset(
        "/global/scratch/users/ethantqiu/data/benchmark.json", args
    )
    save_path = "/global/scratch/users/ethantqiu/model_weights"

else:
    train_dataset = MeshDataset(f"data/{args.num}.json", args)
    test_dataset = MeshDataset("data/benchmark.json", args)
    save_path = "save"

os.makedirs(save_path, exist_ok=True)
trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = model.to(DEVICE)
print(torch.cuda.memory_summary(device=None, abbreviated=False))


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
            wandb.log({"batch loss": loss})
            del loss
        with torch.no_grad():
            model.eval()
            for idx, (data, labels) in enumerate(tqdm(testloader)):
                output = model(data.to(DEVICE))
                loss = torch.nn.L1Loss()(
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
        # wandb.log_model(file_path, name=f"{args.name}_{epoch}")


train(model, optimizer, LOSS, EPOCH)
