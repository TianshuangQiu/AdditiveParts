import torch
import wandb
import os
from torch import random
from torch.utils.data.dataloader import DataLoader, Dataset
from torch.utils.data import random_split
import numpy as np
from tqdm import tqdm
from additive_parts.models.cloud import PointCloudProcessor
import json
import argparse
import math
from datetime import datetime
import pdb


class CloudDataset(Dataset):
    def __init__(self, dict_path):
        with open(dict_path, "r") as r:
            dictionary = json.load(r)
        self.files = np.array(list(dictionary.items()))

    def __getitem__(self, index):
        entry = self.files[index]
        path = entry[0]
        value = float(entry[1])
        value = np.log(value) / 6
        data = torch.load(path)
        # Crop so label is greater than 10
        return data.float(), torch.tensor(value).float()

    def __len__(self):
        return self.files.shape[0]


class NormDataset(Dataset):
    def __init__(self, dict_path):
        with open(dict_path, "r") as r:
            dictionary = json.load(r)
        self.files = np.array(list(dictionary.items()))

    def __getitem__(self, index):
        entry = self.files[index]
        path = entry[0]
        value = float(entry[1])
        # value = np.log(value) / 6
        # value /= 10
        value = [1, 0] if value <= 2 else [0, 1]
        data = torch.load(path)

        return (data[0].float(), data[1].float()), torch.tensor(value).float()

    def __len__(self):
        return self.files.shape[0]


EPOCH = 5
LR = 0.01
BATCH_SIZE = 8
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOSS = torch.nn.L1Loss()
NUM_ATTN = 2
NUM_LAYER = 2
ABLATION = None
if DEVICE == "cuda":
    print("emptying cache")
    torch.cuda.empty_cache()
    print(torch.cuda.memory_summary(device=None, abbreviated=False))

output_loss = torch.nn.CrossEntropyLoss()
parser = argparse.ArgumentParser()
parser.add_argument("json_dir", help="json output path")
parser.add_argument("--test", action="store_true")
args = parser.parse_args()

model_type = None
data_type = None
if "cloud" in args.json_dir:
    model = PointCloudProcessor(3, NUM_ATTN, NUM_LAYER, 4, True, [8, 4], device=DEVICE)
    model_type = "Point-cloud-transformer"
elif "semi" in args.json_dir:
    model = PointCloudProcessor(
        8, NUM_ATTN, NUM_LAYER, 32, True, [32, 16], device=DEVICE
    )
    model_type = "8-Dim-norms-transformer"
elif "norm" in args.json_dir:
    model = PointCloudProcessor(7, NUM_ATTN, NUM_LAYER, 8, True, [8, 4], device=DEVICE)
    model_type = "Face-norms-transformer"

else:
    raise NotImplementedError("Unknown model")

if "semi" in args.json_dir:
    data_type = "semi_normalized"
elif "raw" in args.json_dir:
    data_type = "raw"
else:
    data_type = "normalized"


print(model_type, torch.cuda.memory_summary(device=None, abbreviated=False))
print(
    "Model parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad)
)
if args.test:
    model_type = "TEST RUN"
if not args.test and DEVICE == "cpu":
    raise ValueError("cuda is not available")

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Plz don't use maliciously :(
wandb.login(key="ddb6406253b10bb52a73e1c61e24a54994725c96")
wandb.init(
    # set the wandb project where this run will be logged
    project="additive-parts",
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
        "data_type": data_type,
        "ablation": ABLATION,
    },
)

save_path = os.path.join("/home/akita/cs182/models", model_type)
os.makedirs(save_path, exist_ok=True)

full_dataset = (
    CloudDataset(args.json_dir) if "cloud" in model_type else NormDataset(args.json_dir)
)
train_dataset, test_dataset = random_split(full_dataset, [0.8, 0.2])
trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


def train(model, optimizer, criterion, epochs, seed):
    model.train()
    model.float()
    model = model.to(DEVICE)
    for epoch in range(epochs + 1):
        np.random.seed(seed + epoch)
        torch.manual_seed(seed + epoch)
        losses = [[], []]
        # pdb.set_trace()
        for idx, (data, labels) in enumerate(tqdm(trainloader)):
            if "norm" in model_type:
                tensor, mask = data
                tensor = tensor.to(DEVICE)
                mask = mask.to(DEVICE)
                output = model(tensor, src_key_padding=mask)

                loss = criterion(
                    output.float(),
                    labels.reshape((output.shape[0], -1)).float().to(DEVICE),
                ).float()
                optimizer.zero_grad()
                if epoch > 0:
                    loss.backward()
                    optimizer.step()
            else:
                output = model.to(DEVICE)(data.to(DEVICE))
                loss = criterion(
                    output.float(),
                    labels.reshape((output.shape[0], -1)).float().to(DEVICE),
                ).float()
                optimizer.zero_grad()
                if epoch > 0:
                    loss.backward()
                    optimizer.step()

            l = torch.mean(
                output_loss(
                    output.float(),
                    labels.reshape((output.shape[0], -1)).float().to(DEVICE),
                ).float()
            )
            losses[0].append(l.cpu().detach().numpy())
            del l
            del loss
        with torch.no_grad():
            for idx, (data, labels) in enumerate(tqdm(testloader)):
                if "norm" in model_type:
                    tensor, mask = data
                    output = model.to(DEVICE)(
                        tensor.to(DEVICE), src_key_padding=mask.to(DEVICE)
                    )
                    loss = criterion(
                        output.float(),
                        labels.reshape((output.shape[0], -1)).float().to(DEVICE),
                    ).float()
                else:
                    output = model.to(DEVICE)(data.to(DEVICE))
                    loss = criterion(
                        output.float(),
                        labels.reshape((output.shape[0], -1)).float().to(DEVICE),
                    ).float()
                    losses[1].append(loss.cpu())

                l = torch.mean(
                    output_loss(
                        output.float(),
                        labels.reshape((output.shape[0], -1)).float().to(DEVICE),
                    ).float()
                )
                losses[1].append(l.cpu().detach().numpy())
                del l
                del loss

        wandb.log(
            {
                "epoch": epoch,
                "train_loss": np.average(losses[0]),
                "validation_loss": np.average(losses[1]),
            }
        )

        torch.save(
            model.state_dict(),
            os.path.join(
                save_path, datetime.now().strftime("%m-%d-%Y_%H:%M:%S") + ".sav"
            ),
        )


wandb.watch(model)
train(model, optimizer, LOSS, EPOCH, SEED)
