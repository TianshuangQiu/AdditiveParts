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
from datetime import datetime


class JsonDataset(Dataset):
    def __init__(self, dict_path):
        with open(dict_path, "r") as r:
            dictionary = json.load(r)
        self.files = np.array(list(dictionary.items()))

    def __getitem__(self, index):
        entry = self.files[index]
        path = entry[0]
        value = float(entry[1])
        if value > 10:
            value = 10
        data = torch.load(path)
        # Crop so label is greater than 10
        return data, torch.tensor(value).double()

    def __len__(self):
        return self.files.shape[0]


EPOCH = 20
LR = 0.01
BATCH_SIZE = 512
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


parser = argparse.ArgumentParser()
parser.add_argument("json_dir", help="json output path")
parser.add_argument("--test", action="store_true")
args = parser.parse_args()

model_type = None
if "cloud" in args.json_dir:
    model = PointCloudProcessor(3, 64, 8, 64, True, [64, 32], device=DEVICE)
    model_type = "Point cloud transformer"
elif "norm" in args.json_dir:
    model = PointCloudProcessor(7, 64, 8, 64, True, [64, 32], device=DEVICE)
    model_type = "Seven dim representation transformer"
else:
    raise NotImplementedError("Unknown model")

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
    },
)

save_path = os.path.join("/global/scratch/users/ethantqiu/models", model_type)
os.makedirs(save_path, exist_ok=True)

full_dataset = JsonDataset(args.json_dir)
train_dataset, test_dataset = random_split(full_dataset, [0.8, 0.2])
trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


def train(model, optimizer, criterion, epochs, seed):
    model.to(DEVICE)
    model.train()
    train_losses = []
    eval_losses = []
    length = [len(trainloader) // BATCH_SIZE, len(testloader) // BATCH_SIZE]
    for epoch in range(epochs):
        np.random.seed(seed + epoch)
        torch.manual_seed(seed + epoch)
        for idx, (data, labels) in tqdm(enumerate(trainloader)):
            print(f"working on batch {idx}")
            output = model.to(DEVICE)(data.to(DEVICE))
            loss = criterion(output, labels.to(DEVICE))
            train_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log({"train_loss": torch.norm(loss)}, step=epoch * length[0] + idx)

        with torch.no_grad():
            for idx, data, labels in tqdm(enumerate(testloader)):
                output = model.to(DEVICE)(data.to(DEVICE))
                loss = criterion(output, labels.to(DEVICE))
                eval_losses.append(loss.item())
                wandb.log(
                    {"validation_loss": torch.norm(loss)},
                    step=epoch * length[0] + int(idx / length[1] * length[0]),
                )

        torch.save(model.state_dict(), os.path.join(save_path, datetime.now() + ".sav"))


wandb.watch(model)
train(model, optimizer, torch.nn.MSELoss, EPOCH, SEED)
