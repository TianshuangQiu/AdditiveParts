import torch
from torch import random
from torch.utils.data.dataloader import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
from models.cloud import PointCloudProcessor
import json
import argparse

EPOCH = 20
LR = 0.01
device = "cuda"
model = PointCloudProcessor(3, 64, 8, 64, True, [64, 32], device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, momentum=0.9)

parser = argparse.ArgumentParser()
parser.add_argument("csv_dir", help="csv base folder")
parser.add_argument("json_dir", help="json output path")
parser.add_argument("base_dir", help="base directory")
parser.add_argument("cloud_tensor", help="path to store tensor for point clouds")
parser.add_argument("norms_tensor", help="path to store tensor fo norms")
args = parser.parse_args()

trainloader = DataLoader()
proto_dataset = json.load()


def build_batch(dataset, indices):
    """
    Helper function for creating a batch during training. Builds a batch
    of source and target elements from the dataset. See the next cell for
    when and how it's used.

    Arguments:
        dataset: List[db_element] -- A list of dataset elements
        indices: List[int] -- A list of indices of the dataset to sample
    Returns:
        batch_input: List[List[int]] -- List of tensorized names
        batch_target: List[int] -- List of numerical categories
        batch_indices: List[int] -- List of starting indices of padding
    """
    # Recover what the entries for the batch are
    batch = [dataset[i] for i in indices]
    batch_input = np.array(list(zip(*batch))[0])
    batch_target = np.array(list(zip(*batch))[1])
    batch_indices = np.array(list(zip(*batch))[2])
    return batch_input, batch_target, batch_indices  # lines, categories


def train(model, optimizer, criterion, epochs, batch_size, seed):
    model.to(device)
    model.train()
    train_losses = []
    train_accuracies = []
    eval_accuracies = []
    for epoch in range(epochs):
        random.seed(seed + epoch)
        np.random.seed(seed + epoch)
        torch.manual_seed(seed + epoch)
        indices = np.random.permutation(range(len(train_data)))
        n_correct, n_total = 0, 0
        progress_bar = tqdm(range(0, (len(train_data) // batch_size) + 1))
        for i in progress_bar:
            batch = build_batch(
                train_data, indices[i * batch_size : (i + 1) * batch_size]
            )
            (batch_input, batch_target, batch_indices) = batch_to_torch(*batch)
            (batch_input, batch_target, batch_indices) = list_to_device(
                (batch_input, batch_target, batch_indices)
            )

            logits = model(batch_input, batch_indices)
            loss = criterion(logits, batch_target)
            train_losses.append(loss.item())

            predictions = logits.argmax(dim=-1)
            n_correct += (predictions == batch_target).sum().item()
            n_total += batch_target.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                progress_bar.set_description(
                    f"Epoch: {epoch}  Iteration: {i}  Loss: {np.mean(train_losses[-10:])}"
                )
        train_accuracies.append(n_correct / n_total * 100)
        print(f"Epoch: {epoch}  Train Accuracy: {n_correct / n_total * 100}")

        with torch.no_grad():
            indices = list(range(len(test_data)))
            n_correct, n_total = 0, 0
            for i in range(0, (len(test_data) // batch_size) + 1):
                batch = build_batch(
                    test_data, indices[i * batch_size : (i + 1) * batch_size]
                )
                (batch_input, batch_target, batch_indices) = batch_to_torch(*batch)
                (batch_input, batch_target, batch_indices) = list_to_device(
                    (batch_input, batch_target, batch_indices)
                )

                logits = model(batch_input, batch_indices)
                predictions = logits.argmax(dim=-1)
                n_correct += (predictions == batch_target).sum().item()
                n_total += batch_target.size(0)
            eval_accuracies.append(n_correct / n_total * 100)
            print(f"Epoch: {epoch}  Eval Accuracy: {n_correct / n_total * 100}")

    to_save = {
        "history": {
            "train_losses": train_losses,
            "train_accuracies": train_accuracies,
            "eval_accuracies": eval_accuracies,
        },
        "hparams": {
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "optimizer_class": optimizer_class.__name__,
            "lr": lr,
            "batch_size": batch_size,
            "epochs": epochs,
            "seed": seed,
        },
        "model": [
            (name, list(param.shape)) for name, param in rnn_model.named_parameters()
        ],
    }
    return to_save
