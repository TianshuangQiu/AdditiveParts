from collections import OrderedDict
from torch.utils.data import Dataset
import torch
import json


class FieldDepthDataset(Dataset):
    def __init__(self, dict_path, dataset_type):
        with open(dict_path, "r") as r:
            dictionary = json.load(r, object_pairs_hook=OrderedDict)
        self.items = list(dictionary.items())
        self.dataset_type = dataset_type

    def __getitem__(self, index):
        entry = self.items[index]
        return torch.load(
            entry[0].replace("rotated_files", self.dataset_type) + ".pth"
        ).unsqueeze(0), torch.tensor(float(entry[1]) / 100, dtype=torch.float)

    def __len__(self):
        return len(self.items)
