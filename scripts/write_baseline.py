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
from tqdm.contrib.concurrent import process_map
import json
import argparse
import trimesh
from datetime import datetime
from collections import OrderedDict
import random
import pymeshlab as ml
import pdb

parser = argparse.ArgumentParser()
parser.add_argument("-savio", action="store_true")

args = parser.parse_args()


def compute_stable_pose(mesh_path: str):
    ms = ml.MeshSet()
    ms.load_new_mesh(mesh_path)
    m = ms.current_mesh()
    # print("input mesh has", m.vertex_number(), "vertex and", m.face_number(), "faces")
    if m.face_number() > 512:
        # print("decimating")
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=512, preservenormal=True
        )
        # print("decimated")
    ms.save_current_mesh(f"{mesh_path}.obj")

    mesh = trimesh.load(f"{mesh_path}.obj")

    stable_pose, prob = trimesh.poses.compute_stable_poses(
        mesh, threshold=0.5, n_samples=256
    )
    if len(stable_pose) == 0:
        stable_pose = np.eye(4).reshape(-1, 4, 4) * -1
        prob = np.array([0.5])
    # returns the norm of rotation matrices scaled by their probabilities
    baseline_value = np.linalg.norm(
        (stable_pose[:, :3, :3] - np.eye(3)) * prob.reshape(-1, 1, 1)
    )
    torch.save(
        baseline_value,
        mesh_path + ".txt",
    )


if args.savio:
    data = [
        "/global/scratch/users/ethantqiu/data/10000.json",
        "/global/scratch/users/ethantqiu/data/benchmark.json",
    ]
else:
    data = ["data/1000.json", "data/benchmark.json"]


for d in data:
    write_dict = {}
    with open(d, "r") as r:
        data_dict = json.load(r, object_pairs_hook=OrderedDict)
    for k, v in data_dict.items():
        write_dict[k + ".txt"] = v / 100
    with open(d.replace(".json", "_baseline.json"), "w") as w:
        json.dump(write_dict, w)
    # for k in tqdm(data_dict.keys()):
    #     compute_stable_pose(k)

    process_map(compute_stable_pose, list(data_dict.keys()), chunksize=500)
