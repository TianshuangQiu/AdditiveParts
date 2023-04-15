import argparse
import json
import os
import torch
from preprocess import point_cloudify, seven_dim_extraction
from jsonmaker import JsonMaker
from tqdm.contrib.concurrent import process_map

parser = argparse.ArgumentParser()
parser.add_argument("csv_dir", help="csv base folder")
parser.add_argument("json_dir", help="json output path")
parser.add_argument("base_dir", help="base directory")
parser.add_argument("cloud_tensor", help="path to store tensor for point clouds")
parser.add_argument("norms_tensor", help="path to store tensor fo norms")
args = parser.parse_args()

# j = JsonMaker(args.base_dir, args.csv_dir)
# vox_256, stl, vox_64 = j.make_json(args.json_dir)


def make_tensor(stl_path):
    name = stl_path.split("/")[-1]
    cloud = point_cloudify(stl_path, 10000)
    norms = seven_dim_extraction(stl_path)
    with open(os.path.join(cloud_path, name), "w") as w:
        torch.save(torch.from_numpy(cloud), w)
    with open(os.path.join(norm_path, name), "w") as w:
        torch.save(torch.from_numpy(norms), w)

    return (
        {os.path.join(cloud_path, name): d[stl_path]},
        {os.path.join(norm_path, name): d[stl_path]},
    )


with open(args.json_dir, "r") as r:
    d = json.load(r)

cloud_path = os.path.join(args.base_dir, "cloud")
norm_path = os.path.join(args.base_dir, "norm")
os.makedirs(cloud_path, exist_ok=True)
os.makedirs(norm_path, exist_ok=True)

out_paths = process_map(make_tensor, d.keys())
cloud_dict = {}
norms_dict = {}
for o in out_paths:
    cloud_dict.update(o[0])
    norms_dict.update(o[1])

with open("cloud.json", "w") as w:
    json.dump(cloud_dict, w)
with open("norms.json", "w") as w:
    json.dump(norms_dict, w)
