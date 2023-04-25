import argparse
import json
import os
import torch
from preprocess import point_cloudify, seven_dim_extraction
from jsonmaker import JsonMaker
from tqdm.contrib.concurrent import process_map

parser = argparse.ArgumentParser()
parser.add_argument("csv_dir", nargs="?", help="csv base folder", default="")
parser.add_argument(
    "json_dir",
    nargs="?",
    help="json output path",
    default="/global/scratch/users/ethantqiu/Data/stl.json",
)
parser.add_argument(
    "base_dir",
    nargs="?",
    help="base directory",
    default="/global/scratch/users/ethantqiu/Data",
)
args = parser.parse_args()

j = JsonMaker(args.base_dir, args.csv_dir)
vox_256, stl, vox_64 = j.make_json(args.json_dir)

args.json_dir = os.path.join(args.json_dir, "stl.json")


def make_tensor(stl_path):
    global max_face
    name = stl_path.split("/")[-1]
    # cloud = point_cloudify(stl_path, 4096)
    norms = seven_dim_extraction(stl_path)
    # torch.save(torch.from_numpy(cloud), os.path.join(cloud_path, name))
    # torch.save(
    #     (torch.from_numpy(norms[0]), torch.from_numpy(norms[1])),
    #     os.path.join(norm_path, name),
    # )
    cloud_path = ""
    return (
        {os.path.join(cloud_path, name): d[stl_path]},
        {os.path.join(norm_path, name): d[stl_path]},
    )


with open(args.json_dir, "r") as r:
    d = json.load(r)

# cloud_path = os.path.join(args.base_dir, "rawcloud")
norm_path = os.path.join(args.base_dir, "norm")
# os.makedirs(cloud_path, exist_ok=True)
os.makedirs(norm_path, exist_ok=True)

out_paths = process_map(make_tensor, d.keys(), chunksize=3000)
# for k in d.keys():
#     make_tensor(k)
cloud_dict = {}
norms_dict = {}
for o in out_paths:
    cloud_dict.update(o[0])
    norms_dict.update(o[1])

# with open(os.path.join(args.base_dir, "rawcloud.json"), "w") as w:
#     json.dump(cloud_dict, w)
with open(os.path.join(args.base_dir, "norm.json"), "w") as w:
    json.dump(norms_dict, w)
