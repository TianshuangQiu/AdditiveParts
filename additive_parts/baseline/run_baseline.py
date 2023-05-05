from baseline import trimesh_score
import json
import numpy as np
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("base_dir", nargs="?", help="base directory as demonstrated in readme", default="")
parser.add_argument("json_in", nargs="?", help="input json file with actual scores", default="")
parser.add_argument("json_out",nargs="?",help="output json file with predicted sores",default="",)
args = parser.parse_args()

with open(args.json_in, "r") as infile:
    json_object = json.load(infile)

out = args.json_out
outfile = open(out, "w")

keys = list(json_object.keys())
fixed_keys = [os.path.join(args.base_dir, key) for key in keys]

#result = process_map(trimesh_score, keys, chunksize=31250)
resultDict = {k:trimesh_score(k) for k in tqdm(fixed_keys)}
#resultDict = dict(zip(keys, result))
json.dump(resultDict, outfile)

infile.close()
outfile.close()