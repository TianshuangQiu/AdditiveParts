from additive_parts.utils.baseline import trimesh_score
import json
import numpy as np
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

with open("/global/scratch/users/ethantqiu/Data/stl.json", "r") as infile:
    json_object = json.load(infile)

out = "/global/scratch/users/yifansong/stl_works0_stablepose.json"
outfile = open(out, "w")

keys = list(json_object.keys())

result = process_map(trimesh_score, keys, chunksize=31250)
resultDict = dict(zip(keys, result))
json.dump(resultDict, outfile)

infile.close()
outfile.close()