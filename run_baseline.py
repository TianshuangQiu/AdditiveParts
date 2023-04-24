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

#keys_array = np.array(keys)
# print('array done')
#keys_split = np.array_split(keys_array, 10)
# print('split done')
#for k_list in tqdm(keys_split):
    #result = {k: trimesh_score(k) for k in tqdm(k_list)}
    #json.dump(result, outfile)
result = process_map(trimesh_score, keys, chunksize=31250)
#result = process_map(trimesh_score, keys, chunksize=20)
resultDict = dict(zip(keys, result))
json.dump(resultDict, outfile)

infile.close()
outfile.close()
# result = {key: trimesh_score(key) for key,v in json_object.items()}

# out = '/global/scratch/users/yifansong/stl_v1_stablepose.json'
# with open(out, 'w') as outfile:
#    json.dump(result, outfile)
