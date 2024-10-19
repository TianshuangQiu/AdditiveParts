import os
from glob import glob
import shutil
import json
from tqdm import tqdm

with open("data/50000.json", "r") as r:
    valid_files = json.load(r)

with open("data/benchmark.json", "r") as r:
    valid_files.update(json.load(r))

with open("data/1000.json", "r") as r:
    valid_files.update(json.load(r))

with open("data/10000.json", "r") as r:
    valid_files.update(json.load(r))

valid_files = set(valid_files.keys())
print(len(valid_files))

for file in tqdm(glob("data/parts_*/rotated_files/*")):
    if file not in valid_files:
        os.remove(file)
