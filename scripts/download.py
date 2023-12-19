import gdown
import json
import shutil
import os

os.makedirs("data", exist_ok=True)

with open("config/data_path.json", "r") as r:
    drive_dict = json.load(r)

for key, value in drive_dict.items():
    gdown.download(url=value, output="data/" + key, fuzzy=True)
    shutil.unpack_archive("data/" + key, f"data/{key[:-4]}")
