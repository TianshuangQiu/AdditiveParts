from glob import glob
import shutil
import json
import os

with open("data/50000.json", "r") as r:
    data = json.load(r)
files_list = []
for key, value in data.items():
    files_list.append(key.replace("rotated_files", "depth_image") + ".pth")
    files_list.append(key.replace("rotated_files", "rotated_depth_image") + ".pth")
    files_list.append(key.replace("rotated_files", "distance_field") + ".pth")

missing_dict = {}
for file in files_list:
    file = file.strip()
    if not os.path.exists(file):
        if "rotated_depth_image" in file:
            file = file.replace("rotated_depth_image", "rotated_files")
        elif "depth_image" in file:
            file = file.replace("depth_image", "rotated_files")
        file = file.replace("distance_field", "rotated_files")
        missing_dict[file[:-4]] = 0

with open("missing.json", "w") as w:
    json.dump(missing_dict, w)
