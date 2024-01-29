# import gdown
import json
import shutil
import os

os.makedirs("/global/scratch/users/ethantqiu/data", exist_ok=True)

with open("config/data_path.json", "r") as r:
    drive_dict = json.load(r)

failed = []
for key, value in drive_dict.items():
    try:
        if os.path.exists(f"/global/scratch/users/ethantqiu/data/{key}"):
            # gdown.download(url=value, output="data/" + key, fuzzy=True)
            print("unzipping" + key)
            shutil.unpack_archive("/global/scratch/users/ethantqiu/data/" + key, f"/global/scratch/users/ethantqiu/data/{key[:-4]}")
    except:
        failed.append(key)
        pass

print(failed)
with open("error.txt", "w") as w:
    w.writelines(str(failed))
