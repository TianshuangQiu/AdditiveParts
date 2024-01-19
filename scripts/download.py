import gdown
import json
import shutil
import os

os.makedirs("data", exist_ok=True)

with open("config/data_path.json", "r") as r:
    drive_dict = json.load(r)

failed = []
for key, value in drive_dict.items():
    try:
        if not os.path.exists(f"data/{key}") and not os.path.exists(f"data/{key[:-4]}"):
            gdown.download(url=value, output="data/" + key, fuzzy=True)
            shutil.unpack_archive("data/" + key, f"data/{key[:-4]}")
    except:
        failed.append(key)
        pass

print(failed)
with open("error.txt", "w") as w:
    w.writelines(str(failed))
