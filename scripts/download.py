# import gdown
import json
import shutil
import os

os.makedirs("data", exist_ok=True)

with open("config/data_path.json", "r") as r:
    drive_dict = json.load(r)

failed = []
with open("data/50000.json", "r") as r:
    valid_files = json.load(r).keys()
for key, value in drive_dict.items():
    try:
        if os.path.exists(f"data/{key}"):
            # gdown.download(url=value, output="data/" + key, fuzzy=True)
            print("unzipping " + key)
            shutil.unpack_archive(
                "data/" + key,
                f"data/{key[:-4]}",
            )
        shutil.rmtree(f"data/{key[:-4]}/Binvox_files_default_res", ignore_errors=True)
    except:
        failed.append(key)
        pass

print(failed)
with open("error.txt", "w") as w:
    w.writelines(str(failed))
