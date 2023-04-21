import os
import json
import csv
import pdb
import re
from tqdm.contrib.concurrent import process_map


class JsonMaker:
    def __init__(self, base_dir, csv_dir):
        self.base_dir = base_dir
        self.csv_dir = csv_dir

    def make_json(self, output_path):
        # get all csv's
        files = [
            os.path.join(self.csv_dir, f)
            for f in os.listdir(self.csv_dir)
            if os.path.isfile(os.path.join(self.csv_dir, f)) and f.endswith(".csv")
        ]
        data_array = process_map(self._json_helper, files)
        # data_array = []
        # for f in files:
        #     data_array.append(self._json_helper(f))

        vox_256_dict, stl_dict, vox_64_dict = {}, {}, {}
        for d in data_array:
            vox_256_dict.update(d[0])
            stl_dict.update(d[1])
            vox_64_dict.update(d[2])

        with open(f"{output_path}/vox_256.json", "w") as w:
            json.dump(vox_256_dict, w)
        with open(f"{output_path}/stl.json", "w") as w:
            json.dump(stl_dict, w)
        with open(f"{output_path}/vox_64.json", "w") as w:
            json.dump(vox_64_dict, w)
        return vox_256_dict, stl_dict, vox_64_dict

    def _json_helper(self, csv_path):
        with open(csv_path, newline="") as csvfile:
            data = list(csv.reader(csvfile))
        print(csv_path)
        vox_256_dict, stl_dict, vox_64_dict = {}, {}, {}
        for d in data:
            # csv_name = csv_path.split(os.sep)[-1][:-4]
            # part_group = re.findall(r"\d+_", csv_name)[0][:-1]
            # begin_idx = re.findall(r"_\d+", csv_name)[0][1:]
            # end_idx = re.findall(r"-\d+", csv_name)[0][1:]
            vox_256_path = (
                f"{self.base_dir}/" + f"Binvox_files_default_res/{d[0][:-4]}.binvox"
            )
            stl_path = f"{self.base_dir}/" + f"rotated_files/{d[0]}"

            if os.path.isfile(vox_256_path):
                vox_256_dict[vox_256_path] = d[1]
            if os.path.isfile(stl_path):
                stl_dict[stl_path] = d[1]
            # if len(d[0][:-4].split(r".stl")) > 1:
            #     vox_64_dict[
            #         f"{self.base_dir}/"
            #         + f"(64) parts_{part_group}, files {begin_idx} through {end_idx}/"
            #         + d[0][:-4].split(r".stl")[0]
            #         + "_compressed."
            #         + d[0][:-4].split(r".")[1]
            #         + ".binvox"
            #     ] = d[1]
            # else:
            #     vox_64_dict[
            #         f"{self.base_dir}/"
            #         + f"(64) parts_{part_group}, files {begin_idx} through {end_idx}/"
            #         + d[0][:-4].split(r".stl")[0]
            #         + "_compressed.binvox"
            #     ] = d[1]
        return vox_256_dict, stl_dict, vox_64_dict
