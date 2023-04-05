import os
import json
import csv
from tqdm.contrib.concurrent import process_map


class JsonMaker:
    def __init__(self, base_dir, csv_dir):
        self.base_dir = base_dir
        self.csv_dir = csv_dir

    def make_json(self, output_path):
        # get all csv's
        files = [
            f
            for f in os.listdir(self.csv_dir)
            if os.isfile(os.join(self.csv_dir, f) and f.endswith(".csv"))
        ]

    def _json_helper(self, csv_path):
        with open(csv_path, newline="") as csvfile:
            data = list(csv.reader(csvfile))
        for i, d in enumerate(data):
            data[i] = csv_path.split(os.sep)[-1][:-4] + d
        return data
