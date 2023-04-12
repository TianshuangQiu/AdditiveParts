import argparse
import os
from jsonmaker import JsonMaker

parser = argparse.ArgumentParser()
parser.add_argument("csv_dir", help="csv base folder")
parser.add_argument("json_dir", help="json output path")
parser.add_argument("base_dir", help="base directory")
parser.add_argument("cloud_tensor", help="path to store tensor for point clouds")
parser.add_argument("norms_tensor", help="path to store tensor fo norms")
args = parser.parse_args()

j = JsonMaker(args.base_dir, args.csv_dir)
vox_256, stl, vox_64 = j.make_json(args.json_dir)
