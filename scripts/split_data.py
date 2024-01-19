import numpy as np
from collections import OrderedDict
import json
import random
from tqdm import tqdm
import pdb
from matplotlib import pyplot as plt

random.seed(69420)
np.random.seed(69420)

with open("data/stl.json", "r") as r:
    dictionary = json.load(r, object_pairs_hook=OrderedDict)
items = list(dictionary.items())
random.shuffle(items)
avg = 4.4640826787307315
bench_mark = []
rest_of_data = []
labels = []
for i in tqdm(items):
    labels.append(float(i[1]))
    if len(bench_mark) < 5000:
        if float(i[1]) <= avg:
            bench_mark.append(i)
        else:
            rest_of_data.append(i)
    elif len(bench_mark) < 10000:
        if float(i[1]) > avg:
            bench_mark.append(i)
        else:
            rest_of_data.append(i)
    else:
        rest_of_data.append(i)
labels = np.array(labels)
pdb.set_trace()
# with open("data/benchmark.json", "w") as w:
#     json.dump({i: j for i, j in bench_mark}, w)

# with open("data/agg.json", "w") as w:
#     json.dump({i: j for i, j in rest_of_data}, w)

# with open("data/test.json", "w") as w:
#     json.dump({i: j for i, j in rest_of_data[:10000]}, w)

# with open("data/10000.json", "w") as w:
#     json.dump({i: j for i, j in rest_of_data[10000:20000]}, w)

# with open("data/50000.json", "w") as w:
#     json.dump({i: j for i, j in rest_of_data[10000:60000]}, w)

# with open("data/100000.json", "w") as w:
#     json.dump({i: j for i, j in rest_of_data[10000:110000]}, w)

# with open("data/200000.json", "w") as w:
#     json.dump({i: j for i, j in rest_of_data[10000:210000]}, w)
