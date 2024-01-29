import json
import numpy as np
from matplotlib import pyplot as plt

with open("data/benchmark.json", "r") as r:
    dic = json.load(r)

items = dic.items()
arr = []
for i in items:
    arr.append(float(i[1]))
arr = np.array(arr)
plt.hist(arr)
plt.savefig("arr.png")
