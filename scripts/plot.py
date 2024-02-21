import json
import numpy as np
from matplotlib import pyplot as plt

with open("data/agg.json", "r") as r:
    dic = json.load(r)

items = dic.items()
print(len(items))
arr = []
for i in items:
    arr.append(float(i[1]))
arr = np.array(arr)
plt.hist(arr)
plt.title("STL Histogram")
plt.xlabel("Label Value")
plt.ylabel("Count")
# plt.subplots_adjust(left=0.15, right=0.85, bottom=0.1, top=0.9)
plt.savefig("arr.png")
