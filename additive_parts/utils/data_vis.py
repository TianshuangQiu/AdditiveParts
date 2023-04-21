from matplotlib import pyplot as plt
import numpy as np
import json

with open("/home/akita/cs182/norm.json", "r") as r:
    d = json.load(r)

arr = []
for k in d.keys():
    arr.append(float(d[k]))
arr = np.array(arr)
x = np.random.normal(size=arr.shape[0])
arr = np.log(arr)
joint = np.vstack([x, arr]).T
plt.scatter(x=joint[:, 0], y=joint[:, 1])
plt.show()
