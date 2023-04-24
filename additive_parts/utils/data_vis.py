from matplotlib import pyplot as plt
import numpy as np
import json
from scipy.optimize import differential_evolution as DE

with open("/home/akita/cs182/norm.json", "r") as r:
    d = json.load(r)

arr = []
for k in d.keys():
    arr.append(float(d[k]))

# arr = np.log(arr) / 6
arr = np.array(arr)
arr[arr < 2] = 0
arr[arr >= 2] = 2
# arr = np.log(arr)


def loss(x):
    sample = np.random.normal(loc=x[0], scale=x[1], size=arr.shape[0])
    return np.average(np.abs(sample - arr))


out = DE(loss, bounds=[(-6, 6), (0, 10)], x0=[0, 1])
print(out)
plt.hist(arr)
norm = np.random.normal(loc=out["x"][0], scale=out["x"][1], size=arr.shape[0])
print(arr.shape)
print(norm.shape)
plt.ylabel("Count (log scale)")
plt.xlabel("Label Score")
plt.show()
plt.hist(norm)
plt.show()
