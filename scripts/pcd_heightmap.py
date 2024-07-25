import trimesh
import pyvista as pv
import numpy as np
from matplotlib import pyplot as plt
import open3d as o3d

mesh = trimesh.load_mesh("data/part.stl")
viewer = pv.Plotter()
# viewer.add_mesh(mesh)
# viewer.show()

# mesh = o3d.io.read_triangle_mesh("data/part.stl")


# points = trimesh.sample.volume_mesh(mesh, 2048)
points = trimesh.sample.sample_surface(mesh, 10000)[0]

# sorted_points = np.array(sorted(points, key=lambda x: x[-1]))
buckets = np.linspace(np.min(mesh.vertices[:, -1]), np.max(mesh.vertices[:, -1]), 10)
splits = []
for i in range(len(buckets) - 1):
    mask = (points[:, -1] >= buckets[i]) & (points[:, -1] <= buckets[i + 1])
    splits.append(points[mask])


# for i, split in enumerate(splits):
#     viewer.add_points(split, color=np.random.rand(3))
# viewer.show()
x_min = np.min(points[:, 0])
x_max = np.max(points[:, 0])
y_min = np.min(points[:, 1])
y_max = np.max(points[:, 1])

x_buckets = np.linspace(x_min, x_max, 65)
y_buckets = np.linspace(y_min, y_max, 65)

for i, split in enumerate(splits):
    split_map = np.zeros((64, 64))
    for index, _ in np.ndenumerate(split_map):
        x_mask = (split[:, 0] >= x_buckets[index[0]]) & (
            split[:, 0] <= x_buckets[index[0] + 1]
        )
        y_mask = (split[:, 1] >= y_buckets[index[1]]) & (
            split[:, 1] <= y_buckets[index[1] + 1]
        )
        pts_in_grid = split[x_mask & y_mask]
        if len(pts_in_grid) > 0:
            split_map[index] = np.max(pts_in_grid[:, -1])
    plt.imshow(split_map)
    plt.show()
