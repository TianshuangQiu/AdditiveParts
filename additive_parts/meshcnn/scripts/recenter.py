import open3d as o3d
import os
import numpy as np
import meshio
import sys

# normalize
def normalize_geometry(inputP):
    mesh = o3d.io.read_triangle_mesh(inputP)
    min_bound = mesh.get_min_bound()
    max_bound = mesh.get_max_bound()
    # scaling = abs(max_bound - min_bound)
    center = mesh.get_center()
    mesh.translate(np.array([0, 0, 0]),relative=False)
    # mesh.scale(scale=np.linalg.norm(scaling), center=center)
    o3d.io.write_triangle_mesh(inputP, mesh)

print(sys.argv[-1])
normalize_geometry(sys.argv[-1])

for l in os.listdir(sys.argv[-1]):
    pathy = os.path.join(sys.argv[-1], l)
    # normalize_geometry(pathy)
    cleanup(pathy)
