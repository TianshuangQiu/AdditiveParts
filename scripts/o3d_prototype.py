import trimesh
import pyvista as pv
import numpy as np
from matplotlib import pyplot as plt
import open3d as o3d

X_RES = 6400
Y_RES = 4800

original_mesh = trimesh.load_mesh("data/part.stl")
centroid = original_mesh.centroid

top = None
for bottom in np.linspace(original_mesh.bounds[1, 2], original_mesh.bounds[0, 2], 11):
    if top is None:
        top = bottom
        continue
    print(bottom)
    sliced_mesh = trimesh.intersections.slice_mesh_plane(
        original_mesh,
        plane_normal=[0, 0, 1],
        plane_origin=[
            original_mesh.vertices[:, 0].mean(),
            original_mesh.vertices[:, 1].mean(),
            bottom,
        ],
    )
    sliced_mesh = trimesh.intersections.slice_mesh_plane(
        sliced_mesh,
        plane_normal=[0, 0, -1],
        plane_origin=[
            original_mesh.vertices[:, 0].mean(),
            original_mesh.vertices[:, 1].mean(),
            top,
        ],
    )
    top = bottom

    min_bounds = np.min(original_mesh.vertices, axis=0)
    max_bounds = np.max(original_mesh.vertices, axis=0)

    viewer = pv.Plotter()
    viewer.add_mesh(sliced_mesh)
    viewer.show()

    trimesh.exchange.export.export_mesh(sliced_mesh, "data/sliced_part.stl")
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(
        o3d.io.read_triangle_mesh("data/sliced_part.stl")
    )
    # Create scene and add the cube mesh
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)

    # breakpoint()

    # Rays are 6D vectors with origin and ray direction.
    # Here we use a helper function to create rays for a pinhole camera.

    stride = np.max((max_bounds[:2] - min_bounds[:2]) / np.max([X_RES, Y_RES]))

    x = np.arange(
        centroid[0] - stride * X_RES / 2,
        centroid[0] + stride * X_RES / 2,
        stride,
    )
    y = np.arange(
        centroid[1] - stride * Y_RES / 2,
        centroid[1] + stride * Y_RES / 2,
        stride,
    )

    # full coordinate arrays
    xx, yy = np.meshgrid(x, y)

    ray_center = centroid
    # rays = scene.create_rays_pinhole(
    #     fov_deg=150,
    #     center=ray_center,
    #     eye=ray_center + [0.0001, 0.0001, 3],
    #     up=[0, 0, 1],
    #     width_px=X_RES,
    #     height_px=Y_RES,
    # )

    rays = np.stack(
        [
            xx.flatten(),
            yy.flatten(),
            np.ones_like(xx.flatten()) * max_bounds[2] + 0.1,
            np.zeros_like(xx.flatten()),
            np.zeros_like(xx.flatten()),
            -np.ones_like(xx.flatten()),
        ],
        axis=1,
    )
    rays = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)

    # Compute the ray intersections.
    ans = scene.cast_rays(rays)

    # breakpoint()

    # Visualize the hit distance (depth)
    plt.imshow(ans["t_hit"].cpu().numpy().reshape(Y_RES, X_RES))
    plt.show()
