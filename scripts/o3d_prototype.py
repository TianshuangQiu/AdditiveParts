import trimesh
from trimesh import transformations
import pyvista as pv
import numpy as np
from matplotlib import pyplot as plt
import open3d as o3d

X_RES = 64
Y_RES = 64

original_mesh = trimesh.load_mesh("data/test.obj")
original_mesh.vertices -= original_mesh.centroid
original_mesh.vertices /= np.max(np.linalg.norm(original_mesh.vertices, axis=1))
centroid = original_mesh.centroid.copy()
centroid[0] += 1e-10

total_range = np.max(original_mesh.vertices, axis=0) - np.min(
    original_mesh.vertices, axis=0
)
min_bounds = np.min(original_mesh.vertices, axis=0)
max_bounds = np.max(original_mesh.vertices, axis=0)
fov = 90
trimesh.exchange.export.export_mesh(original_mesh, "data/converted.stl")
sdf_scene = o3d.t.geometry.RaycastingScene()
sdf_scene.add_triangles(
    o3d.t.geometry.TriangleMesh.from_legacy(
        o3d.io.read_triangle_mesh("data/converted.stl")
    )
)

top = None
for i, bottom in enumerate(
    np.linspace(original_mesh.bounds[1, 2], original_mesh.bounds[0, 2], 11)
):
    if top is None:
        top = bottom
        continue

    # Distance field
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

    current_plane = np.stack(
        [
            xx.flatten(),
            yy.flatten(),
            bottom * np.ones_like(xx.flatten()),
        ],
        axis=1,
    )
    distance = sdf_scene.compute_distance(
        o3d.core.Tensor(current_plane, dtype=o3d.core.Dtype.Float32)
    )
    # plt.imshow(distance.cpu().numpy().reshape(Y_RES, X_RES))
    # plt.show()

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
    # normalize verts
    sliced_mesh.vertices -= sliced_mesh.centroid

    viewer = pv.Plotter()
    viewer.add_mesh(sliced_mesh)

    trimesh.exchange.export.export_mesh(sliced_mesh, "data/sliced_part.stl")
    rot_matrix = transformations.rotation_matrix(np.pi, [1, 0, 0], [0, 0, 0])
    sliced_mesh = sliced_mesh.apply_transform(rot_matrix)
    trimesh.exchange.export.export_mesh(sliced_mesh, "data/rotated_sliced_part.stl")
    # viewer.add_mesh(rotated_slice)
    # viewer.show()
    for j, cur_slice_path in enumerate(
        ["data/sliced_part.stl", "data/rotated_sliced_part.stl"]
    ):
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(
            o3d.io.read_triangle_mesh(cur_slice_path)
        )

        # Create scene and add the cube mesh
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(mesh)

        # Rays are 6D vectors with origin and ray direction.
        # Here we use a helper function to create rays for a pinhole camera.

        rays = scene.create_rays_pinhole(
            fov_deg=fov,
            center=centroid,
            eye=np.array([0, 0, 1]),
            up=[0, 0, 1],
            width_px=X_RES,
            height_px=Y_RES,
        )

        # rays = np.stack(
        #     [
        #         xx.flatten(),
        #         yy.flatten(),
        #         np.ones_like(xx.flatten()) * max_bounds[2] + 0.1,
        #         np.zeros_like(xx.flatten()),
        #         np.zeros_like(xx.flatten()),
        #         -np.ones_like(xx.flatten()),
        #     ],
        #     axis=1,
        # )
        # rays = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)

        # Compute the ray intersections.
        ans = scene.cast_rays(rays)

        # Visualize the hit distance (depth)

        depth_image = ans["t_hit"].cpu().numpy().reshape(Y_RES, X_RES)
        depth_image = np.nan_to_num(depth_image, nan=0.0, posinf=0.0, neginf=0.0)
        plt.imshow(depth_image)
        plt.show()

    top = bottom
