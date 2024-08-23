import trimesh
from trimesh import transformations
import numpy as np
import open3d as o3d
import os
import json
import torch
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import argparse

X_RES = 64
Y_RES = 64
NUM_SLICE = 64


def mesh_process_pipeline(mesh_path, output_path, tmp_path):
    original_mesh = trimesh.load_mesh(mesh_path)
    original_mesh.vertices -= original_mesh.centroid
    original_mesh.vertices /= np.max(np.linalg.norm(original_mesh.vertices, axis=1))
    centroid = original_mesh.centroid.copy()
    centroid[0] += 1e-10
    min_bounds = np.min(original_mesh.vertices, axis=0)
    max_bounds = np.max(original_mesh.vertices, axis=0)
    fov = 90
    trimesh.exchange.export.export_mesh(
        original_mesh, f"{output_path}/{tmp_path}_ORIGINAL.stl"
    )
    sdf_scene = o3d.t.geometry.RaycastingScene()
    sdf_scene.add_triangles(
        o3d.t.geometry.TriangleMesh.from_legacy(
            o3d.io.read_triangle_mesh(f"{output_path}/{tmp_path}_ORIGINAL.stl")
        )
    )

    top = None

    depth_images = []
    rotated_depth_images = []
    distance_fields = []
    for bottom in np.linspace(
        original_mesh.bounds[1, 2], original_mesh.bounds[0, 2], NUM_SLICE + 1
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
        distance_field = distance.cpu().numpy().reshape(Y_RES, X_RES)
        distance_fields.append(distance_field)
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

        trimesh.exchange.export.export_mesh(
            sliced_mesh, f"{output_path}/{tmp_path}.stl"
        )
        rot_matrix = transformations.rotation_matrix(np.pi, [1, 0, 0], [0, 0, 0])
        sliced_mesh = sliced_mesh.apply_transform(rot_matrix)
        trimesh.exchange.export.export_mesh(
            sliced_mesh, f"{output_path}/{tmp_path}_ROTATED.stl"
        )
        # viewer.add_mesh(rotated_slice)
        # viewer.show()
        for j, cur_slice_path in enumerate(
            [f"{output_path}/{tmp_path}.stl", f"{output_path}/{tmp_path}_ROTATED.stl"]
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

            # Compute the ray intersections.
            ans = scene.cast_rays(rays)

            # Visualize the hit distance (depth)

            depth_image = ans["t_hit"].cpu().numpy().reshape(Y_RES, X_RES)
            depth_image = np.nan_to_num(depth_image, nan=0.0, posinf=0.0, neginf=0.0)

            if j == 0:
                depth_images.append(depth_image)
            elif j == 1:
                rotated_depth_images.append(depth_image)

    top = bottom

    write_path = mesh_path.replace("rotated_files", "depth_image")
    os.makedirs("/".join(write_path.split("/")[:-1]), exist_ok=True)
    torch.save(
        torch.from_numpy(np.array(depth_images[::-1])),
        write_path + ".pth",
    )

    write_path = mesh_path.replace("rotated_files", "rotated_depth_image")
    os.makedirs("/".join(write_path.split("/")[:-1]), exist_ok=True)
    torch.save(
        torch.from_numpy(np.array(rotated_depth_images[::-1])),
        write_path + ".pth",
    )

    write_path = mesh_path.replace("rotated_files", "distance_field")
    os.makedirs("/".join(write_path.split("/")[:-1]), exist_ok=True)
    torch.save(
        torch.from_numpy(np.array(distance_fields[::-1])),
        write_path + ".pth",
    )


parser = argparse.ArgumentParser()
parser.add_argument("--assignment", type=int, default=0)
args = parser.parse_args()

ASSIGNMENT = args.assignment
print(f"Running assignment {ASSIGNMENT}")

with open("/global/scratch/users/ethantqiu/data/agg.json", "r") as r:
    run_dict = json.load(r)

files = np.array(list(run_dict.keys()))
assignment_split = np.array_split(files, 8)[ASSIGNMENT]

for i, file in enumerate(tqdm(assignment_split)):
    mesh_process_pipeline(
        f"/global/scratch/users/ethantqiu/{file}",
        "/global/scratch/users/ethantqiu/data",
        f"tmp_ASSIGNMENT_{ASSIGNMENT}",
    )
