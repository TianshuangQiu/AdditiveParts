import json
from tqdm import tqdm
import trimesh

with open("/global/scratch/users/ethantqiu/data/agg.json", "r") as r:
    run_dict = json.load(r)

water_tight_dict = {}
volume_dict = {}
for mesh_path, rating in tqdm(run_dict.items()):
    mesh = trimesh.load_mesh(mesh_path)
    if mesh.is_watertight:
        water_tight_dict[mesh_path] = rating
    if mesh.is_volume:
        volume_dict[mesh_path] = rating

with open("/global/scratch/users/ethantqiu/data/watertight.json", "w") as w:
    json.dump(water_tight_dict, w)

with open("/global/scratch/users/ethantqiu/data/volume.json", "w") as w:
    json.dump(volume_dict, w)

print(f"watertight_dict has {len(water_tight_dict)} entries")
print(f"volume has {len(volume_dict)} entries")
