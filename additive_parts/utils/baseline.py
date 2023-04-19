import sys
sys.path.insert(0, '/global/scratch/users/yifansong/stuff/lib/python3.9/site-packages')
import numpy as np
import trimesh
import func_timeout

# returns the F norm of the difference between the (most likely) stable pose and the identity matrix
def trimesh_score_src(filepath):
    try:
        tmesh = trimesh.load_mesh(filepath, file_type = 'stl')
        dict =  trimesh.poses.compute_stable_poses(tmesh, sigma=0, n_samples=1, threshold = 0.0)
        pose = dict[0][0]
        identity = np.identity(4)
        score = np.linalg.norm(np.subtract(identity, pose))
        return score
    except:
	pass
	return -98
