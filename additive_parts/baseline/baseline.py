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
        pose = np.delete(pose, 3, 0)
        rotation = np.delete(pose, 3, 1)
        identity = np.identity(3)
        score = np.linalg.norm(np.subtract(identity, rotation))
        return score
    except:
        return -98
    
def trimesh_score(filepath):
    try:
        return func_timeout.func_timeout(20, trimesh_score_src, args = (filepath,))
    except func_timeout.FunctionTimedOut:
	    return -99