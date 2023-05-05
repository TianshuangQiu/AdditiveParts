import bpy
import os
import sys


'''
Simplifies mesh to target number of faces
Requires Blender 2.8
Author: Rana Hanocka
@input: 
    <obj_file>
    <target_faces> number of target faces
    <outfile> name of simplified .obj file
@output:
    simplified mesh .obj
    to run it from cmd line:
    /opt/blender/blender --background --python blender_process.py /home/rana/koala.obj 1000 /home/rana/koala_1000.obj
'''

class Process:
    def __init__(self, obj_file, target_faces, export_name):
        mesh = self.load_obj(obj_file)
        self.simplify(mesh, target_faces)

    def load_obj(self, obj_file):
        bpy.ops.import_scene.obj(filepath=obj_file, axis_forward='-Z', axis_up='Y', filter_glob="*.obj;*.mtl", use_edges=True,
                                 use_smooth_groups=True, use_split_objects=False, use_split_groups=False,
                                 use_groups_as_vgroups=False, use_image_search=True, split_mode='ON')
        ob = bpy.context.selected_objects[0]
        return ob

    def simplify(self, mesh, target_faces):
        V = len(mesh.data.vertices)
        E = len(mesh.data.edges)
        F = len(mesh.data.polygons)
        # checks manifoldness too
        if (len(mesh.data.edges) > target_faces or V + F - E != 2):
            print("Removed", obj_file)
            os.remove(obj_file)


obj_file = sys.argv[-3]
target_faces = int(sys.argv[-2])
export_name = sys.argv[-1]


print('args: ', obj_file, target_faces, export_name)
blender = Process(obj_file, target_faces, export_name)