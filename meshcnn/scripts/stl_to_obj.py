import bpy
import os
import sys

argv = sys.argv
argv = argv[argv.index("--") + 1:] # get all args after "--"

stl_in = argv[0]

# import meshio

# mesh = meshio.read(stl_in)


# obj_out = argv[1]
# modified_output = os.path.join(os.path.dirname(stl_in)[:2], 'output', os.path.dirname(stl_in)[2:])
filename = os.path.basename(stl_in)[:-3]+'obj'
# meshio.write("./output/"+filename, mesh, file_format="obj")
# fancy 
bpy.ops.import_mesh.stl(filepath=stl_in, axis_forward='-Z', axis_up='Y')
bpy.ops.export_scene.obj(filepath="./output/"+filename, axis_forward='-Z', axis_up='Y')
