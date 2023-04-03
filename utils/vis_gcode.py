import pyvista as pv
import numpy as np
import pdb
import time

# Read in gcode file
filename = ""
with open(filename, "r") as f:
    lines = f.readlines()

# Extract x, y, z coordinates from gcode commands
x = []
y = []
z = []
current_z = -1
n = 0
start_time = time.time()

for line in lines:
    if line.startswith("G1"):
        tokens = line.split()
        for token in tokens:
            if token.startswith("X"):
                x.append(float(token[1:]))
            elif token.startswith("Z"):
                current_z = float(token[1:])
            elif token.startswith("Y"):
                y.append(float(token[1:]))
                z.append(current_z)
    elif line.startswith("G0"):
        tokens = line.split()
        for token in tokens:
            if token.startswith("Z"):
                current_z = float(token[1:])
        # if current_z > 5:
        #     break

# Create point cloud from coordinates
# pdb.set_trace()

points = np.column_stack((x, y, z))
cloud = pv.PolyData(points)

prev = points[0]
lines = []
for p in points[1:]:
    lines.extend([prev, p])
    prev = p
print(time.time() - start_time)

# Create plot
plotter = pv.Plotter()
plotter.add_mesh(cloud, color="white", point_size=2)

plotter.add_lines(np.array(lines), width=0.5, color="b")
print("added")
plotter.show()
