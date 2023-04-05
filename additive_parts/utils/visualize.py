import pyvista as pv


def visualize_pointcloud(cloud, color="white", point_size=2):
    plotter = pv.Plotter()
    plotter.add_mesh(cloud, color=color, point_size=point_size)
