from PyQt5.QtWidgets import QWidget, QVBoxLayout
from vedo import Plotter, Mesh
from vedo import Plotter, Mesh
import numpy as np
from polygon_mesh import PolygonMesh

class VedoViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.plotter = Plotter(qt=True, bg='white')
        layout = QVBoxLayout()
        layout.addWidget(self.plotter.window)
        self.setLayout(layout)
        self.mesh_actor = None
        self.poly_actors = []
        self._last_mesh = None
        self._last_polymesh = None

    def clear(self):
        self.plotter.clear()
        self.mesh_actor = None
        self.poly_actors = []
        self._last_mesh = None
        self._last_polymesh = None

    def set_mesh(self, mesh):
        self.clear()
        self._last_mesh = mesh
        if hasattr(mesh, 'vertices') and hasattr(mesh, 'faces'):
            verts = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.faces)
            m = Mesh([verts, faces])
            m.c('lightblue').alpha(0.7).lw(1)
            self.mesh_actor = m
            self.plotter.show(m, resetcam=True, interactive=False)

    def set_polymesh(self, pmesh: PolygonMesh):
        self.clear()
        self._last_polymesh = pmesh
        verts = np.asarray(pmesh.vertices)
        for face in pmesh.faces:
            poly = Mesh([verts[face], [list(range(len(face)))]]).c('lightblue').alpha(0.7).lw(2)
            self.plotter.show(poly, resetcam=False, interactive=False)
            self.poly_actors.append(poly)
        self.plotter.resetcam()

    def set_background_color(self, color_tuple):
        rgb255 = tuple(int(255*x) for x in color_tuple)
        self.plotter.backgroundColor(rgb255)

    def reset_camera(self):
        self.plotter.resetcam()
// File deleted: replaced by pyvista_viewer.py
