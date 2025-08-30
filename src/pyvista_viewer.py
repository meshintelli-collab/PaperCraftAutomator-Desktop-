from PyQt5.QtWidgets import QWidget, QVBoxLayout
import pyvista as pv
from pyvistaqt import QtInteractor
import numpy as np
from polygon_mesh import PolygonMesh
from PyQt5.QtCore import QTimer
import time

try:
    from noise import pnoise1
except ImportError:
    def pnoise1(x):
        return np.sin(x)


class PyVistaViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.plotter = QtInteractor(self)
        layout = QVBoxLayout()
        layout.addWidget(self.plotter.interactor)
        self.setLayout(layout)
        self._last_polymesh = None
        self._color_faces = False
        self._wireframe = False
        self._autospin = False
        self._spin_timer = None
        self.model_changed_callbacks = []

    def clear(self):
        self.plotter.clear()
        self._last_polymesh = None
        self._mesh_actor = None

    def set_polymesh(self, pmesh: PolygonMesh, reset_camera=True):
        # Only recreate actor if mesh geometry has changed
        verts = np.asarray(pmesh.vertices)
        faces = pmesh.faces
        if len(faces) == 0:
            return
        faces_pv = []
        for face in faces:
            faces_pv.append(len(face))
            faces_pv.extend(face)
        faces_pv = np.array(faces_pv)
        mesh = pv.PolyData(verts, faces_pv)
        # Store and reuse face colors for this mesh
        if not hasattr(pmesh, '_face_colors') or len(getattr(pmesh, '_face_colors', [])) != len(faces):
            n_faces = len(faces)
            pmesh._face_colors = (np.random.rand(n_faces, 3) * 0.5 + 0.5)
        # Compute and store centroid for autospin
        self._mesh_centroid = verts.mean(axis=0) if verts.shape[0] > 0 else np.zeros(3)
        # If mesh topology changed, recreate actor
        recreate = (
            not hasattr(self, '_mesh_actor') or self._mesh_actor is None or
            not hasattr(self, '_last_polymesh') or self._last_polymesh is None or
            not np.array_equal(np.asarray(self._last_polymesh.vertices), verts) or
            len(self._last_polymesh.faces) != len(faces) or
            any(tuple(self._last_polymesh.faces[i]) != tuple(faces[i]) for i in range(len(faces)))
        )
        orientation = position = None
        if hasattr(self, '_mesh_actor') and self._mesh_actor is not None:
            orientation = self._mesh_actor.GetOrientation()
            position = self._mesh_actor.GetPosition()
        if recreate:
            self.clear()
            if self._color_faces:
                mesh.cell_data['colors'] = (pmesh._face_colors * 255).astype(np.uint8)
                self._mesh_actor = self.plotter.add_mesh(mesh, scalars='colors', rgb=True, show_edges=self._wireframe, opacity=0.85)
            else:
                self._mesh_actor = self.plotter.add_mesh(mesh, color='lightblue', show_edges=self._wireframe, opacity=0.85)
            self._mesh_actor.SetOrigin(self._mesh_centroid[0], self._mesh_centroid[1], self._mesh_centroid[2])
            if orientation is not None:
                self._mesh_actor.SetOrientation(*orientation)
            if position is not None:
                self._mesh_actor.SetPosition(*position)
        else:
            # Update color/wireframe in place
            if self._color_faces:
                self._mesh_actor.GetMapper().SetScalarModeToUseCellData()
                self._mesh_actor.GetMapper().SetScalarVisibility(True)
                self._mesh_actor.GetMapper().SetInputData(mesh)
                mesh.cell_data['colors'] = (pmesh._face_colors * 255).astype(np.uint8)
                self._mesh_actor.GetMapper().SetArrayName('colors')
                self._mesh_actor.GetProperty().SetColor(1,1,1)
                self._mesh_actor.GetProperty().SetOpacity(0.85)
            else:
                self._mesh_actor.GetMapper().SetScalarVisibility(False)
                self._mesh_actor.GetProperty().SetColor(0.678, 0.847, 0.902)  # lightblue
                self._mesh_actor.GetProperty().SetOpacity(0.85)
            self._mesh_actor.GetProperty().SetEdgeVisibility(self._wireframe)
        self._last_polymesh = pmesh
        # Notify listeners that the model has changed
        for cb in self.model_changed_callbacks:
            try:
                cb()
            except Exception as e:
                print(f"[DEBUG] Model changed callback error: {e}")
        if reset_camera and recreate:
            self.plotter.reset_camera()
        if self._autospin:
            self.start_autospin()
    def toggle_colour(self):
        self._color_faces = not self._color_faces
        if self._last_polymesh:
            # Only update color in place
            self.set_polymesh(self._last_polymesh, reset_camera=False)

    def toggle_wireframe(self):
        self._wireframe = not self._wireframe
        if self._last_polymesh:
            # Only update wireframe in place
            self.set_polymesh(self._last_polymesh, reset_camera=False)

    def toggle_autospin(self):
        self._autospin = not self._autospin
        if self._autospin:
            # Store the actor's original position before spinning
            if hasattr(self, '_mesh_actor') and self._mesh_actor is not None:
                self._spin_actor_base_pos = self._mesh_actor.GetPosition()
            else:
                self._spin_actor_base_pos = (0, 0, 0)
            self.start_autospin()
        elif self._spin_timer:
            self._spin_timer.stop()
            self._spin_easing = False
            # Restore the actor's position after spinning
            if hasattr(self, '_mesh_actor') and hasattr(self, '_spin_actor_base_pos'):
                self._mesh_actor.SetPosition(*self._spin_actor_base_pos)

    def start_autospin(self):
        # Do not reset orientation, position, or camera; just start spinning from current state
        if not hasattr(self, '_spin_time'):
            self._spin_time = time.time()
        if not hasattr(self, '_spin_noise_offset'):
            self._spin_noise_offset = np.random.rand(3) * 1000
        if not hasattr(self, '_spin_quat'):
            self._spin_quat = np.array([1, 0, 0, 0], dtype=np.float64)  # w, x, y, z
        self._spin_easing = False
        if not self._spin_timer:
            self._spin_timer = QTimer(self)
            self._spin_timer.timeout.connect(self._spin)
        self._spin_timer.start(30)

    def _spin(self):
        t = time.time() - getattr(self, '_spin_time', 0)
        # Perlin noise for smooth axis drift
        nx = pnoise1(self._spin_noise_offset[0] + t * 0.05)
        ny = pnoise1(self._spin_noise_offset[1] + t * 0.05)
        nz = pnoise1(self._spin_noise_offset[2] + t * 0.05)
        axis = np.array([nx, ny, nz])
        if np.linalg.norm(axis) < 1e-3:
            axis = np.array([0,1,0])
        axis = axis / np.linalg.norm(axis)
        angle = 1.0  # degrees per frame
        # If easing out, slow to zero
        if getattr(self, '_spin_easing', False):
            angle *= 0.90
            if angle < 0.01:
                self._spin_timer.stop()
                self._spin_easing = False
                return
        # Rotate around centroid, but keep the actor at its original position
        if hasattr(self, '_mesh_actor') and self._mesh_actor is not None and hasattr(self, '_mesh_centroid'):
            base_pos = getattr(self, '_spin_actor_base_pos', (0, 0, 0))
            # Move to centroid relative to base position
            self._mesh_actor.SetPosition(
                base_pos[0] - self._mesh_centroid[0],
                base_pos[1] - self._mesh_centroid[1],
                base_pos[2] - self._mesh_centroid[2]
            )
            self._mesh_actor.RotateWXYZ(angle, *axis)
            # Move back to base position
            self._mesh_actor.SetPosition(*base_pos)
        self.plotter.update()

    def set_background_color(self, color_tuple):
        rgb255 = tuple(int(255*x) for x in color_tuple)
        self.plotter.set_background(rgb255)

    def reset_camera(self):
        self.plotter.reset_camera()
