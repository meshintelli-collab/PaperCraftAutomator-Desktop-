// File deleted: replaced by vedo_viewer.py
import os
import trimesh
import pyqtgraph.opengl as gl
from PyQt5.QtWidgets import QWidget, QVBoxLayout
import pyqtgraph as pg
import numpy as np

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
import random
PRELOAD_MODEL = None
model_files = [f for f in os.listdir(MODELS_DIR) if f.lower().endswith(('.stl', '.obj', '.ply', '.gltf', '.glb'))]
if model_files:
    PRELOAD_MODEL = os.path.join(MODELS_DIR, random.choice(model_files))

class ModelViewer(QWidget):
    def set_polygons(self, polygons):
        """
        Set a list of polygons (each as Nx3 array of vertices) to be displayed as filled faces and outlines overlay.
        """
        self._last_polygons = polygons
        self._draw_mesh()
    def get_mesh(self):
        return self._last_mesh

    def set_mesh(self, mesh):
        self._last_mesh = mesh
        self._last_polygons = None  # Clear overlays when setting a new mesh
        self._draw_mesh()
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()
        self.gl_widget = gl.GLViewWidget()
        layout.addWidget(self.gl_widget)
        self.setLayout(layout)
        self.mesh_item = None
        self.current_color = (0.5, 0.5, 1, 1)
        self.wireframe = True
        self.autospin = False
        self._spin_timer = None
        self._last_mesh = None
        self._first_draw = True
        self.load_preload_model()
    
    def load_model(self, filename):
        try:
            mesh = trimesh.load(filename, force='mesh')
            if not isinstance(mesh, trimesh.Trimesh):
                raise ValueError("Loaded file is not a mesh.")
            self._last_mesh = mesh
            self._last_polygons = None  # Clear overlays when loading a new model
            self._first_draw = True
            self._draw_mesh()
        except Exception as e:
            print(f"Failed to load model: {e}")
    
    def load_preload_model(self):
        if 'PRELOAD_MODEL' in globals() and PRELOAD_MODEL:
            self._last_polygons = None  # Clear overlays when loading a new model
            self.load_model(PRELOAD_MODEL)
    

    def _draw_mesh(self):
        # Save camera state
        cam_pos = self.gl_widget.cameraPosition()
        cam_dist = self.gl_widget.opts.get('distance', None)
        cam_elev = self.gl_widget.opts.get('elevation', None)
        cam_azim = self.gl_widget.opts.get('azimuth', None)
        cam_center = self.gl_widget.opts.get('center', None)
        # Remove previous mesh or polygon items
        if hasattr(self, 'mesh_item') and self.mesh_item:
            self.gl_widget.removeItem(self.mesh_item)
            self.mesh_item = None
        if hasattr(self, '_polyline_items'):
            for item in self._polyline_items:
                self.gl_widget.removeItem(item)
            self._polyline_items = []
        if hasattr(self, '_polyface_items'):
            for item in self._polyface_items:
                self.gl_widget.removeItem(item)
            self._polyface_items = []
        # If polygons are set, draw them as filled faces and outlines
        if hasattr(self, '_last_polygons') and self._last_polygons is not None:
            self._polyface_items = []
            self._polyline_items = []
            for poly in self._last_polygons:
                poly = np.asarray(poly)
                if poly.shape[0] < 3:
                    continue
                # Triangulate for filled face
                try:
                    from trimesh import triangulate_polygon
                    tri_faces = triangulate_polygon([poly])
                except Exception:
                    n = poly.shape[0]
                    tri_faces = [[0, i, i+1] for i in range(1, n-1)]
                verts = poly
                faces = np.array(tri_faces)
                color = np.append(np.random.rand(3)*0.5+0.5, 0.7)  # pastel, semi-transparent
                mesh_item = gl.GLMeshItem(vertexes=verts, faces=faces, drawFaces=True, smooth=False, color=color, drawEdges=False)
                self.gl_widget.addItem(mesh_item)
                self._polyface_items.append(mesh_item)
                # Outline
                pts = np.vstack([poly, poly[0]])
                plt = gl.GLLinePlotItem(pos=pts, color=(1,0,0,1), width=2, antialias=True, mode='line_strip')
                self.gl_widget.addItem(plt)
                self._polyline_items.append(plt)
            # Center on first draw after model load, else restore previous camera
            if getattr(self, '_first_draw', False):
                all_pts = np.vstack(self._last_polygons)
                center = all_pts.mean(axis=0)
                self.gl_widget.opts['center'] = pg.Vector(center[0], center[1], center[2])
                self.gl_widget.setCameraPosition(distance=np.ptp(all_pts, axis=0).max()*2)
                self._first_draw = False
            else:
                if cam_center is not None:
                    self.gl_widget.opts['center'] = cam_center
                if cam_dist is not None:
                    self.gl_widget.setCameraPosition(distance=cam_dist)
                if cam_elev is not None:
                    self.gl_widget.opts['elevation'] = cam_elev
                if cam_azim is not None:
                    self.gl_widget.opts['azimuth'] = cam_azim
            return
        # Otherwise, draw mesh as usual
        mesh = self._last_mesh
        edge_args = {}
        if self.wireframe:
            edge_args = dict(drawEdges=True, edgeColor=(0,0,0,1), edgeWidth=4)
        else:
            edge_args = dict(drawEdges=False)
        if hasattr(self, '_face_coloring') and self._face_coloring:
            verts = mesh.vertices
            faces = mesh.faces
            colors = np.zeros((len(verts), 4))
            for i, face in enumerate(faces):
                base = np.random.rand(3) * 0.5 + 0.5
                color = np.append(base, 1.0)
                for vi in face:
                    colors[vi] = color
            self.mesh_item = gl.GLMeshItem(vertexes=verts, faces=faces, drawFaces=True, smooth=False, vertexColors=colors, **edge_args)
        else:
            verts = mesh.vertices
            faces = mesh.faces
            self.mesh_item = gl.GLMeshItem(vertexes=verts, faces=faces, drawFaces=True, smooth=False, color=self.current_color, **edge_args)
        self.gl_widget.addItem(self.mesh_item)
        if getattr(self, '_first_draw', False):
            center = verts.mean(axis=0)
            self.gl_widget.opts['center'] = pg.Vector(center[0], center[1], center[2])
            self.gl_widget.setCameraPosition(distance=max(mesh.extents)*2)
            self._first_draw = False
        else:
            if cam_center is not None:
                self.gl_widget.opts['center'] = cam_center
            if cam_dist is not None:
                self.gl_widget.setCameraPosition(distance=cam_dist)
            if cam_elev is not None:
                self.gl_widget.opts['elevation'] = cam_elev
            if cam_azim is not None:
                self.gl_widget.opts['azimuth'] = cam_azim
        # Otherwise, draw mesh as usual
        mesh = self._last_mesh
        edge_args = {}
        if self.wireframe:
            edge_args = dict(drawEdges=True, edgeColor=(0,0,0,1), edgeWidth=4)
        else:
            edge_args = dict(drawEdges=False)
        if hasattr(self, '_face_coloring') and self._face_coloring:
            verts = mesh.vertices
            faces = mesh.faces
            colors = np.zeros((len(verts), 4))
            for i, face in enumerate(faces):
                base = np.random.rand(3) * 0.5 + 0.5
                color = np.append(base, 1.0)
                for vi in face:
                    colors[vi] = color
            self.mesh_item = gl.GLMeshItem(vertexes=verts, faces=faces, drawFaces=True, smooth=False, vertexColors=colors, **edge_args)
        else:
            verts = mesh.vertices
            faces = mesh.faces
            self.mesh_item = gl.GLMeshItem(vertexes=verts, faces=faces, drawFaces=True, smooth=False, color=self.current_color, **edge_args)
        self.gl_widget.addItem(self.mesh_item)
        if getattr(self, '_first_draw', False):
            center = verts.mean(axis=0)
            self.gl_widget.opts['center'] = pg.Vector(center[0], center[1], center[2])
            self.gl_widget.setCameraPosition(distance=max(mesh.extents)*2)
            self._first_draw = False
        else:
            if cam_center is not None:
                self.gl_widget.opts['center'] = cam_center
            if cam_dist is not None:
                self.gl_widget.setCameraPosition(distance=cam_dist)
            if cam_elev is not None:
                self.gl_widget.opts['elevation'] = cam_elev
            if cam_azim is not None:
                self.gl_widget.opts['azimuth'] = cam_azim


    def toggle_colour(self):
        # Toggle per-face pastel coloring
        if not hasattr(self, '_face_coloring') or not self._face_coloring:
            self._face_coloring = True
        else:
            self._face_coloring = False
        self._draw_mesh()

    def toggle_wireframe(self):
        self.wireframe = not self.wireframe
        self._draw_mesh()

    def toggle_autospin(self):
        self.autospin = not self.autospin
        if self.autospin:
            if not hasattr(self, '_spin_axis'):
                self._spin_axis = np.array([1.0, 0.0])
            if not hasattr(self, '_spin_timer') or self._spin_timer is None:
                from PyQt5.QtCore import QTimer
                self._spin_timer = QTimer(self)
                self._spin_timer.timeout.connect(self._spin)
            self._spin_timer.start(16)  # ~60 FPS for smooth spin
        else:
            if hasattr(self, '_spin_timer') and self._spin_timer:
                self._spin_timer.stop()

    def _spin(self):
        # Only update camera, do not redraw mesh or edges
        if not hasattr(self, '_spin_axis'):
            self._spin_axis = np.array([1.0, 0.0])
        delta = (np.random.rand(2) - 0.5) * 0.01  # smaller random change
        self._spin_axis += delta
        self._spin_axis /= np.linalg.norm(self._spin_axis)
        azim, elev = self._spin_axis * 0.3  # much smaller step for smoothness
        self.gl_widget.orbit(azim, elev)

    def set_background_color(self, color_tuple):
        # color_tuple: (r, g, b) in 0-1, convert to 0-255
        rgb255 = tuple(int(255*x) for x in color_tuple)
        self.gl_widget.setBackgroundColor(rgb255)