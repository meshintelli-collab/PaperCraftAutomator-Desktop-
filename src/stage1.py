import os
import random
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QFileDialog, QColorDialog
from pyvista_viewer import PyVistaViewer
from mesh_utils import load_mesh, trimesh_to_polygonmesh
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

class Stage1Widget(QWidget):
    def __init__(self, model_viewer, parent=None):
        super().__init__(parent)
        self.model_viewer = model_viewer
        layout = QVBoxLayout()
        self.import_btn = QPushButton('Import Model')
        self.random_btn = QPushButton('Random Model')
        self.colour_btn = QPushButton('Toggle Colour')
        self.wire_btn = QPushButton('Toggle Wiremesh')
        self.spin_btn = QPushButton('Toggle Autospin')
        self.bgcolor_btn = QPushButton('Set Background Color')
        self.next_btn = QPushButton('Next')

        self.import_btn.clicked.connect(self.import_model)
        self.random_btn.clicked.connect(self.load_random_model)
        self.colour_btn.clicked.connect(self.toggle_colour)
        self.wire_btn.clicked.connect(self.toggle_wireframe)
        self.spin_btn.clicked.connect(self.toggle_autospin)
        self.bgcolor_btn.clicked.connect(self.set_background_color)

        for btn in [self.import_btn, self.random_btn, self.colour_btn, self.wire_btn, self.spin_btn, self.bgcolor_btn, self.next_btn]:
            layout.addWidget(btn)
        layout.addStretch()
        self.setLayout(layout)

        # Load a random model on startup
        self.load_random_model()

    def set_background_color(self):
        
        color = QColorDialog.getColor()
        if color.isValid():
            rgb = color.getRgbF()[:3]
            self.model_viewer.set_background_color(rgb)

    def import_model(self):
        filetypes = '3D Models (*.stl *.obj *.ply *.gltf *.glb)'
        path, _ = QFileDialog.getOpenFileName(self, 'Import 3D Model', MODELS_DIR, filetypes)
        if path:
            mesh = load_mesh(path)
            pmesh = trimesh_to_polygonmesh(mesh)
            self.model_viewer.set_polymesh(pmesh)

    def load_random_model(self):
        files = [f for f in os.listdir(MODELS_DIR) if f.lower().endswith(('.stl', '.obj', '.ply', '.gltf', '.glb'))]
        if files:
            path = os.path.join(MODELS_DIR, random.choice(files))
            mesh = load_mesh(path)
            pmesh = trimesh_to_polygonmesh(mesh)
            self.model_viewer.set_polymesh(pmesh)

    def toggle_colour(self):
        self.model_viewer.toggle_colour()

    def toggle_wireframe(self):
        self.model_viewer.toggle_wireframe()

    def toggle_autospin(self):
        self.model_viewer.toggle_autospin()