from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QWidget, QPushButton
from PyQt5.QtCore import Qt, QTimer
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from mesh_utils import unfold_to_2d_nets
import numpy as np

class Stage4Widget(QWidget):
    def __init__(self, get_mesh_callback, get_graph_callback, get_tree_edges_callback, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.get_mesh_callback = get_mesh_callback
        self.get_graph_callback = get_graph_callback
        self.get_tree_edges_callback = get_tree_edges_callback
        self.nets_2d = None
        self._setup_layout()

    def _setup_layout(self):
        main_layout = QHBoxLayout(self)
        # Left: buttons
        btn_layout = QVBoxLayout()
        self.unfold_btn = QPushButton("Unfold")
        self.unfold_btn.setMinimumHeight(60)
        self.unfold_btn.setStyleSheet('font-size: 18px;')
        self.unfold_btn.clicked.connect(self._on_unfold)
        btn_layout.addWidget(self.unfold_btn)
        self.next_btn = QPushButton("Next →")
        self.next_btn.setMinimumHeight(60)
        self.next_btn.setStyleSheet('font-size: 18px;')
        self.next_btn.clicked.connect(self._on_next)
        btn_layout.addWidget(self.next_btn)
        btn_layout.addStretch(1)
        main_layout.addLayout(btn_layout, stretch=0)
        # Right: viewer
        self.figure = Figure(figsize=(6, 5))
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self.figure)
        main_layout.addWidget(self.canvas, stretch=1)
        self.setLayout(main_layout)
        self._draw_placeholder()

    def set_graph_and_mesh(self, graph, mesh):
        self.graph = graph
        self.mesh = mesh
        self.nets_2d = None
        self._draw_placeholder()

    def _draw_placeholder(self):
        self.ax.clear()
        self.ax.text(0.5, 0.5, "2D net will appear here after unfolding", ha='center', va='center', fontsize=16, color='gray', transform=self.ax.transAxes)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.canvas.draw()

    def _on_unfold(self):
        if not hasattr(self, 'graph') or not hasattr(self, 'mesh'):
            return
        self.nets_2d = unfold_to_2d_nets(self.graph, self.mesh)
        self._animate_unfolding(self.nets_2d)

    def _animate_unfolding(self, nets_2d):
        self.ax.clear()
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        colors = np.array([[0.5,0.7,1,0.5],[1,0.7,0.5,0.5],[0.7,1,0.5,0.5],[1,0.5,0.7,0.5],[0.7,0.5,1,0.5]])
        self._anim_polygons = []
        self._anim_layers = []
        self._anim_total_layers = 0
        for net in nets_2d:
            if 'bfs_layers' in net:
                self._anim_layers.append(net['bfs_layers'])
                self._anim_total_layers += len(net['bfs_layers'])
            else:
                self._anim_layers.append([net['polygons']])
                self._anim_total_layers += 1
        self._anim_layer_idx = 0
        self._anim_net_idx = 0
        self._anim_colors = colors
        # Calculate delay per layer so total is 5 seconds
        self._anim_total_time_ms = 5000
        self._anim_delay_per_layer = max(50, int(self._anim_total_time_ms / max(1, self._anim_total_layers)))
        self._animate_next_layer()

    def _animate_next_layer(self):
        if self._anim_net_idx >= len(self._anim_layers):
            return
        layers = self._anim_layers[self._anim_net_idx]
        if self._anim_layer_idx >= len(layers):
            self._anim_net_idx += 1
            self._anim_layer_idx = 0
            QTimer.singleShot(self._anim_delay_per_layer, self._animate_next_layer)
            return
        self.ax.clear()
        for l in range(self._anim_layer_idx+1):
            for poly in layers[l]:
                poly = np.array(poly)
                c = self._anim_colors[self._anim_net_idx % len(self._anim_colors)]
                self.ax.fill(poly[:,0], poly[:,1], color=c, edgecolor='k')
                self.ax.plot(poly[:,0], poly[:,1], color='k', lw=1)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        self.canvas.draw()
        self._anim_layer_idx += 1
        QTimer.singleShot(self._anim_delay_per_layer, self._animate_next_layer)

    def _on_next(self):
        pass  # To be connected by parent
