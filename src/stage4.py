from PyQt5.QtWidgets import QVBoxLayout, QWidget, QPushButton, QLabel, QHBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import networkx as nx
import numpy as np

class Stage4Widget(QWidget):
    def __init__(self, get_mesh_callback, get_graph_callback, get_tree_edges_callback, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.get_mesh_callback = get_mesh_callback
        self.get_graph_callback = get_graph_callback
        self.get_tree_edges_callback = get_tree_edges_callback
        self.cut_edges = set()
        self._setup_layout()
        self._init_graph_state()

    def _setup_layout(self):
        layout = QVBoxLayout(self)
        # Top: stats and reset button
        top_container = QWidget()
        top_layout = QVBoxLayout(top_container)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(8)
        self.stats_label = QLabel("")
        self.stats_label.setWordWrap(True)
        top_layout.addWidget(self.stats_label)
        self.reset_btn = QPushButton("Reset Cuts")
        self.reset_btn.clicked.connect(self.reset_cuts)
        top_layout.addWidget(self.reset_btn)
        layout.addWidget(top_container)
        # Middle: viewer (matplotlib canvas)
        self.figure = Figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(self.canvas.sizePolicy().Expanding, self.canvas.sizePolicy().Expanding)
        self.canvas.setMinimumHeight(350)
        layout.addWidget(self.canvas)
        # Bottom: Next button
        self.next_btn = QPushButton("Next")
        self.next_btn.clicked.connect(self._on_next)
        layout.addWidget(self.next_btn)
        self.setLayout(layout)
        self.canvas.mpl_connect('button_press_event', self.on_canvas_click)

    def _init_graph_state(self):
        # Get the current graph and tree edges from previous stage
        G = self.get_graph_callback()
        tree_edges = self.get_tree_edges_callback()
        if G is None or tree_edges is None:
            self.graph = None
            self.cut_edges = set()
            self.stats_label.setText("No graph/tree available.")
            self.figure.clear()
            self.canvas.draw()
            return
        self.graph = G.copy()
        # All non-tree edges are initially cuts
        all_edges = set(tuple(sorted(e)) for e in self.graph.edges())
        self.cut_edges = all_edges - set(tree_edges)
        self._update_stats()
        self._draw_graph()

    def reset_cuts(self):
        self._init_graph_state()

    def on_canvas_click(self, event):
        if self.graph is None or event.inaxes is None:
            return
        # Find the closest edge to the click
        pos = self._get_layout()
        min_dist = float('inf')
        closest_edge = None
        for e in self.graph.edges():
            n1, n2 = e
            x1, y1 = pos[n1]
            x2, y2 = pos[n2]
            px, py = event.xdata, event.ydata
            # Distance from point to segment
            dx, dy = x2 - x1, y2 - y1
            if dx == dy == 0:
                continue
            t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx*dx + dy*dy)))
            proj_x, proj_y = x1 + t*dx, y1 + t*dy
            dist = np.hypot(px - proj_x, py - proj_y)
            if dist < min_dist:
                min_dist = dist
                closest_edge = tuple(sorted((n1, n2)))
        # If close enough, toggle the edge
        if closest_edge and min_dist < 0.05:
            if closest_edge in self.cut_edges:
                self.cut_edges.remove(closest_edge)
            else:
                self.cut_edges.add(closest_edge)
            self._update_stats()
            self._draw_graph()

    def _get_layout(self):
        # Use same layout as Stage 3 for consistency
        n_nodes = len(self.graph.nodes)
        if n_nodes <= 12:
            return nx.circular_layout(self.graph)
        else:
            return nx.spring_layout(self.graph, seed=42, k=1.5/np.sqrt(n_nodes), iterations=200)

    def _update_stats(self):
        n_edges = self.graph.number_of_edges()
        n_cuts = len(self.cut_edges)
        n_intact = n_edges - n_cuts
        # Compute connected components (nets)
        G_working = self.graph.copy()
        G_working.remove_edges_from(self.cut_edges)
        n_nets = nx.number_connected_components(G_working)
        self.stats_label.setText(f"Edges: {n_edges}   Cuts: {n_cuts}   Intact: {n_intact}   Nets: {n_nets}")

    def _draw_graph(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_aspect('equal')
        ax.axis('off')
        pos = self._get_layout()
        # Draw intact edges
        intact_edges = [e for e in self.graph.edges() if tuple(sorted(e)) not in self.cut_edges]
        nx.draw_networkx_edges(self.graph, pos, edgelist=intact_edges, ax=ax, edge_color='#888', width=2, alpha=0.7)
        # Draw cut edges
        nx.draw_networkx_edges(self.graph, pos, edgelist=list(self.cut_edges), ax=ax, edge_color='red', width=2.5, alpha=0.8, style='dashed')
        # Draw nodes as polygons (like Stage 3)
        mesh = self.get_mesh_callback()
        faces = mesh['faces'] if mesh else []
        verts = mesh['vertices'] if mesh else []
        for n in self.graph.nodes():
            if n >= len(faces):
                continue
            face = faces[n]
            face_verts = np.array([verts[idx] for idx in face])
            centroid = face_verts.mean(axis=0)
            centered = face_verts - centroid
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            plane = Vt[:2]
            projected = centered @ plane.T
            scale = 0.18 / (np.max(np.linalg.norm(projected, axis=1)) + 1e-8)
            projected *= scale
            px, py = projected[:,0] + pos[n][0], projected[:,1] + pos[n][1]
            # Drop shadow
            ax.fill(px+0.01, py-0.01, facecolor='gray', edgecolor='none', alpha=0.25, zorder=1)
            # Main polygon
            ax.fill(px.tolist()+[px[0]], py.tolist()+[py[0]], facecolor='#ffe066', edgecolor='#222', linewidth=2, alpha=0.95, zorder=2)
        # Draw node labels with a white outline for readability
        for n, (x, y) in pos.items():
            ax.text(x, y, str(n), fontsize=13, ha='center', va='center', color='#222', weight='bold', zorder=3,
                    path_effects=[__import__('matplotlib.patheffects').patheffects.withStroke(linewidth=3, foreground='white')])
        self.figure.tight_layout()
        self.canvas.draw()

    def _on_next(self):
        # Check for cycles in the current graph (after cuts)
        G_working = self.graph.copy()
        G_working.remove_edges_from(self.cut_edges)
        if not nx.is_forest(G_working):
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Cycles Detected", "Remove all cycles in the graph first before moving on!")
            return
        # Advance to next stage (handled by MainWindow)
        mw = self.parent()
        while mw and not hasattr(mw, 'next_stage'):
            mw = mw.parent()
        if mw and hasattr(mw, 'next_stage'):
            mw.next_stage()

    def get_current_cuts(self):
        return set(self.cut_edges)

    def get_current_nets(self):
        G_working = self.graph.copy()
        G_working.remove_edges_from(self.cut_edges)
        return list(nx.connected_components(G_working))
