from PyQt5.QtWidgets import QVBoxLayout

from PyQt5.QtWidgets import QWidget

class ResponsiveVBox(QVBoxLayout):
    """A QVBoxLayout that automatically scales font and button sizes for its children widgets."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._parent_widget = parent

    def eventFilter(self, obj, event):
        if event.type() == event.Resize and self._parent_widget:
            self._parent_widget._responsive_resize()
        return super().eventFilter(obj, event)

class ResponsiveWidget(QWidget):
    """A QWidget that uses ResponsiveVBox and auto-scales its children."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._responsive_layout = ResponsiveVBox(self)
        self.setLayout(self._responsive_layout)
    def _responsive_resize(self):
        # Override in subclasses to scale fonts/buttons
        pass
    def resizeEvent(self, event):
        self._responsive_resize()
        super().resizeEvent(event)

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSizePolicy, QMessageBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import networkx as nx
import numpy as np

class Stage3Widget(ResponsiveWidget):
    def _get_layout(self, G):
        n_nodes = len(G.nodes)
        if n_nodes <= 12:
            pos = nx.circular_layout(G)
        else:
            pos = nx.spring_layout(G, seed=42, k=1.5/np.sqrt(n_nodes), iterations=200)
        return pos
    def __init__(self, get_mesh_callback, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.get_mesh_callback = get_mesh_callback
        self.graph = None
        self.spanning_tree = None
        self._last_tree_edges = set()
        self._last_cut_edges = set()
        self._setup_layout()

    def _setup_layout(self):
        # Top: status and buttons in a vertical group
        top_container = QWidget()
        top_layout = QVBoxLayout(top_container)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(8)
        self.status_label = QLabel("")
        self.status_label.setStyleSheet('font-weight: bold;')
        top_layout.addWidget(self.status_label)
        self.build_graph_btn = QPushButton("Build Face Adjacency Graph")
        self.build_graph_btn.clicked.connect(self.show_graph)
        self.tree_btn = QPushButton("Generate Auto Spanning Tree")
        self.tree_btn.clicked.connect(self.show_tree)
        self.edit_btn = QPushButton("Edit Spanning Tree")
        self.edit_btn.setCheckable(True)
        self.edit_btn.toggled.connect(self.toggle_edit_mode)
        top_layout.addWidget(self.build_graph_btn)
        top_layout.addWidget(self.tree_btn)
        top_layout.addWidget(self.edit_btn)
        # Add Next button at the bottom
        self.next_btn = QPushButton("Next")
        top_layout.addWidget(self.next_btn)
        self.layout().addWidget(top_container)
        # Middle: viewer (matplotlib canvas)
        self.figure = Figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setMinimumHeight(350)
        self.layout().addWidget(self.canvas)
        self.canvas.mpl_connect('button_press_event', self.on_canvas_click)
        self.edit_mode = False
        self.cut_edges = set()

    def _responsive_resize(self):
        min_font = 14
        max_font = 28
        btns = [self.build_graph_btn, self.tree_btn, self.edit_btn, self.next_btn]
        for btn in btns:
            text_len = len(btn.text())
            btn_width = max(120, int(self.width() * 0.25))
            font_size = max(min_font, min(max_font, int(btn_width / (text_len * 1.2))))
            btn.setStyleSheet(f"font-size: {font_size}px;")
            btn.setMinimumWidth(btn_width)
            btn.setMinimumHeight(64)
        # Status label: wrap and fit
        self.status_label.setWordWrap(True)
        label_width = self.width() - 40
        font_size = max(min_font, min(max_font, int(label_width / 18)))
        self.status_label.setStyleSheet('font-weight: bold; font-size: {}px;'.format(font_size))

    def toggle_edit_mode(self, checked):
        self.edit_mode = checked
        if self.edit_mode:
            self.cut_edges = set(self._last_cut_edges)
            self._draw_graph(self.graph, tree_edges=None)
        else:
            # Exiting edit mode: allow any forest (multiple trees), but no cycles
            if self.graph is not None and self.spanning_tree is not None:
                all_edges = set(tuple(sorted(e)) for e in self.graph.edges())
                tree_edges = all_edges - self.cut_edges
                G2 = self.graph.copy()
                G2.remove_edges_from(self.cut_edges)
                # Check for cycles: if any connected component is not a tree, block
                has_cycle = False
                for component in nx.connected_components(G2):
                    subgraph = G2.subgraph(component)
                    if not nx.is_tree(subgraph):
                        has_cycle = True
                        break
                if has_cycle:
                    QMessageBox.warning(self, "Invalid Spanning Forest", "No cycles allowed! Please ensure your cuts do not create any cycles.")
                    # Do not update the tree/cuts, revert to previous
                    self._draw_graph(self.graph, tree_edges=self._last_tree_edges)
                    return
                # Valid forest: update tree/cuts
                self.spanning_tree = G2  # Now a forest, not a single tree
                self._last_tree_edges = set(tree_edges)
                self._last_cut_edges = all_edges - tree_edges
                self.status_label.setText(f"Spanning forest: {len(tree_edges)} edges, {len(self.graph.edges())-len(tree_edges)} cuts.")
                self._draw_graph(self.graph, tree_edges=tree_edges)

    def on_canvas_click(self, event):
        if not self.edit_mode or self.graph is None or event.inaxes is None:
            return
        pos = self._get_layout(self.graph)
        min_dist = float('inf')
        closest_edge = None
        for e in self.graph.edges():
            n1, n2 = e
            x1, y1 = pos[n1]
            x2, y2 = pos[n2]
            px, py = event.xdata, event.ydata
            dx, dy = x2 - x1, y2 - y1
            if dx == dy == 0:
                continue
            t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx*dx + dy*dy)))
            proj_x, proj_y = x1 + t*dx, y1 + t*dy
            dist = np.hypot(px - proj_x, py - proj_y)
            if dist < min_dist:
                min_dist = dist
                closest_edge = tuple(sorted((n1, n2)))
        if closest_edge and min_dist < 0.05:
            if closest_edge in self.cut_edges:
                self.cut_edges.remove(closest_edge)
            else:
                self.cut_edges.add(closest_edge)
            self._draw_graph(self.graph, tree_edges=None)

    @property
    def viewer_widget(self):
        return self.canvas

    def clear_graph(self):
        self.graph = None
        self.spanning_tree = None
        self.figure.clear()
        self.canvas.draw()



    # Removed showEvent/hideEvent to avoid .show()/.hide() on the 3D viewer, which causes extra window bugs


    def show_graph(self):
        mesh = self.get_mesh_callback()
        if mesh is None:
            self.status_label.setText("No mesh loaded.")
            self.clear_graph()
            return
        faces = mesh['faces']
        G = nx.Graph()
        for i, f1 in enumerate(faces):
            G.add_node(i)
        edge_to_faces = {}
        for i, f in enumerate(faces):
            n = len(f)
            for j in range(n):
                e = tuple(sorted((f[j], f[(j+1)%n])))
                edge_to_faces.setdefault(e, []).append(i)
        for edge, adj_faces in edge_to_faces.items():
            if len(adj_faces) == 2:
                G.add_edge(adj_faces[0], adj_faces[1], edge=edge)
        self.graph = G
        self.spanning_tree = None
        self.status_label.setText(f"Graph built: {G.number_of_nodes()} faces, {G.number_of_edges()} adjacencies.")
        self._draw_graph(G)

    def show_tree(self):
        if self.graph is None:
            self.status_label.setText("Build the graph first.")
            return
        # Find node with highest degree
        degrees = dict(self.graph.degree())
        if not degrees:
            self.status_label.setText("Graph is empty.")
            return
        start_node = max(degrees, key=degrees.get)
        # Use BFS tree from that node
        T = nx.bfs_tree(self.graph, source=start_node)
        tree_edges = set(tuple(sorted(e)) for e in T.edges())
        all_edges = set(tuple(sorted(e)) for e in self.graph.edges())
        cut_edges = all_edges - tree_edges
        self.spanning_tree = T
        self._last_tree_edges = set(tree_edges)
        self._last_cut_edges = set(cut_edges)
        self.status_label.setText(f"Spanning tree (BFS from node {start_node}): {len(tree_edges)} tree edges, {len(cut_edges)} cuts.")
        self._draw_graph(self.graph, tree_edges=tree_edges)

    def _draw_graph(self, G, tree_edges=None):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_aspect('equal')
        ax.axis('off')
        pos = self._get_layout(G)
        # Draw all edges in base color
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#888', width=1.5, alpha=0.7)
        # Edit mode: highlight tree edges in red dashed, non-tree edges in gray
        def filter_edges(edges):
            return [e for e in edges if e[0] in pos and e[1] in pos]
        if self.edit_mode:
            # In edit mode, tree edges are the ones NOT in cut_edges
            all_edges = set(tuple(sorted(e)) for e in G.edges())
            tree_edges = all_edges - self.cut_edges
            non_tree_edges = self.cut_edges
            # Draw non-tree edges in base color (gray)
            if non_tree_edges:
                nx.draw_networkx_edges(G, pos, edgelist=filter_edges(non_tree_edges), ax=ax, edge_color='#888', width=1.5, alpha=0.7)
            # Draw tree edges in red dashed
            if tree_edges:
                nx.draw_networkx_edges(G, pos, edgelist=filter_edges(tree_edges), ax=ax, edge_color='red', width=2.5, style='dashed', alpha=0.95)
        else:
            if not tree_edges:
                tree_edges = self._last_tree_edges
            if tree_edges:
                nx.draw_networkx_edges(G, pos, edgelist=filter_edges(tree_edges), ax=ax, edge_color='orange', width=3, alpha=0.9)
        # Draw nodes as polygons with a drop shadow for visual pop
        mesh = self.get_mesh_callback()
        faces = mesh['faces'] if mesh else []
        verts = mesh['vertices'] if mesh else []
        for n in G.nodes():
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


    def generate_spanning_tree(self):
        if self.graph is None:
            self.status_label.setText("Build the graph first.")
            return
        T = nx.dfs_tree(self.graph, source=0)
        tree_edges = set(tuple(sorted(e)) for e in T.edges())
        all_edges = set(tuple(sorted(e)) for e in self.graph.edges())
        cut_edges = all_edges - tree_edges
        self.spanning_tree = T
        self.status_label.setText(f"Spanning tree: {len(tree_edges)} tree edges, {len(cut_edges)} cuts.")
        self._show_graph(tree_edges=tree_edges)
