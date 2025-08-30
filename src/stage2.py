import numpy as np
import math
import trimesh
from collections import defaultdict, deque, Counter
from PyQt5.QtWidgets import QVBoxLayout, QLabel, QSlider, QHBoxLayout, QPushButton, QWidget
from stage3 import ResponsiveWidget
from PyQt5.QtCore import Qt
from mesh_utils import simplify_mesh, merge_coplanar_faces, trimesh_to_polygonmesh
import traceback

class Stage2Widget(ResponsiveWidget):
    def __init__(self, model_viewer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_viewer = model_viewer
        self._setup_layout()
        self._merged_polygons = None
        self._update_face_count()
        if hasattr(self.model_viewer, 'model_changed_callbacks'):
            self.model_viewer.model_changed_callbacks.append(self._update_face_count)

    def _setup_layout(self):
        # Top: all controls stacked vertically
        top_container = QWidget()
        top_layout = QVBoxLayout(top_container)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(8)
        self.face_count_label = QLabel("Faces: N/A")
        top_layout.addWidget(self.face_count_label)
        slider_layout = QHBoxLayout()
        self.simplify_slider = QSlider(Qt.Horizontal)
        self.simplify_slider.setMinimum(10)
        self.simplify_slider.setMaximum(90)
        self.simplify_slider.setValue(50)
        slider_layout.addWidget(QLabel("Decimation (% faces kept):"))
        slider_layout.addWidget(self.simplify_slider)
        top_layout.addLayout(slider_layout)
        self.simplify_btn = QPushButton("Simplify")
        self.simplify_btn.clicked.connect(self._on_simplify)
        top_layout.addWidget(self.simplify_btn)
        self.merge_btn = QPushButton("Merge Coplanar Faces")
        self.merge_btn.clicked.connect(self._on_merge)
        top_layout.addWidget(self.merge_btn)
        self.next_btn = QPushButton('Next')
        top_layout.addWidget(self.next_btn)
        self.layout().addWidget(top_container)

    def _responsive_resize(self):
        min_font = 14
        max_font = 28
        btns = [self.simplify_btn, self.merge_btn, self.next_btn]
        for btn in btns:
            text_len = len(btn.text())
            btn_width = max(120, int(self.width() * 0.25))
            font_size = max(min_font, min(max_font, int(btn_width / (text_len * 1.2))))
            btn.setStyleSheet(f"font-size: {font_size}px;")
            btn.setMinimumWidth(btn_width)
            btn.setMinimumHeight(64)
        # Face count label: wrap and fit
        self.face_count_label.setWordWrap(True)
        label_width = self.width() - 40
        font_size = max(min_font, min(max_font, int(label_width / 18)))
        self.face_count_label.setStyleSheet(f"font-size: {font_size}px;")
        # Slider label
        for i in range(self.layout().count()):
            item = self.layout().itemAt(i)
            if isinstance(item, QHBoxLayout):
                for j in range(item.count()):
                    w = item.itemAt(j).widget()
                    if isinstance(w, QLabel):
                        w.setStyleSheet(f"font-size: {font_size}px;")




    def clean_mesh(self, mesh):
        mesh.compute_vertex_normals()
        mesh.remove_duplicated_vertices()
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_non_manifold_edges()
        return mesh

    def simplify_with_open3d(self, mesh, target_triangles):
        try:
            simplified = mesh.simplify_quadric_decimation(target_number_of_triangles=target_triangles)
            simplified.compute_vertex_normals()
            return simplified
        except Exception as e:
            raise RuntimeError(f"Open3D simplification failed: {str(e)}")

    def merge_coplanar_faces(self, mesh, angle_tol_deg=1.0, distance_threshold=0.01):
        # Custom grouping: group triangles by connectivity (shared edge) and normal similarity (dot product near 1 or -1)
        mesh.compute_triangle_normals()
        faces = np.asarray(mesh.triangles)
        vertices = np.asarray(mesh.vertices)
        normals = np.asarray(mesh.triangle_normals)
        n_faces = faces.shape[0]
        angle_tol_rad = math.radians(angle_tol_deg)
        cos_tol = np.cos(angle_tol_rad)
        # Build adjacency: for each triangle, which triangles share an edge
        tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        adjacency = tri_mesh.face_adjacency
        neighbors = defaultdict(list)
        for i, j in adjacency:
            n1 = normals[i]
            n2 = normals[j]
            dot = np.dot(n1, n2)
            if abs(dot) >= cos_tol:
                neighbors[i].append(j)
                neighbors[j].append(i)
        # Find connected components (clusters) in this graph
        visited = np.zeros(n_faces, dtype=bool)
        clusters = []
        for idx in range(n_faces):
            if not visited[idx]:
                cluster = []
                queue = deque([idx])
                visited[idx] = True
                while queue:
                    curr = queue.popleft()
                    cluster.append(curr)
                    for nbr in neighbors[curr]:
                        if not visited[nbr]:
                            visited[nbr] = True
                            queue.append(nbr)
                clusters.append(cluster)
        # For each cluster, extract boundary and create a single polygonal face
        merged_polygons = []
        merged_triangles = 0
        for cluster in clusters:
            if len(cluster) < 1:
                continue
            part_faces = faces[cluster]
            part = trimesh.Trimesh(vertices=vertices, faces=part_faces, process=False)
            # Use facets to get coplanar polygons (should be one per cluster)
            facets = part.facets
            facet_normals = part.facets_normal
            for facet in facets:
                # facet: indices of faces in this facet (should be all faces in part)
                facet_faces = part.faces[facet]
                # Build all edges for these faces
                edges = []
                for face in facet_faces:
                    edges.extend([
                        tuple(sorted((face[0], face[1]))),
                        tuple(sorted((face[1], face[2]))),
                        tuple(sorted((face[2], face[0])))
                    ])
                # Count edge occurrences
                edge_counts = Counter(edges)
                boundary_edges = [e for e, c in edge_counts.items() if c == 1]
                if len(boundary_edges) < 3:
                    continue
                # Order boundary edges into a loop
                # Start with any edge
                loop = [boundary_edges[0]]
                used = set(loop)
                while len(loop) < len(boundary_edges):
                    last = loop[-1][1]
                    found = False
                    for e in boundary_edges:
                        if e not in used:
                            if e[0] == last:
                                loop.append(e)
                                used.add(e)
                                found = True
                                break
                            elif e[1] == last:
                                loop.append((e[1], e[0]))
                                used.add((e[1], e[0]))
                                found = True
                                break
                    if not found:
                        break  # Can't close loop
                # Get the ordered vertex indices
                polygon = [loop[0][0]]
                for e in loop:
                    polygon.append(e[1])
                # Remove duplicates
                polygon = [polygon[0]] + [v for i, v in enumerate(polygon[1:]) if v != polygon[i]]
                if len(polygon) < 3:
                    continue
                merged_polygons.append(vertices[polygon])
            merged_triangles += len(cluster)
        print(f"[DEBUG] Coplanar merge: {n_faces} triangles, {len(merged_polygons)} merged polygons, {merged_triangles} triangles merged, {n_faces-merged_triangles} triangles unmerged.")
        return merged_polygons

    def _on_simplify(self):
        try:
            pmesh = getattr(self.model_viewer, '_last_polymesh', None)
            if pmesh is not None:
                percent = self.simplify_slider.value() / 100.0
                target_faces = max(4, int(len(pmesh.faces) * percent))
                tri_mesh = pmesh.as_trimesh()
                simp = simplify_mesh(tri_mesh, target_faces)
                simp_pmesh = trimesh_to_polygonmesh(simp)
                self.model_viewer.set_polymesh(simp_pmesh)
                print(f"Simplified: {len(pmesh.faces)} -> {len(simp_pmesh.faces)} faces (target: {target_faces})")
            self._update_face_count()
        except Exception as e:
            import traceback
            print(f"[CRITICAL] Simplification crashed: {e}\n{traceback.format_exc()}")

    def _on_merge(self):
        try:
            pmesh = getattr(self.model_viewer, '_last_polymesh', None)
            if pmesh is None:
                return
            merged = merge_coplanar_faces(pmesh, angle_tol_deg=1.0)
            self.model_viewer.set_polymesh(merged)
            self._merged_polygons = merged.faces
            print(f"[INFO] Merged coplanar faces: {len(pmesh.faces)} -> {len(merged.faces)} faces")
            self._update_face_count()
        except Exception as e:
            import traceback
            print(f"Coplanar merge crashed: {e}\n{traceback.format_exc()}")

    def _update_face_count(self):
        # Show face, vertex, and edge stats
        pmesh = getattr(self.model_viewer, '_last_polymesh', None)
        if pmesh is not None:
            n_polys = len(pmesh.faces)
            n_verts = len({tuple(v) for v in pmesh.vertices})
            # Count unique edges
            edge_set = set()
            for face in pmesh.faces:
                n = len(face)
                for i in range(n):
                    v1, v2 = face[i], face[(i+1)%n]
                    edge = tuple(sorted((v1, v2)))
                    edge_set.add(edge)
            n_edges = len(edge_set)
            self.face_count_label.setText(f"Faces: {n_polys}   Vertices: {n_verts}   Edges: {n_edges}")
        else:
            self.face_count_label.setText("Faces: N/A   Vertices: N/A   Edges: N/A")
