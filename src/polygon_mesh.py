import numpy as np
import trimesh

class PolygonMesh:
    """
    Stores a polygonal mesh: vertices, faces (as polygons), and edges.
    Faces are lists of vertex indices (CCW order).
    """
    def __init__(self, vertices=None, faces=None):
        self.vertices = np.array(vertices) if vertices is not None else np.zeros((0, 3))
        self.faces = [list(face) for face in faces] if faces is not None else []
        self._update_edges()

    def _update_edges(self):
        self.edges = set()
        for face in self.faces:
            n = len(face)
            for i in range(n):
                v1, v2 = face[i], face[(i+1)%n]
                self.edges.add(tuple(sorted((v1, v2))))

    def as_trimesh(self):
        """Triangulate faces for trimesh/vedo display."""
        tris = []
        for face in self.faces:
            if len(face) == 3:
                tris.append(face)
            elif len(face) > 3:
                # Use trimesh to triangulate polygon
                poly = np.array([self.vertices[i] for i in face])
                try:
                    tri_faces = trimesh.triangulate_polygon([poly])
                    for tri in tri_faces:
                        tris.append([face[idx] for idx in tri])
                except Exception:
                    # fallback: fan triangulation
                    for i in range(1, len(face)-1):
                        tris.append([face[0], face[i], face[i+1]])
        return trimesh.Trimesh(vertices=self.vertices, faces=tris, process=False)

    def update(self, vertices, faces):
        self.vertices = np.array(vertices)
        self.faces = [list(face) for face in faces]
        self._update_edges()

    def copy(self):
        return PolygonMesh(np.copy(self.vertices), [list(f) for f in self.faces])
