import numpy as np
import trimesh

from polygon_mesh import PolygonMesh

def load_mesh(filename):
    mesh = trimesh.load(filename, force='mesh')
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Loaded file is not a mesh.")
    return mesh

def trimesh_to_polygonmesh(mesh):
    verts = np.asarray(mesh.vertices)
    faces = [list(face) for face in mesh.faces]
    return PolygonMesh(verts, faces)

def simplify_mesh(mesh, target_faces):
    simp = mesh.simplify_quadric_decimation(target_faces)
    return simp

def merge_coplanar_faces(pmesh, angle_tol_deg=1.0):
    # Custom coplanar merge: group faces by normal and adjacency
    verts = pmesh.vertices
    faces = pmesh.faces
    normals = []
    for face in faces:
        v0, v1, v2 = verts[face[0]], verts[face[1]], verts[face[2]]
        n = np.cross(v1-v0, v2-v0)
        n = n / (np.linalg.norm(n) + 1e-8)
        normals.append(n)
    normals = np.array(normals)
    # Group faces by normal similarity
    groups = []
    used = set()
    cos_tol = np.cos(np.radians(angle_tol_deg))
    for i, ni in enumerate(normals):
        if i in used:
            continue
        group = [i]
        used.add(i)
        for j in range(i+1, len(normals)):
            if j in used:
                continue
            if np.dot(ni, normals[j]) >= cos_tol:
                # Check adjacency
                if _faces_share_edge(faces[i], faces[j]):
                    group.append(j)
                    used.add(j)
        groups.append(group)
    # Merge faces in each group
    merged_faces = []
    for group in groups:
        if len(group) == 1:
            merged_faces.append(faces[group[0]])
        else:
            # Merge polygons by union of their edges
            merged = _merge_faces([faces[idx] for idx in group])
            if merged is not None:
                merged_faces.append(merged)
    return PolygonMesh(verts, merged_faces)

def _faces_share_edge(f1, f2):
    s1 = set((f1[i], f1[(i+1)%len(f1)]) for i in range(len(f1)))
    s2 = set((f2[i], f2[(i+1)%len(f2)]) for i in range(len(f2)))
    return len(s1 & s2) > 0

def _merge_faces(faces):
    # Naive: flatten all vertices, remove duplicates, order CCW
    verts = [v for face in faces for v in face]
    uniq = []
    for v in verts:
        if v not in uniq:
            uniq.append(v)
    return uniq
