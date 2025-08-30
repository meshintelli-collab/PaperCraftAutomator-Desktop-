import open3d as o3d
import numpy as np
import trimesh
import math

def clean_mesh(mesh):
    mesh.compute_vertex_normals()
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    return mesh

def simplify_with_open3d(mesh, target_triangles):
    try:
        simplified = mesh.simplify_quadric_decimation(target_number_of_triangles=target_triangles)
        simplified.compute_vertex_normals()
        return simplified
    except Exception as e:
        raise RuntimeError(f"Open3D simplification failed: {str(e)}")

def remove_low_impact_vertices(mesh, angle_threshold_deg=5):

    return mesh

def merge_coplanar_faces(mesh, angle_tol_deg=1.0):
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    tmesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    # Ensure normals are 2D
    face_normals = np.asarray(tmesh.face_normals)
    if face_normals.ndim != 2:
        face_normals = face_normals.reshape((-1, 3))

    angle_tol_rad = math.radians(angle_tol_deg)

    # 🔧 This is the line that was failing
    groups = trimesh.grouping.group_vectors(face_normals, angle=angle_tol_rad)

    parts = []
    for group in groups:
        group = np.asarray(group, dtype=int)
        if len(group) == 0:
            continue

        group_faces = tmesh.faces[group]
        if group_faces.ndim != 2:
            group_faces = group_faces.reshape((-1, 3))

        part = tmesh.submesh([group_faces], append=True, repair=False)
        if part.is_empty:
            continue
        parts.append(part)

    return parts


def simplify_and_merge(mesh, method="Open3D Quadric", target_parts=100, angle_tol=1.0):
    mesh = clean_mesh(mesh)
    simplified = simplify_with_open3d(mesh, target_parts)
    simplified = remove_low_impact_vertices(simplified, angle_threshold_deg=angle_tol)

    # Triangle parts list
    triangle_parts = []
    vertices = np.asarray(simplified.vertices)
    triangles = np.asarray(simplified.triangles)
    for tri in triangles:
        tri_vertices = np.array([vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]])
        if tri_vertices.shape == (3, 3):
            triangle_parts.append(tri_vertices.copy())

    # Merged coplanar parts list (may include quads or ngons)
    merged_parts = []
    try:
        grouped = merge_coplanar_faces(simplified, angle_tol)
        for part in grouped:
            merged_parts.append(part)  # Trimesh object with one or more faces
    except Exception as e:
        print(f"Merging failed: {e}")
        merged_parts = []

    return simplified, triangle_parts, merged_parts
