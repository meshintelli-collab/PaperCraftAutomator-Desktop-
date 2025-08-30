import numpy as np
import trimesh

import open3d as o3d

from polygon_mesh import PolygonMesh


def clean_and_weld_mesh(mesh, weld_tol=1e-5):
    # Remove duplicate/close vertices and update faces
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    # Weld vertices: snap close vertices together
    from scipy.spatial import cKDTree
    kdtree = cKDTree(verts)
    groups = kdtree.query_ball_point(verts, weld_tol)
    # Map each vertex to its group leader
    leader = np.arange(len(verts))
    for i, group in enumerate(groups):
        min_idx = min(group)
        leader[i] = min_idx
    # Remap all vertices to their leader
    unique_map = {old: new for old, new in enumerate(leader)}
    verts_new = []
    old_to_new = {}
    for i, idx in enumerate(leader):
        if idx == i:
            old_to_new[i] = len(verts_new)
            verts_new.append(verts[i])
        else:
            old_to_new[i] = old_to_new[idx]
    verts_new = np.array(verts_new)
    # Remap faces
    faces_new = np.vectorize(lambda x: old_to_new[x])(faces)
    # Remove degenerate (zero-area or duplicate-vertex) faces
    faces_clean = []
    for f in faces_new:
        if len(set(f)) < 3:
            continue
        v0, v1, v2 = verts_new[f[0]], verts_new[f[1]], verts_new[f[2]]
        area = np.linalg.norm(np.cross(v1-v0, v2-v0)) * 0.5
        if area < 1e-10:
            continue
        faces_clean.append(f)
    faces_clean = np.array(faces_clean)
    mesh_clean = trimesh.Trimesh(vertices=verts_new, faces=faces_clean, process=False)
    mesh_clean.remove_unreferenced_vertices()
    return mesh_clean

def load_mesh(filename):
    mesh = trimesh.load(filename, force='mesh')
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Loaded file is not a mesh.")
    mesh = clean_and_weld_mesh(mesh)
    return mesh

def trimesh_to_polygonmesh(mesh):
    verts = np.asarray(mesh.vertices)
    faces = [list(face) for face in mesh.faces]
    return PolygonMesh(verts, faces)

def simplify_mesh(mesh, target_faces):
    # Convert trimesh to Open3D mesh
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    o3d_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces))
    o3d_mesh.compute_vertex_normals()
    simp = o3d_mesh.simplify_quadric_decimation(target_number_of_triangles=target_faces)
    simp.compute_vertex_normals()
    # Convert back to trimesh
    simp_trimesh = trimesh.Trimesh(vertices=np.asarray(simp.vertices), faces=np.asarray(simp.triangles), process=False)
    return simp_trimesh

def merge_coplanar_faces(pmesh, angle_tol_deg=1.0):
    # Improved coplanar merge: group faces by normal and edge connectivity, then extract ordered boundary
    verts = pmesh.vertices
    faces = pmesh.faces
    # Compute normals for each face
    normals = []
    for face in faces:
        v0, v1, v2 = verts[face[0]], verts[face[1]], verts[face[2]]
        n = np.cross(v1-v0, v2-v0)
        n = n / (np.linalg.norm(n) + 1e-8)
        normals.append(n)
    normals = np.array(normals)
    # Use a small but nonzero angle tolerance to allow for floating point errors
    min_angle_tol_deg = 0.01
    angle_tol_deg = max(angle_tol_deg, min_angle_tol_deg)
    cos_tol = np.cos(np.radians(angle_tol_deg))
    # Build edge-to-face map
    edge_to_faces = {}
    for idx, face in enumerate(faces):
        n = len(face)
        for k in range(n):
            e = tuple(sorted((face[k], face[(k+1)%n])))
            edge_to_faces.setdefault(e, set()).add(idx)
    # Build adjacency: faces are adjacent if they share an edge and are coplanar
    adj = {i: set() for i in range(len(faces))}
    for e, face_idxs in edge_to_faces.items():
        if len(face_idxs) == 2:
            i, j = tuple(face_idxs)
            if np.dot(normals[i], normals[j]) >= cos_tol:
                adj[i].add(j)
                adj[j].add(i)
    # Find connected components (groups of coplanar, edge-connected faces)
    groups = []
    visited = set()
    for i in range(len(faces)):
        if i in visited:
            continue
        group = set()
        queue = [i]
        while queue:
            curr = queue.pop()
            if curr in group:
                continue
            group.add(curr)
            visited.add(curr)
            queue.extend(adj[curr] - group)
        groups.append(sorted(group))
    # For each group, extract boundary and order it into a polygon
    #print(f"[DEBUG] Found {len(groups)} coplanar groups.")
    merged_faces = []
    for i, group in enumerate(groups):
        
        if len(group) == 1:
            merged_faces.append(faces[group[0]])
            continue
        else:
            #print(f"[DEBUG]  Group {i}: {len(group)} faces.")
            # Collect all edges and count occurrences
            edge_count = {}
            for idx in group:
                f = faces[idx]
                n = len(f)
                for k in range(n):
                    e = (f[k], f[(k+1)%n])
                    e_sorted = tuple(sorted(e))
                    edge_count[e_sorted] = edge_count.get(e_sorted, 0) + 1
            # Check for non-manifold edges (edges shared by >2 faces)
            nonmanifold_edges = [e for e, c in edge_count.items() if c > 2]
            if nonmanifold_edges:
                #print(f"[DEBUG]   Skipping group {i} (non-manifold edges detected: {len(nonmanifold_edges)})")
                for e in nonmanifold_edges:
                    pass  # debug print placeholder
                # Keep original triangles
                for idx in group:
                    merged_faces.append(faces[idx])
                continue
        # Boundary edges appear only once
        boundary_edges = [e for e, c in edge_count.items() if c == 1]
        if not boundary_edges or len(boundary_edges) < 3:
            #print(f"[DEBUG]   Skipping group {i} (no valid boundary or open boundary detected)")
            # Keep original triangles
            for idx in group:
                merged_faces.append(faces[idx])
            continue
        # --- Robust boundary extraction and polygon formation ---
        # 1. Count all directed edges in the group
        from collections import defaultdict
        edge_count = defaultdict(int)
        edge_to_face = defaultdict(list)
        for idx in group:
            f = faces[idx]
            n = len(f)
            for k in range(n):
                v1, v2 = f[k], f[(k+1)%n]
                edge = tuple(sorted((v1, v2)))
                edge_count[edge] += 1
                edge_to_face[edge].append(idx)
        # 2. Boundary edges are those that appear only once
        boundary_edges = [e for e, c in edge_count.items() if c == 1]
        if not boundary_edges:
            #print(f"[DEBUG]   No boundary found for group {i}, keeping original triangles")
            for idx in group:
                merged_faces.append(faces[idx])
            continue
        # 3. Build a map from vertex to its outgoing boundary edge(s)
        vertex_to_next = defaultdict(list)
        for e in boundary_edges:
            # Recover direction from original faces
            found = False
            for idx in group:
                f = faces[idx]
                n = len(f)
                for k in range(n):
                    v1, v2 = f[k], f[(k+1)%n]
                    if tuple(sorted((v1, v2))) == e:
                        vertex_to_next[v1].append(v2)
                        found = True
                        break
                if found:
                    break
        # 4. Walk the boundary loop
        start = boundary_edges[0][0]
        polygon = [start]
        curr = start
        for _ in range(len(boundary_edges)):
            nexts = vertex_to_next.get(curr, [])
            if not nexts:
                break
            nxt = nexts.pop(0)
            if nxt == polygon[0]:
                polygon.append(nxt)
                break
            if nxt in polygon:
                break
            polygon.append(nxt)
            curr = nxt
        # Remove duplicate at end if closed
        if len(polygon) > 1 and polygon[0] == polygon[-1]:
            polygon = polygon[:-1]
        if len(polygon) >= 3 and len(set(polygon)) == len(polygon):
            merged_faces.append(polygon)
            #print(f"[DEBUG]   Merged polygon with {len(polygon)} vertices.")
        else:
            #print(f"[DEBUG]   Skipping group {i} (could not robustly extract polygon, keeping original triangles)")
            for idx in group:
                merged_faces.append(faces[idx])
    #print(f"[DEBUG] Output mesh: {len(merged_faces)} faces.")
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
