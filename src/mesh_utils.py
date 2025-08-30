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

def unfold_to_2d_nets(graph, mesh):
    import networkx as nx
    if graph is None or mesh is None:
        return []
    faces = mesh.faces if hasattr(mesh, 'faces') else mesh['faces']
    vertices = mesh.vertices if hasattr(mesh, 'vertices') else mesh['vertices']
    nets = []
    net_offset_x = 0.0
    for component in nx.connected_components(graph):
        root = next(iter(component))
        placed = {}
        face_to_parent = {}
        face_to_shared = {}
        from collections import deque, defaultdict
        queue = deque([root])
        visited = set([root])
        layer_map = {root: 0}
        bfs_layers = defaultdict(list)
        while queue:
            fidx = queue.popleft()
            for nbr in graph.neighbors(fidx):
                if nbr not in visited:
                    f1, f2 = faces[fidx], faces[nbr]
                    shared = list(set(f1) & set(f2))
                    if len(shared) == 2:
                        idxs_f1 = [f1.index(shared[0]), f1.index(shared[1])]
                        idxs_f2 = [f2.index(shared[0]), f2.index(shared[1])]
                        face_to_parent[nbr] = fidx
                        face_to_shared[nbr] = (idxs_f2, idxs_f1, shared)
                        queue.append(nbr)
                        visited.add(nbr)
                        layer_map[nbr] = layer_map[fidx] + 1
        def best_fit_plane(points):
            centroid = points.mean(axis=0)
            uu, dd, vv = np.linalg.svd(points - centroid)
            normal = vv[2]
            return centroid, normal
        def flatten_face(face_idx, ref_2d=None, ref_3d=None, shared_idxs=None):
            face = faces[face_idx]
            verts3d = vertices[face]
            if ref_2d is None:
                centroid, normal = best_fit_plane(verts3d)
                z_axis = np.array([0,0,1])
                axis = np.cross(normal, z_axis)
                angle = np.arccos(np.clip(np.dot(normal, z_axis), -1, 1))
                if np.linalg.norm(axis) < 1e-8:
                    R = np.eye(3)
                else:
                    axis = axis / np.linalg.norm(axis)
                    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
                    R = np.eye(3) + np.sin(angle)*K + (1-np.cos(angle))*(K@K)
                verts3d_flat = (verts3d - centroid) @ R.T
                verts2d = verts3d_flat[:,:2]
                placed[face_idx] = verts2d
                return verts2d
            idxs_child, idxs_parent, shared = shared_idxs
            parent_face = faces[face_to_parent[face_idx]]
            parent_2d = ref_2d
            # Find the ordered shared edge in parent
            for i in range(len(parent_face)):
                if parent_face[i] == shared[0] and parent_face[(i+1)%len(parent_face)] == shared[1]:
                    parent_edge_idx = (i, (i+1)%len(parent_face))
                    break
                if parent_face[i] == shared[1] and parent_face[(i+1)%len(parent_face)] == shared[0]:
                    parent_edge_idx = ((i+1)%len(parent_face), i)
                    break
            else:
                return None
            # Find the ordered shared edge in child
            for i in range(len(face)):
                if face[i] == shared[0] and face[(i+1)%len(face)] == shared[1]:
                    child_edge_idx = (i, (i+1)%len(face))
                    break
                if face[i] == shared[1] and face[(i+1)%len(face)] == shared[0]:
                    child_edge_idx = ((i+1)%len(face), i)
                    break
            else:
                return None
            # Get 2D coordinates of parent edge
            vA2, vB2 = parent_2d[parent_edge_idx[0]], parent_2d[parent_edge_idx[1]]
            # Flatten child face to XY
            centroid, normal = best_fit_plane(verts3d)
            z_axis = np.array([0,0,1])
            axis = np.cross(normal, z_axis)
            angle = np.arccos(np.clip(np.dot(normal, z_axis), -1, 1))
            if np.linalg.norm(axis) < 1e-8:
                R = np.eye(3)
            else:
                axis = axis / np.linalg.norm(axis)
                K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
                R = np.eye(3) + np.sin(angle)*K + (1-np.cos(angle))*(K@K)
            verts3d_flat = (verts3d - verts3d[child_edge_idx[0]]) @ R.T
            verts2d = verts3d_flat[:,:2]
            # Get child edge in 2D
            vA2c, vB2c = verts2d[child_edge_idx[0]], verts2d[child_edge_idx[1]]
            # If direction is opposite, flip child face
            parent_vec = vB2 - vA2
            child_vec = vB2c - vA2c
            if np.linalg.norm(child_vec) < 1e-8 or np.linalg.norm(parent_vec) < 1e-8:
                return None
            if np.dot(parent_vec, child_vec) < 0:
                # Flip child face in 2D (reverse order)
                verts2d = verts2d[::-1]
                vA2c, vB2c = verts2d[child_edge_idx[1]], verts2d[child_edge_idx[0]]
                child_vec = vB2c - vA2c
            # Rotate child so edge aligns with parent edge
            angle2 = np.arctan2(parent_vec[1], parent_vec[0]) - np.arctan2(child_vec[1], child_vec[0])
            R2 = np.array([[np.cos(angle2), -np.sin(angle2)], [np.sin(angle2), np.cos(angle2)]])
            verts2d = verts2d @ R2.T
            # Translate so both endpoints match
            offset = vA2 - verts2d[child_edge_idx[0]]
            verts2d = verts2d + offset
            # After transform, ensure both endpoints match exactly
            verts2d[child_edge_idx[0]] = vA2
            verts2d[child_edge_idx[1]] = vB2
            # Ensure face is folded outwards (not back over parent):
            n = len(face)
            next_child_idx = (child_edge_idx[1]+1)%n
            vA = vA2
            vB = vB2
            vC = verts2d[next_child_idx]
            # --- Robust fold direction check ---
            # 3D: get parent and child normals, shared edge direction
            parent_face_3d = vertices[parent_face]
            child_face_3d = verts3d
            # Get parent normal
            def face_normal(verts):
                v0, v1, v2 = verts[0], verts[1], verts[2]
                return np.cross(v1-v0, v2-v0)
            n_parent = face_normal(parent_face_3d)
            n_child = face_normal(child_face_3d)
            edge3d = verts3d[child_edge_idx[1]] - verts3d[child_edge_idx[0]]
            # Dihedral sign: positive if child is folded "outwards" from parent
            dihedral_sign = np.sign(np.dot(np.cross(n_parent, n_child), edge3d))
            # 2D: check which side vC is on relative to parent edge
            cross2d = (vB[0]-vA[0])*(vC[1]-vA[1]) - (vB[1]-vA[1])*(vC[0]-vA[0])
            fold_sign_2d = np.sign(cross2d)
            # If 2D fold direction does not match 3D, reflect across edge
            if fold_sign_2d != dihedral_sign and fold_sign_2d != 0 and dihedral_sign != 0:
                # Reflect verts2d across edge vA-vB
                edge_dir = (vB-vA)/np.linalg.norm(vB-vA)
                perp = np.array([-edge_dir[1], edge_dir[0]])
                verts2d = verts2d - vA
                verts2d = verts2d - 2*np.dot(verts2d, perp)[:,None]*perp
                verts2d = verts2d + vA
            placed[face_idx] = verts2d
            return verts2d
        # Place root
        flatten_face(root)
        # Place all children breadth-first and group by BFS layer
        queue = deque([root])
        visited = set([root])
        bfs_layers = defaultdict(list)
        while queue:
            fidx = queue.popleft()
            poly = placed[fidx]
            bfs_layers[layer_map[fidx]].append(poly)
            for nbr in graph.neighbors(fidx):
                if nbr not in visited and nbr in face_to_parent:
                    shared_idxs = face_to_shared[nbr]
                    ref_2d = placed[face_to_parent[nbr]]
                    flatten_face(nbr, ref_2d, None, shared_idxs)
                    queue.append(nbr)
                    visited.add(nbr)
        # Sort layers by BFS order
        bfs_layers_sorted = [bfs_layers[i] for i in sorted(bfs_layers.keys())]
        polygons = [placed[fidx] for fidx in placed]
        # Center and scale to fit in [-1,1], then offset horizontally
        all_pts = np.vstack(polygons)
        min_xy = all_pts.min(axis=0)
        max_xy = all_pts.max(axis=0)
        center = (min_xy + max_xy) / 2
        scale = 1.8 / max(max_xy - min_xy) if np.any(max_xy - min_xy > 0) else 1.0
        polygons = [(poly - center) * scale + np.array([net_offset_x, 0]) for poly in polygons]
        # Also scale/offset each layer for animation
        bfs_layers_scaled = []
        for layer in bfs_layers_sorted:
            layer_polys = [(np.array(poly) - center) * scale + np.array([net_offset_x, 0]) for poly in layer]
            bfs_layers_scaled.append(layer_polys)
        net_offset_x += (max_xy[0] - min_xy[0]) * scale + 1.0
        nets.append({'polygons': polygons, 'bfs_layers': bfs_layers_scaled})
    return nets
