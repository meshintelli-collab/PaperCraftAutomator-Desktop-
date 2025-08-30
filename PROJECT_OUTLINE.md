<!--
Methodology Document: Automatic Papercraft Tool (Python GUI Edition)
Overview

This desktop tool takes a 3D model (mesh) as input and generates a printable 2D papercraft net with glue tabs and annotations. The process is split into stages, each with a clear backend function and GUI representation.


Stage 1: Import & Preview

User Actions:
Load STL/OBJ/PLY/GLTF/3MF file via file dialog.
Rotate, pan, zoom model in a 3D viewport.
Toggle rendering modes: wireframe, face colors, auto-spin.

Backend:
Store mesh in a way that can handle complex polygons.
Store as vertices, faces, edges.
GUI Representation:
Interactive 3D preview.



Stage 2: Simplify & Merge Coplanar Faces

User Actions:
Adjust a "Simplification Level" slider, OR select a number of faces to end up with (face reduction target).
Toggle “Merge coplanar faces” button.

Backend:
Apply quadric decimation (trimesh.simplify_quadratic_decimation). applies it to the parametric model not the duplicated vertices model.
when merge coplanar faces button clicked, Detect coplanar faces which share an edge, merge into larger polygons, following the right hand rule. 
Then update the model in viewer, removing the merged triangles and replacing with the resulting polygons.
This way the number of faces in the model gets smaller, as triangles merge together.

GUI Representation:
Display current number of vertices, edges, faces in the model
Model updates in viewer and memory as decimation or simplification occurs.



Stage 3: Build Graph & Spanning Tree

User Actions:
Press “Build Graph” button. (which turns the 3d model into a 2d graph representing it faces -> nodes, edges -> edges)
Press "Generate Spanning Tree" button. this uses either depth or breadth first search, whatever it better to prevent unfolding on top of each other.
Press "Edit Spanning Tree" button. Click edges in the 2d graph representation to add/remove cuts.

Backend:
Build face adjacency graph from 3d model.
Compute spanning tree using DFS/BFS.
Mark non-tree edges as “cuts.”
Updating adjacency graph
possible for user to split graph into two sub graphs, in which case we need to carry them forwards in parallel.

GUI Representation:
Show the 3d model turning into the graph when build graph button clicked.
Animate edges disappearing as spanning tree forms.
display the updated graph as each edge is clicked on and toggled from cut to not cut.
Display number of connected components (nets).
Display number of initial edges, number of cuts, number of remaining intact edges.


Stage 4: Unfold Trees to 2D Nets

User Actions:
Press “Unfold” button.

Backend:
Pick root face for each tree.
Recursively unfold adjacent faces into 2D using hinge rotations, or some other methodology of unfolding the nets.
Track 3D→2D vertex mapping.

GUI Representation:
Starts with the graph from stage 3, in whatever state it was left in.
animation: 2D canvas shows final unfolded polygons.


Stage 5: Annotate with Tabs & Labels

User Actions:
Toggle “Show glue tabs.”
Adjust tab size (slider).

Backend:
For each cut edge pair:
Assign one side to hold tab.
Generate tab geometry (trapezoid or triangle flap).
Label edge pair with unique ID.

GUI Representation:
Tabs drawn on 2D canvas extending from polygons.
edge labels, face labels. 
edges are dotted for fold, and solid for cutting.
edge for folding (dotted) have the angle of folding annotated on them in the middle.


Stage 6: Bin Packing & Layout

User Actions:
Choose paper size (A4, A3, Letter).
Choose scaling factor if model is too large.

Backend:
Apply simple rectangle packing to distribute nets across pages.
Add page breaks as needed.

GUI Representation:
Show multi-page preview.
Nets “drop” into place one page at a time (animated).


Stage 7: Export

User Actions:
Select output format: PDF / SVG.
Press “Export.”

Backend:
Render 2D nets with annotations using reportlab (PDF) or svgwrite (SVG).
Multi-page support.

GUI Representation:
Confirmation message with file path.
Option to auto-open PDF after export.







Tech Stack Recommendations

???






Flow Summary

Import & view →

Simplify/merge →

Build graph + Auto tree + manual cuts →

Unfold →

Annotate tabs →

Bin packing →

Export (PDF/SVG).



Data Structures
class Vertex:
    id: int
    coords: np.ndarray  # [x,y,z]

class Edge:
    id: int
    v1: int
    v2: int
    faces: list  # adjacent faces

class Face:
    id: int
    vertices: list[int]  # ordered CCW
    normal: np.ndarray
    edges: list[int]

class Graph:
    faces: dict[int, Face]
    edges: dict[int, Edge]
    vertices: dict[int, Vertex]





UNFOLDING ALGORITHM STEPS

Here’s a clear breakdown of the 3D-to-2D unfolding algorithm for a papercraft net, tailored to your project’s data structures and workflow:

1. Input
Mesh: List of faces (polygons), each with ordered vertex indices and 3D coordinates.
Spanning Tree(s): For each connected component, a tree structure where each node is a face and edges represent non-cut (hinge) connections.
2. Algorithm Steps
a. Choose Root Face
For each tree, pick a root face (e.g., the largest, or just the first in the list).
b. Place Root Face in 2D
Project the root face’s 3D vertices onto the 2D plane (e.g., XY plane), preserving edge lengths.
c. Recursively Unfold Adjacent Faces
For each neighbor face connected by a hinge (tree edge):
Identify Shared Edge: Find the two shared vertices between the parent and child face.
Compute Hinge Axis: The shared edge in 3D is the axis of rotation.
Calculate Rotation: Determine the dihedral angle between the parent and child face normals.
Transform Child Face:
Rotate the child face around the shared edge so it lies flat in 2D, attached to the parent.
Place the shared edge in the same 2D location as in the parent.
Map the rest of the child’s vertices using the rotation.
Repeat: Recursively unfold all unplaced neighbors.
d. Track 3D→2D Mapping
Maintain a mapping from each 3D vertex to its 2D position for each face as it is placed.
e. Handle Multiple Trees
Repeat the above for each connected component (forest), producing multiple 2D nets if needed.
3. Output
For each tree: a set of 2D polygons (nets), each corresponding to a face, with shared edges aligned.
4. Notes & Best Practices
Edge Consistency: Always align shared edges exactly in 2D to avoid gaps/overlaps.
Numerical Stability: Use robust geometric calculations to avoid drift as you recurse.
Cycles: The spanning tree ensures no cycles, so unfolding is always possible without overlap.
Disconnected Components: Each tree produces a separate net.
Summary:
The algorithm is a recursive “hinge unfolding” process, starting from a root face and laying out each adjacent face in 2D by rotating it around the shared edge, using the spanning tree to avoid cycles and overlaps.



-->
