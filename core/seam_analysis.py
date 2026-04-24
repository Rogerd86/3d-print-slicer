"""
seam_analysis.py
Computes per-vertex seam quality scores and detects natural crease lines.

Scores:
  0.0 = perfect seam location (flat surface, panel crease, body line)
  1.0 = bad seam location (smooth curved surface — hard to hide/fill)

Used for:
  1. Heatmap overlay in the viewport (green=good, red=bad)
  2. Natural seam suggestion — highlight crease lines as suggested cut positions
"""
import numpy as np
import trimesh
from typing import Tuple, List, Optional


def compute_seam_scores(mesh: trimesh.Trimesh) -> np.ndarray:
    """
    Compute per-vertex seam quality score (0=good, 1=bad).

    A good seam location is:
    - On a flat surface (easy to sand/fill)
    - On an existing crease or panel line (seam is hidden by the existing edge)

    A bad seam location is:
    - On a smoothly curving surface (seam is visible and hard to disguise)
    """
    n_verts = len(mesh.vertices)
    faces = mesh.faces
    face_normals = mesh.face_normals

    # Step 1: Compute mean face normal per vertex (vectorised)
    # Accumulate face normals to each vertex
    vert_normal_sum = np.zeros((n_verts, 3), dtype=np.float64)
    vert_face_count = np.zeros(n_verts, dtype=np.float64)

    for col in range(3):  # iterate face vertex columns, not individual faces
        vi = faces[:, col]
        np.add.at(vert_normal_sum, vi, face_normals)
        np.add.at(vert_face_count, vi, 1.0)

    # Avoid divide by zero
    vert_face_count = np.maximum(vert_face_count, 1.0)
    mean_normals = vert_normal_sum / vert_face_count[:, np.newaxis]
    norms = np.linalg.norm(mean_normals, axis=1, keepdims=True)
    mean_normals = mean_normals / np.where(norms == 0, 1, norms)

    # Compute variance: average of (1 - dot(face_normal, mean_normal)) per vertex
    # This measures how much surrounding faces disagree → curvature proxy
    curvature_sum = np.zeros(n_verts, dtype=np.float64)
    for col in range(3):
        vi = faces[:, col]
        dots = np.clip(np.sum(face_normals * mean_normals[vi], axis=1), -1, 1)
        np.add.at(curvature_sum, vi, 1.0 - dots)

    curvature = curvature_sum / vert_face_count

    # Step 2: Crease bonus — vertices on sharp edges are GOOD seam locations
    crease_bonus = np.zeros(n_verts)
    try:
        adj    = mesh.face_adjacency
        fn     = mesh.face_normals
        dots   = np.clip(np.sum(fn[adj[:, 0]] * fn[adj[:, 1]], axis=1), -1, 1)
        angles = np.degrees(np.arccos(dots))

        # Edges with face-angle difference > 20° are creases
        sharp_mask  = angles > 20.0
        sharp_edges = mesh.face_adjacency_edges[sharp_mask]
        sharp_angles = angles[sharp_mask]

        # Compute bonus per edge
        bonuses = np.minimum(0.8, sharp_angles / 120.0)

        # Apply to both vertices of each sharp edge
        np.maximum.at(crease_bonus, sharp_edges[:, 0], bonuses)
        np.maximum.at(crease_bonus, sharp_edges[:, 1], bonuses)

    except Exception as e:
        print(f"Crease detection warning: {e}")

    # Final score
    score = np.clip(curvature - crease_bonus, 0.0, 1.0)
    mx = score.max()
    if mx > 0:
        score /= mx

    return score


def find_natural_seams(mesh: trimesh.Trimesh,
                        min_angle: float = 25.0,
                        min_edge_length: float = 5.0) -> List[dict]:
    """
    Find natural crease lines on the mesh that are good cut positions.

    Returns list of seam dicts:
      { 'v0', 'v1': edge vertex positions,
        'normal': average face normal,
        'angle': dihedral angle at this edge,
        'midpoint', 'axis': dominant axis, 'length' }
    """
    seams = []
    try:
        adj    = mesh.face_adjacency
        fn     = mesh.face_normals
        edges  = mesh.face_adjacency_edges
        verts  = mesh.vertices
        dots   = np.clip(np.sum(fn[adj[:, 0]] * fn[adj[:, 1]], axis=1), -1, 1)
        angles = np.degrees(np.arccos(dots))

        crease_mask = angles >= min_angle
        crease_indices = np.where(crease_mask)[0]

        if len(crease_indices) == 0:
            return seams

        # Vectorised edge processing
        crease_edges = edges[crease_indices]
        p0s = verts[crease_edges[:, 0]]
        p1s = verts[crease_edges[:, 1]]
        edge_vecs = p1s - p0s
        edge_lens = np.linalg.norm(edge_vecs, axis=1)

        # Filter by minimum length
        long_mask = edge_lens >= min_edge_length
        crease_indices = crease_indices[long_mask]
        crease_edges = crease_edges[long_mask]
        p0s = p0s[long_mask]
        p1s = p1s[long_mask]
        edge_vecs = edge_vecs[long_mask]
        edge_lens = edge_lens[long_mask]

        if len(crease_indices) == 0:
            return seams

        # Average normals
        n0s = fn[adj[crease_indices, 0]]
        n1s = fn[adj[crease_indices, 1]]
        avg_normals = (n0s + n1s) / 2
        nn = np.linalg.norm(avg_normals, axis=1, keepdims=True)
        avg_normals = avg_normals / np.where(nn == 0, 1, nn)

        midpoints = (p0s + p1s) / 2

        # Dominant axis per edge
        edge_dirs = edge_vecs / np.maximum(edge_lens[:, np.newaxis], 1e-9)
        axis_dots = np.abs(edge_dirs)  # dot with [1,0,0], [0,1,0], [0,0,1]
        dominant_axes = np.argmax(axis_dots, axis=1)
        axis_names = np.array(['x', 'y', 'z'])

        crease_angles = angles[crease_indices]

        for i in range(len(crease_indices)):
            seams.append({
                'v0': p0s[i],
                'v1': p1s[i],
                'midpoint': midpoints[i],
                'normal': avg_normals[i],
                'angle': float(crease_angles[i]),
                'axis': axis_names[dominant_axes[i]],
                'length': float(edge_lens[i]),
            })

    except Exception as e:
        print(f"Natural seam detection error: {e}")

    return seams


def suggest_cut_positions(mesh: trimesh.Trimesh,
                           min_angle: float = 25.0) -> List[dict]:
    """
    Suggest cut plane positions based on natural seam lines.
    Groups nearby parallel crease edges into cut plane suggestions.

    Returns list of { axis, position, score, description }
    """
    seams = find_natural_seams(mesh, min_angle)
    if not seams:
        return []

    suggestions = {}

    for seam in seams:
        ax = seam['axis']
        ax_idx = {'x': 0, 'y': 1, 'z': 2}[ax]
        pos = float(seam['midpoint'][ax_idx])
        score = seam['angle']

        # Round to nearest 5mm to group nearby seams
        bucket = round(pos / 5.0) * 5.0
        key = (ax, bucket)

        if key not in suggestions:
            suggestions[key] = {
                'axis': ax,
                'position': bucket,
                'score': 0.0,
                'count': 0,
                'description': f"Crease line at {ax.upper()}={bucket:.0f}mm"
            }
        suggestions[key]['score'] += score
        suggestions[key]['count'] += 1

    # Sort by score (higher = stronger crease = better suggestion)
    result = sorted(suggestions.values(),
                    key=lambda x: x['score'], reverse=True)
    return result[:20]  # top 20 suggestions


def scores_to_rgba(scores: np.ndarray,
                    alpha: float = 0.7) -> np.ndarray:
    """
    Convert seam scores (0=good, 1=bad) to RGBA vertex colours.
    0.0 = green (good seam location)
    0.5 = yellow (mediocre)
    1.0 = red (bad seam location)
    """
    n = len(scores)
    s = np.clip(scores, 0, 1).astype(np.float32)

    rgba = np.zeros((n, 4), dtype=np.float32)

    # Green → Yellow → Red colour ramp (vectorised)
    low_mask = s < 0.5
    high_mask = ~low_mask

    # Low half: green to yellow (r increases, g=1, b=0)
    t_low = s[low_mask] * 2
    rgba[low_mask, 0] = t_low
    rgba[low_mask, 1] = 1.0
    rgba[low_mask, 2] = 0.0

    # High half: yellow to red (r=1, g decreases, b=0)
    t_high = (s[high_mask] - 0.5) * 2
    rgba[high_mask, 0] = 1.0
    rgba[high_mask, 1] = 1.0 - t_high
    rgba[high_mask, 2] = 0.0

    rgba[:, 3] = alpha

    return rgba
