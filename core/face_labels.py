"""
face_labels.py
Emboss matching labels (A1/A2, B1/B2, etc.) on both sides of each cut face.
This helps users know which pieces connect during assembly.

Uses the same 7-segment geometry system as part_numbers.py for digits,
plus simple box-based letter geometry for A-Z.
"""
import numpy as np
import trimesh
from typing import List, Tuple, Optional, Dict


# Letter geometry: each letter is a list of (cx, cy, w, h) rectangles
# Coordinates are relative to a 5×8 unit cell, centred at (2.5, 0)
_LETTER_RECTS = {
    'A': [(2.5,3.5,5,1), (0.5,0,1,7), (4.5,0,1,7), (2.5,0,3,1)],
    'B': [(0.5,0,1,8), (2.5,7.5,4,1), (2.5,3.5,4,1), (2.5,-0.5,4,1), (4.5,5.5,1,3), (4.5,1.5,1,3)],
    'C': [(0.5,0,1,8), (2.5,7.5,5,1), (2.5,-0.5,5,1)],
    'D': [(0.5,0,1,8), (2.5,7.5,3,1), (2.5,-0.5,3,1), (4.5,1,1,6)],
    'E': [(0.5,0,1,8), (2.5,7.5,5,1), (2.5,3.5,4,1), (2.5,-0.5,5,1)],
    'F': [(0.5,0,1,8), (2.5,7.5,5,1), (2.5,3.5,4,1)],
}

# 7-segment digits (reused from part_numbers)
_SEGMENTS = {
    0:[1,1,1,0,1,1,1], 1:[0,0,1,0,0,1,0], 2:[1,0,1,1,1,0,1],
    3:[1,0,1,1,0,1,1], 4:[0,1,1,1,0,1,0], 5:[1,1,0,1,0,1,1],
    6:[1,1,0,1,1,1,1], 7:[1,0,1,0,0,1,0], 8:[1,1,1,1,1,1,1],
    9:[1,1,1,1,0,1,1],
}


def _make_letter(letter: str, x_off: float, height: float = 8.0,
                  emboss: float = 0.6) -> List[trimesh.Trimesh]:
    """Build a letter from box geometry."""
    rects = _LETTER_RECTS.get(letter.upper(), _LETTER_RECTS.get('A'))
    if rects is None:
        return []
    scale = height / 8.0
    parts = []
    for cx, cy, w, h in rects:
        box = trimesh.creation.box([w * scale, h * scale, emboss])
        box.apply_translation([x_off + cx * scale, cy * scale, emboss / 2])
        parts.append(box)
    return parts


def _make_digit_for_label(digit: int, x_off: float, height: float = 8.0,
                           emboss: float = 0.6) -> List[trimesh.Trimesh]:
    """Build one 7-segment digit."""
    seg = _SEGMENTS.get(digit, [0]*7)
    dh = height
    dw = height * 0.65
    t = dh * 0.13
    h2 = dh / 2
    parts = []

    def bar(cx, cy, bw, bh):
        b = trimesh.creation.box([bw, bh, emboss])
        b.apply_translation([x_off + cx, cy, emboss / 2])
        return b

    if seg[0]: parts.append(bar(dw/2,  h2-t/2,     dw,   t))
    if seg[1]: parts.append(bar(t/2,   h2/2,        t,    h2-t))
    if seg[2]: parts.append(bar(dw-t/2,h2/2,        t,    h2-t))
    if seg[3]: parts.append(bar(dw/2,  0,           dw-2*t, t))
    if seg[4]: parts.append(bar(t/2,  -h2/2,        t,    h2-t))
    if seg[5]: parts.append(bar(dw-t/2,-h2/2,       t,    h2-t))
    if seg[6]: parts.append(bar(dw/2, -h2+t/2,      dw,   t))
    return parts


def make_label_geometry(label: str, height: float = 6.0,
                         emboss: float = 0.5) -> Optional[trimesh.Trimesh]:
    """
    Build embossed label geometry (e.g. "A1", "B2").
    Returns mesh in XY plane, embossed along +Z, centred at origin.
    """
    all_parts = []
    x = 0.0
    char_w = height * 0.8
    for ch in label.upper():
        if ch.isalpha():
            all_parts.extend(_make_letter(ch, x, height, emboss))
            x += char_w
        elif ch.isdigit():
            all_parts.extend(_make_digit_for_label(int(ch), x, height, emboss))
            x += char_w * 0.7
        else:
            x += char_w * 0.3  # space

    if not all_parts:
        return None
    combined = trimesh.util.concatenate(all_parts)
    c = combined.centroid
    combined.apply_translation([-c[0], -c[1], 0])
    return combined


def generate_face_labels(parts) -> Dict[str, List[Tuple]]:
    """
    Determine matching labels for all cut faces across all parts.

    Returns dict: { part_id: [(face_centre, face_normal, label_text), ...] }

    Naming convention:
      - Each shared cut face pair gets a letter (A, B, C, ...)
      - Side 1 gets "A1", side 2 gets "A2"
    """
    # Find cut faces on all parts and match pairs
    all_faces = []  # (part, face_centre, face_normal, face_area)
    for part in parts:
        try:
            if not hasattr(part.mesh, 'facets') or len(part.mesh.facets) == 0:
                continue
            for i, facet in enumerate(part.mesh.facets):
                area = part.mesh.facets_area[i]
                if area < part.mesh.area * 0.02:
                    continue
                normal = part.mesh.facets_normal[i]
                if np.max(np.abs(normal)) < 0.9:
                    continue
                verts = part.mesh.vertices[part.mesh.faces[facet].flatten()]
                centre = verts.mean(axis=0)
                all_faces.append((part, centre, normal, area))
        except Exception:
            pass

    # Match pairs: faces at same position with opposite normals
    matched = set()
    pairs = []
    for i, (p1, c1, n1, a1) in enumerate(all_faces):
        if i in matched:
            continue
        for j, (p2, c2, n2, a2) in enumerate(all_faces):
            if j <= i or j in matched:
                continue
            if p1.id == p2.id:
                continue
            # Same position (within tolerance), opposite normals
            dist = np.linalg.norm(c1 - c2)
            dot = np.dot(n1, n2)
            if dist < 10.0 and dot < -0.8:
                pairs.append((i, j))
                matched.add(i)
                matched.add(j)
                break

    # Assign labels
    labels = {}  # part_id -> [(centre, normal, label)]
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for pi, (i, j) in enumerate(pairs):
        letter = letters[pi % len(letters)]
        p1, c1, n1, _ = all_faces[i]
        p2, c2, n2, _ = all_faces[j]

        if p1.id not in labels:
            labels[p1.id] = []
        labels[p1.id].append((c1, n1, f"{letter}1"))

        if p2.id not in labels:
            labels[p2.id] = []
        labels[p2.id].append((c2, n2, f"{letter}2"))

    return labels


def add_labels_to_mesh(mesh: trimesh.Trimesh,
                        face_labels: List[Tuple],
                        label_height: float = 6.0,
                        emboss_depth: float = 0.5,
                        inset: float = 2.0) -> trimesh.Trimesh:
    """
    Emboss face labels onto a mesh's cut faces.

    face_labels: list of (centre, normal, label_text) from generate_face_labels()
    """
    result = mesh
    for centre, normal, text in face_labels:
        try:
            label_mesh = make_label_geometry(text, label_height, emboss_depth)
            if label_mesh is None:
                continue

            # Orient label to lie on the cut face
            n = np.array(normal, dtype=float)
            n /= max(np.linalg.norm(n), 1e-9)
            rot = trimesh.geometry.align_vectors([0, 0, 1], (-n).tolist())
            label_mesh.apply_transform(rot)

            # Position: face centre, inset from surface
            pos = np.array(centre) - n * inset
            label_mesh.apply_translation(pos)

            result = trimesh.util.concatenate([result, label_mesh])
        except Exception as e:
            print(f"Label emboss error ({text}): {e}")

    return result
