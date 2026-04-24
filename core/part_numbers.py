"""
part_numbers.py
Emboss part numbers onto the inner face of printed parts.
Uses 7-segment display geometry built entirely from trimesh boxes — no font library needed.
"""
import numpy as np
import trimesh
from typing import Optional

# 7 segments: [top, top-left, top-right, middle, bot-left, bot-right, bottom]
SEGMENTS = {
    0:[1,1,1,0,1,1,1], 1:[0,0,1,0,0,1,0], 2:[1,0,1,1,1,0,1],
    3:[1,0,1,1,0,1,1], 4:[0,1,1,1,0,1,0], 5:[1,1,0,1,0,1,1],
    6:[1,1,0,1,1,1,1], 7:[1,0,1,0,0,1,0], 8:[1,1,1,1,1,1,1],
    9:[1,1,1,1,0,1,1],
}


def _make_digit(digit: int, x_off: float,
                dh: float = 8.0, dw: float = 5.0, emboss: float = 0.6) -> list:
    """Build one digit from box segments. Returns list of trimesh meshes."""
    seg = SEGMENTS.get(digit, [0]*7)
    t = dh * 0.13   # bar thickness
    h2 = dh / 2
    parts = []

    def bar(cx, cy, bw, bh):
        b = trimesh.creation.box([bw, bh, emboss])
        b.apply_translation([x_off + cx, cy, emboss / 2])
        return b

    if seg[0]: parts.append(bar(dw/2,  h2-t/2,     dw,   t))    # top
    if seg[1]: parts.append(bar(t/2,   h2/2,        t,    h2-t)) # top-left
    if seg[2]: parts.append(bar(dw-t/2,h2/2,        t,    h2-t)) # top-right
    if seg[3]: parts.append(bar(dw/2,  0,           dw-2*t, t))  # middle
    if seg[4]: parts.append(bar(t/2,  -h2/2,        t,    h2-t)) # bot-left
    if seg[5]: parts.append(bar(dw-t/2,-h2/2,       t,    h2-t)) # bot-right
    if seg[6]: parts.append(bar(dw/2, -h2+t/2,      dw,   t))    # bottom
    return parts


def make_number_geometry(number: int,
                          digit_height: float = 8.0,
                          emboss_depth: float = 0.6) -> Optional[trimesh.Trimesh]:
    """
    Build embossed number geometry for the given integer.
    Returns a flat mesh (in XY plane, embossed along +Z) ready to be
    placed on a part face.
    """
    s = str(abs(number))
    dw = digit_height * 0.65   # digit width
    spacing = digit_height * 0.2
    all_segs = []
    for i, ch in enumerate(s):
        if ch.isdigit():
            segs = _make_digit(int(ch),
                               i * (dw + spacing),
                               dh=digit_height,
                               dw=dw,
                               emboss=emboss_depth)
            all_segs.extend(segs)
    if not all_segs:
        return None
    combined = trimesh.util.concatenate(all_segs)
    # Centre the number at origin
    c = combined.centroid
    combined.apply_translation([-c[0], -c[1], 0])
    return combined


def add_part_number_to_mesh(mesh: trimesh.Trimesh,
                              number: int,
                              emboss_depth: float = 0.6,
                              digit_height: float = 8.0,
                              inset_from_face: float = 2.0) -> trimesh.Trimesh:
    """
    Add an embossed part number to the inner face of a mesh.
    Finds the largest flat face and places the number there.
    The number protrudes inward (into the part interior).

    Returns mesh with number attached, or original mesh on failure.
    """
    try:
        num_geo = make_number_geometry(number, digit_height, emboss_depth)
        if num_geo is None:
            return mesh

        # Find the largest flat face on the mesh to place the number
        face, normal, origin = _find_best_face(mesh)

        if face is None:
            # Fallback: use -Z face
            bounds = mesh.bounds
            origin = np.array([
                (bounds[0][0] + bounds[1][0]) / 2,
                (bounds[0][1] + bounds[1][1]) / 2,
                bounds[0][2]
            ])
            normal = np.array([0, 0, -1.0])

        # Build a rotation that takes XY plane → face plane
        # Normal of our number geo is +Z, we want it to point opposite to face normal
        # (so it protrudes inward)
        target_n = -normal
        rot = trimesh.geometry.align_vectors([0, 0, 1], target_n.tolist())
        num_geo.apply_transform(rot)

        # Position on face, offset inward slightly
        place_pt = origin + normal * inset_from_face
        num_geo.apply_translation(place_pt)

        # Combine
        combined = trimesh.util.concatenate([mesh, num_geo])
        return combined

    except Exception as e:
        print(f"Part number error for {number}: {e}")
        return mesh


def _find_best_face(mesh: trimesh.Trimesh):
    """Find the largest flat face on the mesh."""
    try:
        if not hasattr(mesh, 'facets') or len(mesh.facets) == 0:
            return None, None, None

        best_idx = np.argmax(mesh.facets_area)
        facet = mesh.facets[best_idx]
        normal = mesh.facets_normal[best_idx]
        verts = mesh.vertices[mesh.faces[facet].flatten()]
        origin = verts.mean(axis=0)
        return facet, normal, origin

    except Exception:
        return None, None, None
