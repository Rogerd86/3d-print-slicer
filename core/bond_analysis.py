"""
bond_analysis.py
Analyse cut face bonding surface area and structural integrity.

No competing tool does this — it's a unique differentiator.
Helps users decide if cut faces provide enough bonding area
for structural integrity, and flags thin/narrow joints.
"""
import numpy as np
import trimesh
from typing import Dict, List, Optional


def analyse_bond_surfaces(mesh: trimesh.Trimesh) -> List[Dict]:
    """
    Find all flat cut faces on a mesh and analyse their bonding properties.

    Returns list of dicts, one per detected cut face:
      { 'normal', 'centre', 'area_mm2', 'axis',
        'aspect_ratio', 'is_narrow', 'bond_quality', 'suggestion' }
    """
    faces = []
    try:
        if not hasattr(mesh, 'facets') or len(mesh.facets) == 0:
            return faces

        total_area = max(mesh.area, 1e-9)
        min_area = total_area * 0.02  # ignore tiny facets

        for i, facet in enumerate(mesh.facets):
            area = mesh.facets_area[i]
            if area < min_area:
                continue

            normal = mesh.facets_normal[i]
            abs_n = np.abs(normal)
            if np.max(abs_n) < 0.9:
                continue  # not axis-aligned = not a cut face

            # Get bounding box of this facet's vertices
            vert_indices = mesh.faces[facet].flatten()
            verts = mesh.vertices[vert_indices]
            dims = verts.max(axis=0) - verts.min(axis=0)
            # Remove the axis-aligned dimension (it's ~0 for a flat face)
            ax = int(np.argmax(abs_n))
            face_dims = np.delete(dims, ax)
            w, h = float(max(face_dims)), float(min(face_dims))
            aspect = w / max(h, 0.1)

            is_narrow = h < 8.0 or aspect > 6.0
            centre = verts.mean(axis=0)

            # Bond quality score (0-100)
            # Based on area, aspect ratio, and minimum dimension
            area_score = min(50, float(area) / 200.0 * 50)  # up to 50 pts for area
            shape_score = max(0, 30 - (aspect - 1) * 5)  # up to 30 pts for squareness
            min_dim_score = min(20, h / 10.0 * 20)  # up to 20 pts for minimum width
            quality = min(100, area_score + shape_score + min_dim_score)

            if quality >= 70:
                suggestion = "Strong bond — good area and proportions"
            elif quality >= 40:
                suggestion = "Adequate bond — use strong adhesive"
            elif quality >= 20:
                suggestion = "Weak bond — add dowels or mechanical fasteners"
            else:
                suggestion = "Very weak bond — reposition cut to get more surface area"

            axis_name = ['X', 'Y', 'Z'][ax]
            faces.append({
                'normal': normal.copy(),
                'centre': centre.copy(),
                'area_mm2': round(float(area), 1),
                'axis': axis_name,
                'width_mm': round(w, 1),
                'height_mm': round(h, 1),
                'aspect_ratio': round(aspect, 1),
                'is_narrow': is_narrow,
                'bond_quality': round(quality),
                'suggestion': suggestion,
            })

    except Exception as e:
        print(f"Bond analysis error: {e}")

    # Sort by quality (worst first — user should fix these)
    faces.sort(key=lambda f: f['bond_quality'])
    return faces


def analyse_all_parts(parts) -> List[Dict]:
    """Analyse bond surfaces for all parts."""
    results = []
    for part in parts:
        faces = analyse_bond_surfaces(part.mesh)
        if faces:
            worst = faces[0]  # sorted worst-first
            results.append({
                'label': part.label,
                'n_cut_faces': len(faces),
                'min_quality': worst['bond_quality'],
                'worst_face': worst,
                'all_faces': faces,
            })
        else:
            results.append({
                'label': part.label,
                'n_cut_faces': 0,
                'min_quality': 100,
                'worst_face': None,
                'all_faces': [],
            })
    return results


def bond_summary(analyses: List[Dict]) -> str:
    """Human-readable summary."""
    weak = [a for a in analyses if a['min_quality'] < 40]
    ok = [a for a in analyses if 40 <= a['min_quality'] < 70]
    strong = [a for a in analyses if a['min_quality'] >= 70]

    lines = []
    if weak:
        labels = ", ".join(a['label'] for a in weak[:3])
        extra = f" +{len(weak)-3} more" if len(weak) > 3 else ""
        lines.append(f"⚠ {len(weak)} parts with weak bonds: {labels}{extra}")
    if ok:
        lines.append(f"~ {len(ok)} parts with adequate bonds")
    if strong:
        lines.append(f"✓ {len(strong)} parts with strong bonds")
    return "\n".join(lines) if lines else "No cut faces detected."
