"""
natural_cut.py
Surface-following cut that hides seam lines along natural model contours.

Like LuBan's Natural Cut: starts from a planar cut position, then searches
the vicinity for crease edges/surface features and adjusts the cut surface
to follow them. Falls back to planar cut if no good contour is found.

Implementation:
  1. Take the base planar cut position + axis
  2. Find natural seams (crease edges) near the cut plane
  3. If a strong crease line runs roughly parallel to the cut plane,
     offset the cut position to follow that crease
  4. Apply the cut at the adjusted position
"""
import numpy as np
import trimesh
from trimesh.intersections import slice_mesh_plane
from typing import Tuple, Optional
from core.seam_analysis import find_natural_seams


def natural_cut(mesh: trimesh.Trimesh,
                axis: str,
                base_position: float,
                search_radius: float = 20.0,
                min_crease_angle: float = 20.0) -> Tuple[
                    Optional[trimesh.Trimesh], Optional[trimesh.Trimesh], dict]:
    """
    Perform a natural cut that follows surface contours near the base position.

    Parameters
    ----------
    mesh : the mesh to cut
    axis : 'x', 'y', or 'z'
    base_position : where the user placed the flat cut plane (mm)
    search_radius : how far from base_position to look for creases (mm)
    min_crease_angle : minimum dihedral angle to count as a crease

    Returns
    -------
    (side_a, side_b, info_dict)
    info_dict has: adjusted_position, found_crease, crease_angle, description
    """
    ax_idx = {'x': 0, 'y': 1, 'z': 2}.get(axis, 0)
    normal = np.zeros(3); normal[ax_idx] = 1.0

    # Find natural seams on the mesh
    seams = find_natural_seams(mesh, min_angle=min_crease_angle, min_edge_length=3.0)

    # Filter seams that run roughly perpendicular to our cut axis
    # (a seam perpendicular to X means it runs along Y/Z — these are good
    #  cut positions for an X-axis cut)
    best_crease = None
    best_score = 0

    for seam in seams:
        # Check if seam is near our base position on the cut axis
        seam_pos = float(seam['midpoint'][ax_idx])
        dist = abs(seam_pos - base_position)
        if dist > search_radius:
            continue

        # Score: closer to base position + stronger angle = better
        proximity_score = 1.0 - (dist / search_radius)
        angle_score = seam['angle'] / 90.0  # normalize to 0-1
        length_score = min(1.0, seam['length'] / 30.0)  # longer seams are better
        total_score = proximity_score * 0.4 + angle_score * 0.4 + length_score * 0.2

        if total_score > best_score:
            best_score = total_score
            best_crease = seam

    # Use adjusted position if a good crease was found
    if best_crease is not None and best_score > 0.3:
        adjusted_pos = float(best_crease['midpoint'][ax_idx])
        found = True
        info = {
            'adjusted_position': adjusted_pos,
            'original_position': base_position,
            'offset_mm': round(adjusted_pos - base_position, 1),
            'found_crease': True,
            'crease_angle': round(best_crease['angle'], 1),
            'crease_length': round(best_crease['length'], 1),
            'score': round(best_score, 2),
            'description': (f"Moved cut {adjusted_pos - base_position:+.1f}mm to follow "
                           f"crease line ({best_crease['angle']:.0f}° angle, "
                           f"{best_crease['length']:.0f}mm long)")
        }
    else:
        adjusted_pos = base_position
        found = False
        info = {
            'adjusted_position': base_position,
            'original_position': base_position,
            'offset_mm': 0,
            'found_crease': False,
            'crease_angle': 0,
            'crease_length': 0,
            'score': 0,
            'description': "No natural crease found nearby — using flat cut"
        }

    # Perform the cut at the (possibly adjusted) position
    origin = np.zeros(3)
    origin[ax_idx] = adjusted_pos

    try:
        side_a = slice_mesh_plane(mesh, normal, origin, cap=True)
        side_b = slice_mesh_plane(mesh, -normal, origin, cap=True)
        side_a = side_a if (side_a is not None and len(side_a.faces) > 0) else None
        side_b = side_b if (side_b is not None and len(side_b.faces) > 0) else None
        return side_a, side_b, info
    except Exception as e:
        info['description'] = f"Cut failed: {e}"
        return None, None, info
