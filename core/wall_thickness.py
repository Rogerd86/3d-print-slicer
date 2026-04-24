"""
wall_thickness.py
Detect thin wall sections that may fail to print properly.

Casts rays inward from sampled face centres and measures distance
to the opposite wall. Flags sections thinner than the minimum
(typically 1.5mm for FDM = ~2 perimeter lines at 0.4mm nozzle).
"""
import numpy as np
import trimesh
from typing import Dict, List, Optional


def analyse_wall_thickness(mesh: trimesh.Trimesh,
                            min_thickness: float = 1.5,
                            n_samples: int = 1000) -> Dict:
    """
    Check for thin wall sections by ray casting.

    Parameters
    ----------
    mesh : the mesh to check
    min_thickness : minimum acceptable wall thickness in mm
    n_samples : number of face samples to test

    Returns
    -------
    dict with:
      thin_count: number of thin samples found
      total_samples: total samples tested
      pct_thin: percentage of thin sections
      min_found_mm: thinnest wall found
      avg_thickness_mm: average wall thickness
      thin_locations: list of (position, thickness) for thin spots
      suggestion: human-readable recommendation
    """
    try:
        n_faces = len(mesh.faces)
        if n_faces == 0:
            return _empty_result()

        # Sample face centres
        sample_count = min(n_samples, n_faces)
        if sample_count < n_faces:
            indices = np.random.choice(n_faces, sample_count, replace=False)
        else:
            indices = np.arange(n_faces)

        centres = mesh.triangles_center[indices]
        normals = mesh.face_normals[indices]

        # Cast rays inward (opposite to face normal)
        ray_origins = centres + normals * 0.01  # tiny offset to avoid self-hit
        ray_dirs = -normals

        thicknesses = []
        thin_locations = []

        # Use trimesh ray casting
        try:
            locations, index_ray, index_tri = mesh.ray.intersects_location(
                ray_origins, ray_dirs, multiple_hits=False)

            if len(locations) > 0:
                # Compute distances
                for i in range(len(locations)):
                    ray_idx = index_ray[i]
                    origin = ray_origins[ray_idx]
                    hit = locations[i]
                    dist = float(np.linalg.norm(hit - origin))

                    if dist > 0.1:  # ignore self-intersections
                        thicknesses.append(dist)
                        if dist < min_thickness:
                            thin_locations.append((
                                centres[ray_idx].copy(),
                                round(dist, 2)
                            ))
        except Exception as e:
            print(f"Ray cast warning: {e}")
            # Fallback: estimate from bounding box
            dims = mesh.extents
            min_dim = float(np.min(dims))
            thicknesses = [min_dim]

        if not thicknesses:
            return _empty_result()

        arr = np.array(thicknesses)
        thin_count = int(np.sum(arr < min_thickness))
        pct_thin = round(thin_count / len(arr) * 100, 1)
        min_found = round(float(arr.min()), 2)
        avg_found = round(float(arr.mean()), 2)

        if pct_thin > 20:
            suggestion = (f"Warning: {pct_thin}% of walls are thinner than {min_thickness}mm. "
                         f"Thinnest: {min_found}mm. Consider increasing wall count in slicer "
                         f"or repositioning cuts to avoid thin areas.")
        elif pct_thin > 5:
            suggestion = (f"Some thin areas found ({pct_thin}%). Thinnest: {min_found}mm. "
                         f"Print with 3+ walls for strength.")
        elif min_found < min_thickness:
            suggestion = f"Minor thin spot: {min_found}mm. Should print OK with 3 walls."
        else:
            suggestion = f"Wall thickness OK. Minimum: {min_found}mm, average: {avg_found}mm."

        return {
            'thin_count': thin_count,
            'total_samples': len(thicknesses),
            'pct_thin': pct_thin,
            'min_found_mm': min_found,
            'avg_thickness_mm': avg_found,
            'thin_locations': thin_locations[:20],  # cap for display
            'suggestion': suggestion,
        }

    except Exception as e:
        print(f"Wall thickness error: {e}")
        return _empty_result()


def analyse_all_parts(parts, min_thickness: float = 1.5) -> List[Dict]:
    """Run wall thickness analysis on all parts."""
    results = []
    for part in parts:
        r = analyse_wall_thickness(part.mesh, min_thickness)
        r['label'] = part.label
        results.append(r)
    return results


def _empty_result():
    return {
        'thin_count': 0,
        'total_samples': 0,
        'pct_thin': 0.0,
        'min_found_mm': 0.0,
        'avg_thickness_mm': 0.0,
        'thin_locations': [],
        'suggestion': 'No data — mesh may be too simple to analyse.',
    }
