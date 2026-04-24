"""
mesh_smoother.py
Smooth cut parts while preserving flat cut faces so parts still fit together.

Three algorithms available:
  taubin     — Best overall: low volume change, good noise removal. Recommended.
  laplacian  — Fastest, slight shrinkage on curved surfaces.
  humphrey   — Good feature preservation, slight volume change.

Key feature: cut face preservation
  Vertices on flat axis-aligned faces (created by slicing) are locked in place.
  Only the organic surface vertices get smoothed.
  This means smoothed parts still align correctly during assembly.
"""
import numpy as np
import trimesh
from trimesh import smoothing as tm_smooth
from typing import Tuple, Optional
import copy


METHODS = {
    'taubin':    'Taubin (recommended — best volume preservation)',
    'laplacian': 'Laplacian (fastest)',
    'humphrey':  'Humphrey (good feature preservation)',
}


def smooth_part(mesh: trimesh.Trimesh,
                method: str = 'taubin',
                iterations: int = 10,
                strength: float = 0.5,
                preserve_cut_faces: bool = True,
                cut_face_angle_threshold: float = 15.0) -> Tuple[trimesh.Trimesh, dict]:
    """
    Smooth a mesh, optionally preserving flat cut faces.

    Parameters
    ----------
    mesh : input mesh
    method : 'taubin', 'laplacian', or 'humphrey'
    iterations : smoothing passes (5-30 typical range)
    strength : how aggressively to smooth (0.1 = gentle, 0.9 = heavy)
    preserve_cut_faces : lock vertices on flat axis-aligned faces
    cut_face_angle_threshold : faces within this many degrees of axis-aligned
                               are considered cut faces and locked

    Returns
    -------
    (smoothed_mesh, stats_dict)
    stats_dict has: locked_verts, smoothed_verts, volume_change_pct, watertight
    """
    m = copy.deepcopy(mesh)
    orig_volume = float(mesh.volume) if mesh.is_watertight else 0.0
    locked_verts = set()

    if preserve_cut_faces:
        locked_verts = _find_cut_face_vertices(m, angle_threshold=cut_face_angle_threshold)

    # Save original positions of locked vertices
    orig_positions = m.vertices.copy()

    # Apply chosen smoothing algorithm
    try:
        if method == 'taubin':
            # nu slightly larger than lambda to prevent shrinkage
            lam = strength * 0.5
            nu  = min(lam + 0.03, 0.99)
            tm_smooth.filter_taubin(m, lamb=lam, nu=nu, iterations=iterations)

        elif method == 'laplacian':
            tm_smooth.filter_laplacian(m, lamb=strength, iterations=iterations)

        elif method == 'humphrey':
            alpha = max(0.01, 0.2 - strength * 0.15)
            beta  = min(0.95, 0.3 + strength * 0.5)
            tm_smooth.filter_humphrey(m, alpha=alpha, beta=beta, iterations=iterations)

        else:
            raise ValueError(f"Unknown method: {method}")

    except Exception as e:
        print(f"Smoothing error ({method}): {e}")
        return mesh, {'error': str(e)}

    # Restore locked vertices (cut faces stay flat)
    if locked_verts:
        for vi in locked_verts:
            if vi < len(m.vertices):
                m.vertices[vi] = orig_positions[vi]

    new_volume = float(m.volume) if m.is_watertight else 0.0
    vol_change = (100.0 * (new_volume - orig_volume) / orig_volume
                  if orig_volume > 0 else 0.0)

    stats = {
        'locked_verts': len(locked_verts),
        'smoothed_verts': len(m.vertices) - len(locked_verts),
        'total_verts': len(m.vertices),
        'volume_change_pct': round(vol_change, 2),
        'watertight': m.is_watertight,
        'method': method,
        'iterations': iterations,
    }

    return m, stats


def smooth_all_parts(meshes: list,
                      method: str = 'taubin',
                      iterations: int = 10,
                      strength: float = 0.5,
                      preserve_cut_faces: bool = True,
                      progress_cb=None) -> list:
    """
    Smooth a list of meshes in parallel.
    Returns list of (smoothed_mesh, stats) tuples.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    results = [None] * len(meshes)
    done = 0
    lock = threading.Lock()

    def do_one(args):
        idx, mesh = args
        return idx, smooth_part(mesh, method, iterations, strength, preserve_cut_faces)

    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(do_one, (i, m)): i for i, m in enumerate(meshes)}
        for fut in futs:
            idx, result = fut.result()
            results[idx] = result
            with lock:
                done += 1
                if progress_cb:
                    progress_cb(done, len(meshes))

    return results


def _find_cut_face_vertices(mesh: trimesh.Trimesh,
                              angle_threshold: float = 15.0) -> set:
    """
    Find vertex indices belonging to flat cut faces.
    Cut faces are large, flat, roughly axis-aligned planar regions.
    """
    locked = set()

    try:
        if not hasattr(mesh, 'facets') or len(mesh.facets) == 0:
            return locked

        total_area = max(mesh.area, 1e-9)
        min_facet_area = total_area * 0.02  # only consider facets > 2% of total area
        thresh = np.cos(np.radians(angle_threshold))

        for i, facet in enumerate(mesh.facets):
            if mesh.facets_area[i] < min_facet_area:
                continue

            normal = mesh.facets_normal[i]
            abs_n  = np.abs(normal)

            # Check if this facet is roughly axis-aligned (i.e. a cut face)
            if np.max(abs_n) < thresh:
                continue  # too tilted — not a cut face

            # Lock all vertices of this facet's faces
            for fi in facet:
                for vi in mesh.faces[fi]:
                    locked.add(int(vi))

    except Exception as e:
        print(f"Cut face detection error: {e}")

    return locked


def estimate_smoothing_impact(mesh: trimesh.Trimesh,
                               method: str = 'taubin',
                               iterations: int = 10,
                               strength: float = 0.5) -> dict:
    """
    Quick estimate of how much smoothing will change the mesh.
    Runs on a simplified copy so it's fast.
    """
    try:
        # Simplify for speed
        simple = mesh.simplify_quadric_decimation(
            max(50, len(mesh.faces) // 10))
        _, stats = smooth_part(simple, method, iterations, strength,
                                preserve_cut_faces=False)
        return stats
    except Exception:
        return {'volume_change_pct': 0, 'watertight': mesh.is_watertight}
