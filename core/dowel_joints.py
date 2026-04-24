"""
dowel_joints.py
Creates dowel recesses in cut faces. Supports:
  - Round holes (for steel rod stock)
  - Rectangular slots (for flat bar / key stock)

Each DowelConfig is per-joint and fully customisable.
Both sides of a cut get matching recesses so the rod/bar slides through.
"""
import os
import numpy as np
import trimesh
from typing import List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class DowelConfig:
    """Full configuration for dowels on one cut face."""
    shape: str = 'round'        # 'round' or 'rect'
    count: int = 2              # number of dowels across the face
    spacing: float = 0.0        # mm between dowel centres (0 = auto-distribute)
    depth_a: float = 30.0       # mm recess into part A
    depth_b: float = 30.0       # mm recess into part B
    tolerance: float = 0.3      # mm extra clearance (holes slightly bigger than rod)

    # Round specific
    radius: float = 5.0         # mm radius of steel rod

    # Rectangular specific
    rect_width: float = 3.0     # mm (narrow dimension — fits into cut)
    rect_height: float = 12.0   # mm (tall dimension)

    # Offset from face centre (let user shift dowels away from edges)
    offset_u: float = 0.0       # mm offset along face width axis
    offset_v: float = 0.0       # mm offset along face height axis


def apply_dowels_to_pair(mesh_a: trimesh.Trimesh,
                          mesh_b: trimesh.Trimesh,
                          cut_normal: np.ndarray,
                          cut_origin: np.ndarray,
                          config: DowelConfig) -> Tuple[trimesh.Trimesh, trimesh.Trimesh]:
    """
    Cut matching dowel recesses into mesh_a and mesh_b at their shared cut face.
    mesh_a is on the negative-normal side, mesh_b on the positive-normal side.
    Returns (mesh_a_recessed, mesh_b_recessed).
    """
    try:
        positions = _distribute_dowel_positions(
            mesh_a, cut_normal, cut_origin, config)

        if not positions:
            print("No dowel positions found — face too small or missed.")
            return mesh_a, mesh_b

        # Cut into part A (going in the +normal direction)
        cutters_a = _make_cutters(positions, cut_normal, config, depth=config.depth_a, into_a=True)
        # Cut into part B (going in the -normal direction)
        cutters_b = _make_cutters(positions, cut_normal, config, depth=config.depth_b, into_a=False)

        result_a = _boolean_subtract(mesh_a, cutters_a)
        result_b = _boolean_subtract(mesh_b, cutters_b)

        return result_a, result_b

    except Exception as e:
        print(f"Dowel application error: {e}")
        return mesh_a, mesh_b


def _distribute_dowel_positions(mesh: trimesh.Trimesh,
                                  normal: np.ndarray,
                                  origin: np.ndarray,
                                  config: DowelConfig) -> List[np.ndarray]:
    """
    Work out where to place dowel centres on the cut face.

    Uses the origin directly as the face centre (provided by _find_cut_faces
    which computes it from actual facet vertex positions). This avoids the
    unreliable triangle-filtering approach.
    """
    u, v = _face_axes(normal)
    n_vec = np.array(normal, dtype=float)
    n_vec /= max(np.linalg.norm(n_vec), 1e-9)

    # Use origin as the face centre — it comes from _find_cut_faces which
    # computes verts.mean(axis=0) of the actual facet vertices.
    base = origin.copy()

    # Find the actual cut face triangles near the origin to get true face dimensions
    face_centres = mesh.triangles_center
    # Distance from each triangle centre to the cut plane (through origin, with normal)
    plane_dist = np.abs(np.dot(face_centres - origin, n_vec))
    # Also check that triangles are near the origin in the face-plane direction
    near_mask = plane_dist < 5.0  # within 5mm of the cut plane

    if np.any(near_mask):
        # Use actual face triangle centres for span calculation
        pts = face_centres[near_mask]
        u_proj = np.dot(pts - origin, u)  # project relative to origin
        v_proj = np.dot(pts - origin, v)
        face_u_span = u_proj.max() - u_proj.min() if len(u_proj) > 1 else 20.0
        face_v_span = v_proj.max() - v_proj.min() if len(v_proj) > 1 else 20.0
    else:
        # Fallback: use part dimensions perpendicular to cut normal
        dims = mesh.bounds[1] - mesh.bounds[0]
        # Remove the normal-axis dimension
        ax_idx = int(np.argmax(np.abs(n_vec)))
        face_dims = np.delete(dims, ax_idx)
        face_u_span = float(face_dims[0])
        face_v_span = float(face_dims[1]) if len(face_dims) > 1 else face_u_span

    # Use the larger face span for distributing connectors across the face
    span = max(face_u_span, face_v_span)
    if span < 5.0:
        span = 20.0  # minimum reasonable span

    # Apply user offsets
    base = base + config.offset_u * u + config.offset_v * v

    n = config.count
    if n == 1:
        return [base]

    # Space evenly across the face, leaving 20% margin from edges
    if config.spacing > 0:
        step = config.spacing
    else:
        margin = span * 0.20
        usable = span - 2 * margin
        step = usable / (n - 1) if n > 1 else 0

    half = step * (n - 1) / 2
    positions = []
    for i in range(n):
        offset = -half + i * step
        positions.append(base + offset * u)

    return positions


def _face_axes(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return two perpendicular axes lying in the cut plane."""
    n = normal / np.linalg.norm(normal)
    helper = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, helper); u /= np.linalg.norm(u)
    v = np.cross(n, u);     v /= np.linalg.norm(v)
    return u, v


def _make_cutters(positions: List[np.ndarray],
                   normal: np.ndarray,
                   config: DowelConfig,
                   depth: float,
                   into_a: bool) -> List[trimesh.Trimesh]:
    """Build cutter geometry for one side of the joint."""
    direction = -normal if into_a else normal
    tol = config.tolerance
    cutters = []

    for pos in positions:
        try:
            if config.shape == 'round':
                r = config.radius + tol
                cutter = trimesh.creation.cylinder(radius=r, height=depth + 2, sections=24)
                rot = trimesh.geometry.align_vectors([0, 0, 1], direction.tolist())
                cutter.apply_transform(rot)
                # Centre the cutter: half goes through face, half goes into part
                cutter.apply_translation(pos + direction * (depth / 2))

            else:  # rect
                u, v = _face_axes(normal)
                w = config.rect_width + tol
                h = config.rect_height + tol
                d = depth + 2
                cutter = trimesh.creation.box([d, w, h])
                # Rotate so the long axis aligns with the cut direction
                rot = trimesh.geometry.align_vectors([1, 0, 0], direction.tolist())
                cutter.apply_transform(rot)
                cutter.apply_translation(pos + direction * (depth / 2))

            cutters.append(cutter)
        except Exception as e:
            print(f"Cutter build error: {e}")

    return cutters


def _boolean_subtract(mesh: trimesh.Trimesh,
                        cutters: List[trimesh.Trimesh],
                        fallback_path: Optional[str] = None,
                        part_label: str = "") -> trimesh.Trimesh:
    """
    Subtract all cutters from mesh using available boolean engine.
    If booleans fail and fallback_path is provided, exports cutters as separate STL.
    """
    if not cutters:
        return mesh
    try:
        from core.boolean_ops import boolean_difference
        all_cutters = trimesh.util.concatenate(cutters) if len(cutters) > 1 else cutters[0]
        result = boolean_difference([mesh, all_cutters])
        if result is not None and len(result.faces) > 0:
            return result
    except Exception as e:
        print(f"Boolean subtract error: {e}")
        # Fallback: export cutters as a separate file for manual subtraction
        if fallback_path:
            try:
                all_cutters = trimesh.util.concatenate(cutters) if len(cutters) > 1 else cutters[0]
                cutter_file = os.path.join(
                    fallback_path, f"{part_label}_dowel_cutters.stl")
                all_cutters.export(cutter_file, file_type="stl")
                print(f"Boolean fallback: exported cutters to {cutter_file}")
            except Exception as fe:
                print(f"Fallback export error: {fe}")
    return mesh


def generate_round_rod_guide(radius: float, length: float) -> trimesh.Trimesh:
    """Generate a separate printable round rod guide / sleeve."""
    return trimesh.creation.cylinder(radius=radius * 0.96, height=length, sections=24)


def generate_rect_key(width: float, height: float, length: float) -> trimesh.Trimesh:
    """Generate a separate printable rectangular key."""
    return trimesh.creation.box([length, width * 0.96, height * 0.96])
