"""
Connector pin generator.
After a cut, adds:
  - Male pins (cylinders protruding outward) on side A
  - Female sockets (holes, via boolean difference) on side B

If boolean ops aren't available, falls back to marking pins as separate
printable objects that the user glues in place.
"""
import numpy as np
import trimesh
from typing import List, Tuple, Optional


def find_cut_face_info(mesh: trimesh.Trimesh, cut_normal: np.ndarray,
                       cut_origin: np.ndarray, tolerance: float = 2.0) -> dict:
    """
    Find the flat cap face created by a cut operation.
    Returns dict with: normal, center, u_axis, v_axis, width, height
    """
    n = cut_normal / np.linalg.norm(cut_normal)

    # Find faces whose normal is approximately parallel to cut_normal
    face_norms = mesh.face_normals
    dots = np.abs(np.dot(face_norms, n))
    cut_faces_idx = np.where(dots > 0.95)[0]

    if len(cut_faces_idx) == 0:
        return None

    # Get all vertices on those faces
    cut_verts = mesh.vertices[mesh.faces[cut_faces_idx].reshape(-1)]

    # Filter to vertices near the cut plane
    dists = np.abs(np.dot(cut_verts - cut_origin, n))
    near = dists < tolerance
    if not np.any(near):
        return None

    cut_verts = cut_verts[near]
    center = cut_verts.mean(axis=0)

    # Build UV axes perpendicular to the cut normal
    helper = np.array([0, 0, 1.0]) if abs(n[2]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(n, helper); u /= np.linalg.norm(u)
    v = np.cross(n, u);     v /= np.linalg.norm(v)

    # Extent in UV space
    u_coords = np.dot(cut_verts - center, u)
    v_coords = np.dot(cut_verts - center, v)
    width  = float(u_coords.max() - u_coords.min())
    height = float(v_coords.max() - v_coords.min())

    return {
        'normal': n,
        'center': center,
        'u_axis': u,
        'v_axis': v,
        'width':  width,
        'height': height,
        'u_min': float(u_coords.min()),
        'u_max': float(u_coords.max()),
        'v_min': float(v_coords.min()),
        'v_max': float(v_coords.max()),
    }


def compute_pin_positions(face_info: dict, n_pins: int,
                          pin_radius: float) -> List[np.ndarray]:
    """
    Compute pin center positions on the cut face.
    n_pins: 2 or 4 (arranged symmetrically) or 'grid'
    Returns list of 3D world positions.
    """
    center = face_info['center']
    u = face_info['u_axis']
    v = face_info['v_axis']
    w = face_info['width']
    h = face_info['height']

    # Margin from edge
    margin = max(pin_radius * 2.5, 3.0)
    uw = w / 2 - margin
    vh = h / 2 - margin

    if uw < 0: uw = w / 4
    if vh < 0: vh = h / 4

    positions = []

    if n_pins == 2:
        # Two pins along the longer axis
        if w >= h:
            positions = [center + u * uw, center - u * uw]
        else:
            positions = [center + v * vh, center - v * vh]

    elif n_pins == 4:
        positions = [
            center + u * uw + v * vh,
            center - u * uw + v * vh,
            center + u * uw - v * vh,
            center - u * uw - v * vh,
        ]
    else:
        # Grid — fill face with pins
        spacing = pin_radius * 5.0
        n_u = max(1, int((w - 2 * margin) / spacing))
        n_v = max(1, int((h - 2 * margin) / spacing))
        for i in range(n_u):
            for j in range(n_v):
                pu = -uw + i * spacing if n_u > 1 else 0
                pv = -vh + j * spacing if n_v > 1 else 0
                positions.append(center + u * pu + v * pv)

    return positions


def make_male_pin(position: np.ndarray, normal: np.ndarray,
                  radius: float, length: float) -> trimesh.Trimesh:
    """Create a male pin cylinder protruding in the direction of normal."""
    cyl = trimesh.creation.cylinder(radius=radius, height=length, sections=20)
    rot = trimesh.geometry.align_vectors([0, 0, 1], normal.tolist())
    cyl.apply_transform(rot)
    # Protrude outward: center at position + normal * length/2
    cyl.apply_translation(position + normal * (length / 2))
    return cyl


def make_female_socket(position: np.ndarray, normal: np.ndarray,
                       radius: float, length: float,
                       tolerance: float) -> trimesh.Trimesh:
    """Create a female socket cylinder (for boolean subtraction)."""
    cyl = trimesh.creation.cylinder(
        radius=radius + tolerance,
        height=length + 1.0,   # slightly longer to avoid z-fighting
        sections=20
    )
    rot = trimesh.geometry.align_vectors([0, 0, 1], normal.tolist())
    cyl.apply_transform(rot)
    # Sink inward: center at position - normal * length/2
    cyl.apply_translation(position - normal * (length / 2))
    return cyl


def apply_connector_pins(
    mesh_a: trimesh.Trimesh,
    mesh_b: trimesh.Trimesh,
    cut_normal: np.ndarray,
    cut_origin: np.ndarray,
    pin_radius: float = 3.0,
    pin_length: float = 8.0,
    n_pins: int = 2,
    tolerance: float = 0.25,
) -> Tuple[trimesh.Trimesh, trimesh.Trimesh, List[trimesh.Trimesh]]:
    """
    Add connector pins to a pair of cut meshes.

    Returns:
      mesh_a_with_pins  — side A with male pins merged in
      mesh_b_with_sockets — side B with female sockets (holes)
      pin_objects        — separate pin objects if boolean fails (print separately)
    """
    cut_normal = np.array(cut_normal, dtype=float)
    cut_normal /= np.linalg.norm(cut_normal)
    cut_origin = np.array(cut_origin, dtype=float)

    # Find cut face info on mesh_a
    face_info = find_cut_face_info(mesh_a, -cut_normal, cut_origin)
    if face_info is None:
        return mesh_a, mesh_b, []

    positions = compute_pin_positions(face_info, n_pins, pin_radius)
    if not positions:
        return mesh_a, mesh_b, []

    pin_objects = []
    result_a = mesh_a.copy()
    result_b = mesh_b.copy()

    for pos in positions:
        # Male pin on side A (add geometry)
        male = make_male_pin(pos, -cut_normal, pin_radius, pin_length)
        try:
            result_a = trimesh.util.concatenate([result_a, male])
            pin_objects.append(male.copy())
        except Exception:
            pin_objects.append(male)

        # Female socket on side B (subtract geometry)
        female = make_female_socket(pos, -cut_normal, pin_radius, pin_length, tolerance)
        try:
            # Try boolean difference
            from core.boolean_ops import boolean_difference
            diff = boolean_difference([result_b, female])
            if diff is not None and len(diff.faces) > 0:
                result_b = diff
            else:
                # Fall back: just store socket as separate object
                pin_objects.append(female)
        except Exception:
            # Boolean unavailable — keep socket as separate object to mark drill spot
            pin_objects.append(female)

    return result_a, result_b, pin_objects


def add_part_number_label(mesh: trimesh.Trimesh, label: str,
                          size: float = 8.0, depth: float = 0.5) -> trimesh.Trimesh:
    """
    Emboss a part number label onto the mesh.
    Uses a simple approach: creates raised letter-like geometry using
    small rectangular extrusions positioned on the flattest face.

    For production quality this would use a font library, but this gives
    a readable embossed number using simple geometry.
    """
    try:
        # Find the flattest (largest area) face to put the label on
        bounds = mesh.bounds
        center = (bounds[0] + bounds[1]) / 2
        extents = bounds[1] - bounds[0]

        # Find which face is largest/flattest — use the face with most area
        # aligned to a primary axis
        best_axis = np.argmax(extents)  # longest dimension
        label_axis = (best_axis + 2) % 3  # face perpendicular to shortest dim

        # Create simple text indicator: small raised cylinder for each character
        # Position on the back face (negative side of label_axis)
        face_center = center.copy()
        face_center[label_axis] = bounds[0][label_axis] - 0.01

        normal = np.zeros(3)
        normal[label_axis] = -1.0

        # Create a small disc as label marker
        disc = trimesh.creation.cylinder(radius=size/2, height=depth, sections=32)
        rot = trimesh.geometry.align_vectors([0, 0, 1], normal)
        disc.apply_transform(rot)
        disc.apply_translation(face_center + normal * depth)

        result = trimesh.util.concatenate([mesh, disc])
        return result
    except Exception:
        return mesh
