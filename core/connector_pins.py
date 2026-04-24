"""
connector_pins.py — adds real cylinder geometry to cut faces.
Male pins on one side, female holes on the other.
"""
import numpy as np
import trimesh
from typing import List, Tuple


def add_pins_to_parts(mesh_a, mesh_b, cut_normal, cut_origin,
                       pin_radius=3.0, pin_depth=6.0, tolerance=0.2, pins_per_face=2):
    """
    Add connector pins between two meshes sharing a cut face.
    mesh_a gets protruding pins, mesh_b gets holes.
    Returns (mesh_a_pinned, mesh_b_holed) — falls back to originals on failure.
    """
    try:
        positions = _find_pin_positions(mesh_a, cut_normal, cut_origin, pins_per_face, pin_radius)
        if not positions:
            return mesh_a, mesh_b
        result_a = _add_male_pins(mesh_a, positions, cut_normal, pin_radius, pin_depth)
        result_b = _add_female_holes(mesh_b, positions, cut_normal, pin_radius + tolerance, pin_depth)
        return result_a, result_b
    except Exception as e:
        print(f"Pin generation error: {e}")
        return mesh_a, mesh_b


def _find_pin_positions(mesh, normal, origin, count, min_spacing):
    face_centres = mesh.triangles_center
    dot = np.dot(face_centres - origin, normal)
    mask = np.abs(dot) < 2.0
    if not np.any(mask):
        return []
    cut_verts = face_centres[mask]

    u = np.cross(normal, np.array([0, 0, 1.0]))
    if np.linalg.norm(u) < 0.01:
        u = np.cross(normal, np.array([0, 1.0, 0]))
    u /= np.linalg.norm(u)
    v = np.cross(normal, u); v /= np.linalg.norm(v)

    u_coords = cut_verts @ u; v_coords = cut_verts @ v
    u_min, u_max = u_coords.min(), u_coords.max()
    v_min, v_max = v_coords.min(), v_coords.max()
    spacing_u = (u_max - u_min) / (count + 1)

    positions = []
    for i in range(count):
        u_pos = u_min + spacing_u * (i + 1)
        v_pos = (v_min + v_max) / 2
        positions.append(origin + u_pos * u + v_pos * v)
    return positions


def _add_male_pins(mesh, positions, normal, radius, depth):
    parts = [mesh]
    for pos in positions:
        try:
            pin = trimesh.creation.cylinder(radius=radius, height=depth, sections=16)
            rot = trimesh.geometry.align_vectors([0, 0, 1], normal.tolist())
            pin.apply_transform(rot)
            pin.apply_translation(pos + normal * (depth / 2))
            parts.append(pin)
        except Exception as e:
            print(f"Male pin error: {e}")
    return trimesh.util.concatenate(parts) if len(parts) > 1 else mesh


def _add_female_holes(mesh, positions, normal, radius, depth):
    if not mesh.is_watertight:
        return mesh
    result = mesh
    for pos in positions:
        try:
            cutter = trimesh.creation.cylinder(radius=radius, height=depth+2, sections=16)
            rot = trimesh.geometry.align_vectors([0, 0, 1], normal.tolist())
            cutter.apply_transform(rot)
            cutter.apply_translation(pos - normal * (depth / 2))
            try:
                from core.boolean_ops import boolean_difference
                diff = boolean_difference([result, cutter])
                if diff is not None and len(diff.faces) > 0:
                    result = diff
            except Exception:
                pass
        except Exception as e:
            print(f"Female hole error: {e}")
    return result


def generate_separate_pin(radius=3.0, length=12.0):
    """Standalone printable pin to insert into both holes."""
    return trimesh.creation.cylinder(radius=radius*0.97, height=length, sections=16)
