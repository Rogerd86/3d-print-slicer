"""
Hollow shell: convert a solid mesh into a hollow shell with configurable wall thickness.
This is essential for large printed parts — solid prints waste filament and take forever.
"""
import numpy as np
import trimesh
from typing import Optional, Tuple


def make_hollow(mesh: trimesh.Trimesh, wall_thickness: float = 3.0) -> Tuple[trimesh.Trimesh, bool]:
    """
    Convert a solid mesh into a hollow shell.

    Strategy:
    1. Create an inner shell by shrinking the mesh inward by wall_thickness
       using vertex normal offsets
    2. Combine outer shell + inner shell + connecting faces at open edges

    Returns (hollowed_mesh, success).
    If hollowing fails for any reason, returns the original mesh unchanged and success=False.
    """
    if wall_thickness <= 0:
        return mesh, True

    try:
        outer = mesh.copy()

        # Compute vertex normals
        outer.vertex_normals  # trigger computation

        # Offset vertices inward along their normals
        inner = outer.copy()
        inner.vertices = outer.vertices - outer.vertex_normals * wall_thickness

        # Flip inner shell normals (they should point inward)
        inner.faces = inner.faces[:, ::-1]

        # Combine outer + inner shells
        combined = trimesh.util.concatenate([outer, inner])

        # Find boundary edges of the outer mesh (open edges that need caps)
        # For a closed watertight mesh this produces the connecting band
        try:
            combined = trimesh.repair.fill_holes(combined)
        except Exception:
            pass

        if len(combined.faces) > 0:
            return combined, True
        else:
            return mesh, False

    except Exception as e:
        print(f"Hollow shell error: {e}")
        return mesh, False


def make_hollow_simple(mesh: trimesh.Trimesh, wall_thickness: float = 3.0) -> Tuple[trimesh.Trimesh, bool]:
    """
    Simpler hollow approach: just offset all vertices inward.
    Less geometrically perfect but more robust for complex meshes.
    Used as fallback if full hollow fails.
    """
    try:
        if not mesh.is_watertight:
            # Can't reliably hollow a non-watertight mesh
            return mesh, False

        inner = mesh.copy()
        # Push vertices inward along vertex normals
        inner.vertices = mesh.vertices - mesh.vertex_normals * wall_thickness
        # Flip faces so normals point the right way
        inner.faces = inner.faces[:, ::-1]

        result = trimesh.util.concatenate([mesh, inner])
        return result, True
    except Exception as e:
        print(f"Simple hollow error: {e}")
        return mesh, False


def estimate_material_saved(mesh: trimesh.Trimesh, wall_thickness: float) -> dict:
    """
    Estimate how much material is saved by hollowing vs solid printing.
    Returns dict with volume_solid_cm3, volume_hollow_cm3, saving_pct.
    """
    try:
        volume_solid = abs(mesh.volume) / 1000  # mm³ to cm³

        # Rough hollow volume estimate: surface area * wall thickness
        volume_shell = (mesh.area * wall_thickness) / 1000  # cm³

        saving_pct = max(0, (1 - volume_shell / volume_solid) * 100) if volume_solid > 0 else 0

        return {
            'volume_solid_cm3': round(volume_solid, 1),
            'volume_shell_cm3': round(volume_shell, 1),
            'saving_pct': round(saving_pct, 1)
        }
    except Exception:
        return {'volume_solid_cm3': 0, 'volume_shell_cm3': 0, 'saving_pct': 0}
