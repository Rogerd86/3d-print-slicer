"""
Mesh repair: fix common issues in downloaded STL/OBJ/3MF files.
Returns a report of what was fixed.
"""
import numpy as np
import trimesh
from typing import Tuple, List


class RepairReport:
    def __init__(self):
        self.issues_found: List[str] = []
        self.fixes_applied: List[str] = []
        self.was_watertight_before = False
        self.is_watertight_after = False
        self.triangles_before = 0
        self.triangles_after = 0

    def summary(self) -> str:
        if not self.issues_found:
            return "✓ Mesh is clean — no repairs needed."
        lines = []
        for issue in self.issues_found:
            lines.append(f"  ⚠ {issue}")
        for fix in self.fixes_applied:
            lines.append(f"  ✓ {fix}")
        if self.is_watertight_after:
            lines.append("  ✓ Mesh is now watertight")
        else:
            lines.append("  ⚠ Mesh still has issues — slicing may produce gaps")
        return "\n".join(lines)


def repair_mesh(mesh: trimesh.Trimesh) -> Tuple[trimesh.Trimesh, RepairReport]:
    """
    Run a full repair pass on a mesh.
    Returns the (possibly modified) mesh and a report.
    """
    report = RepairReport()
    report.triangles_before = len(mesh.faces)
    report.was_watertight_before = mesh.is_watertight

    # --- Diagnose ---
    if not mesh.is_watertight:
        report.issues_found.append("Mesh is not watertight (has holes or open edges)")

    if hasattr(mesh, 'faces') and len(mesh.faces) == 0:
        report.issues_found.append("Mesh has no faces")
        return mesh, report

    # Check for degenerate faces (zero-area triangles)
    areas = mesh.area_faces
    degenerate = np.sum(areas < 1e-10)
    if degenerate > 0:
        report.issues_found.append(f"{degenerate} degenerate (zero-area) triangles")

    # Check for duplicate faces
    try:
        unique_faces = np.unique(np.sort(mesh.faces, axis=1), axis=0)
        dupes = len(mesh.faces) - len(unique_faces)
        if dupes > 0:
            report.issues_found.append(f"{dupes} duplicate faces")
    except Exception:
        pass

    # --- Fix normals ---
    try:
        trimesh.repair.fix_normals(mesh, multibody=False)
        report.fixes_applied.append("Fixed face normals")
    except Exception as e:
        pass

    # --- Fix winding ---
    try:
        trimesh.repair.fix_winding(mesh)
        report.fixes_applied.append("Fixed face winding")
    except Exception:
        pass

    # --- Fill holes ---
    try:
        if not mesh.is_watertight:
            trimesh.repair.fill_holes(mesh)
            report.fixes_applied.append("Filled holes in mesh")
    except Exception:
        pass

    # --- Remove degenerate faces ---
    try:
        mask = mesh.area_faces > 1e-10
        if not np.all(mask):
            mesh.update_faces(mask)
            report.fixes_applied.append(f"Removed {np.sum(~mask)} degenerate triangles")
    except Exception:
        pass

    # --- Remove duplicate faces ---
    try:
        mesh.remove_duplicate_faces()
        report.fixes_applied.append("Removed duplicate faces")
    except Exception:
        pass

    # --- Remove unreferenced vertices ---
    try:
        mesh.remove_unreferenced_vertices()
        report.fixes_applied.append("Removed unreferenced vertices")
    except Exception:
        pass

    # --- Merge vertices ---
    try:
        mesh.merge_vertices()
        report.fixes_applied.append("Merged duplicate vertices")
    except Exception:
        pass

    report.triangles_after = len(mesh.faces)
    report.is_watertight_after = mesh.is_watertight

    if not report.issues_found:
        report.issues_found = []  # clean bill of health

    return mesh, report
