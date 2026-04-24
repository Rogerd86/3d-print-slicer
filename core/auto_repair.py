"""
auto_repair.py
Comprehensive automatic mesh repair pipeline.

Runs a sequence of diagnostic checks and fixes, returning a detailed
report of what was found and what was fixed. Inspired by Netfabb's
Fix Wizard and Meshmixer's Inspector.

Checks and fixes (in order):
  1. Remove degenerate faces (zero-area triangles)
  2. Remove duplicate faces
  3. Remove unreferenced vertices
  4. Merge close vertices (snap together near-duplicates)
  5. Fix face winding consistency
  6. Fix inverted normals (ensure outward-facing)
  7. Fill holes (close gaps in the mesh surface)
  8. Remove small disconnected shells (floating debris)
  9. Detect and report non-manifold edges
  10. Detect and report self-intersections (slow — optional)
  11. Report watertight status
  12. Report overall mesh health score
"""
import numpy as np
import trimesh
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class RepairIssue:
    """A single detected issue."""
    category: str       # e.g., 'degenerate_faces', 'holes', 'normals'
    severity: str       # 'error', 'warning', 'info'
    count: int = 0      # how many instances
    description: str = ""
    fixed: bool = False
    fix_description: str = ""


@dataclass
class RepairReport:
    """Full repair report."""
    issues: List[RepairIssue] = field(default_factory=list)
    triangles_before: int = 0
    triangles_after: int = 0
    vertices_before: int = 0
    vertices_after: int = 0
    was_watertight: bool = False
    is_watertight: bool = False
    health_score: int = 0  # 0-100
    total_fixes: int = 0

    def add(self, issue: RepairIssue):
        self.issues.append(issue)
        if issue.fixed:
            self.total_fixes += 1

    def summary(self) -> str:
        """Human-readable summary."""
        lines = []
        errors = [i for i in self.issues if i.severity == 'error' and not i.fixed]
        warnings = [i for i in self.issues if i.severity == 'warning']
        fixes = [i for i in self.issues if i.fixed]

        if fixes:
            lines.append(f"✓ {len(fixes)} issues fixed:")
            for f in fixes:
                lines.append(f"  ✓ {f.fix_description}")

        if errors:
            lines.append(f"⚠ {len(errors)} remaining issues:")
            for e in errors:
                lines.append(f"  ⚠ {e.description}")

        if not errors and not warnings:
            lines.append("✓ Mesh is clean — no issues found.")

        lines.append(f"\nHealth: {self.health_score}/100  |  "
                     f"Tris: {self.triangles_before:,} → {self.triangles_after:,}  |  "
                     f"{'Watertight ✓' if self.is_watertight else 'Not watertight ⚠'}")
        return "\n".join(lines)

    def issues_by_severity(self) -> Dict[str, List[RepairIssue]]:
        result = {'error': [], 'warning': [], 'info': []}
        for i in self.issues:
            result.get(i.severity, []).append(i)
        return result


def full_repair(mesh: trimesh.Trimesh,
                remove_small_shells: bool = True,
                min_shell_ratio: float = 0.01,
                check_self_intersections: bool = False,
                aggressive: bool = False,
                print_ready: bool = False,
                min_wall_mm: float = 0.8,
                ) -> Tuple[trimesh.Trimesh, RepairReport]:
    """
    Run the full repair pipeline on a mesh.

    Parameters
    ----------
    mesh : input mesh (will be modified in place)
    remove_small_shells : remove disconnected components smaller than min_shell_ratio
    min_shell_ratio : shells with fewer faces than this ratio of total are removed
    check_self_intersections : run slow self-intersection check (can take minutes)

    Returns
    -------
    (repaired_mesh, report)
    """
    report = RepairReport()
    report.triangles_before = len(mesh.faces)
    report.vertices_before = len(mesh.vertices)
    report.was_watertight = mesh.is_watertight

    # === Step 1: Remove degenerate faces ===
    try:
        areas = mesh.area_faces
        degenerate = np.sum(areas < 1e-10)
        if degenerate > 0:
            mask = areas >= 1e-10
            mesh.update_faces(mask)
            report.add(RepairIssue(
                'degenerate_faces', 'warning', degenerate,
                f"{degenerate} zero-area triangles",
                True, f"Removed {degenerate} degenerate faces"))
    except Exception as e:
        report.add(RepairIssue('degenerate_faces', 'info', 0, f"Check skipped: {e}"))

    # === Step 2: Remove duplicate faces ===
    try:
        before = len(mesh.faces)
        try:
            unique_idx = np.unique(np.sort(mesh.faces, axis=1), axis=0, return_index=True)[1]
            if len(unique_idx) < before:
                mesh.update_faces(np.sort(unique_idx))
        except Exception:
            try:
                mesh.remove_duplicate_faces()
            except Exception:
                pass
        removed = before - len(mesh.faces)
        if removed > 0:
            report.add(RepairIssue(
                'duplicate_faces', 'warning', removed,
                f"{removed} duplicate faces",
                True, f"Removed {removed} duplicate faces"))
    except Exception:
        pass

    # === Step 3: Remove unreferenced vertices ===
    try:
        before = len(mesh.vertices)
        mesh.remove_unreferenced_vertices()
        removed = before - len(mesh.vertices)
        if removed > 0:
            report.add(RepairIssue(
                'unreferenced_verts', 'info', removed,
                f"{removed} unreferenced vertices",
                True, f"Removed {removed} orphan vertices"))
    except Exception:
        pass

    # === Step 4: Merge close vertices ===
    try:
        before = len(mesh.vertices)
        mesh.merge_vertices()
        merged = before - len(mesh.vertices)
        if merged > 0:
            report.add(RepairIssue(
                'duplicate_verts', 'info', merged,
                f"{merged} near-duplicate vertices",
                True, f"Merged {merged} close vertices"))
    except Exception:
        pass

    # === Step 5: Fix face winding ===
    try:
        if not mesh.is_winding_consistent:
            trimesh.repair.fix_winding(mesh)
            report.add(RepairIssue(
                'winding', 'warning', 1,
                "Inconsistent face winding",
                True, "Fixed face winding consistency"))
    except Exception:
        pass

    # === Step 6: Fix normals ===
    try:
        trimesh.repair.fix_normals(mesh, multibody=True)
        report.add(RepairIssue(
            'normals', 'info', 0,
            "Normals checked",
            True, "Ensured all normals face outward"))
    except Exception:
        pass

    # === Step 7: Fix inversion ===
    try:
        trimesh.repair.fix_inversion(mesh)
    except Exception:
        pass

    # === Step 8: Fill holes ===
    try:
        if not mesh.is_watertight:
            before_wt = mesh.is_watertight
            trimesh.repair.fill_holes(mesh)
            if mesh.is_watertight and not before_wt:
                report.add(RepairIssue(
                    'holes', 'warning', 1,
                    "Mesh had holes",
                    True, "Filled holes — mesh is now watertight"))
            elif not mesh.is_watertight:
                # Count remaining boundary edges
                try:
                    broken = trimesh.repair.broken_faces(mesh)
                    n_broken = len(broken) if broken is not None else 0
                except Exception:
                    n_broken = 0
                report.add(RepairIssue(
                    'holes', 'error', n_broken,
                    f"Mesh still has {n_broken} boundary faces after hole-filling",
                    False, ""))
    except Exception as e:
        report.add(RepairIssue('holes', 'error', 0, f"Hole filling failed: {e}"))

    # === Step 9: Remove small disconnected shells ===
    if remove_small_shells:
        try:
            components = mesh.split(only_watertight=False)
            if len(components) > 1:
                # Sort by face count, keep largest
                components.sort(key=lambda c: len(c.faces), reverse=True)
                total_faces = sum(len(c.faces) for c in components)
                min_faces = max(10, int(total_faces * min_shell_ratio))

                keep = [c for c in components if len(c.faces) >= min_faces]
                removed = len(components) - len(keep)

                if removed > 0 and keep:
                    mesh = trimesh.util.concatenate(keep)
                    report.add(RepairIssue(
                        'small_shells', 'warning', removed,
                        f"{removed} small disconnected shells",
                        True, f"Removed {removed} tiny floating shells "
                              f"({len(components)} → {len(keep)} components)"))
                elif len(components) > 1:
                    report.add(RepairIssue(
                        'multi_body', 'info', len(components),
                        f"Mesh has {len(components)} separate bodies",
                        False, ""))
        except Exception:
            pass

    # === Step 10: Check non-manifold edges ===
    try:
        if hasattr(mesh, 'edges_unique') and hasattr(mesh, 'faces'):
            # Non-manifold edges: shared by more than 2 faces
            from collections import Counter
            edge_counts = Counter()
            for face in mesh.faces:
                edges = [(min(face[0],face[1]), max(face[0],face[1])),
                         (min(face[1],face[2]), max(face[1],face[2])),
                         (min(face[2],face[0]), max(face[2],face[0]))]
                for e in edges:
                    edge_counts[e] += 1
            non_manifold = sum(1 for c in edge_counts.values() if c > 2)
            if non_manifold > 0:
                report.add(RepairIssue(
                    'non_manifold', 'error', non_manifold,
                    f"{non_manifold} non-manifold edges (shared by >2 faces)",
                    False, ""))
    except Exception:
        pass

    # === Step 11: Self-intersection check (optional, slow) ===
    if check_self_intersections:
        try:
            # Quick check: ray test a sample of faces
            n_sample = min(200, len(mesh.faces))
            indices = np.random.choice(len(mesh.faces), n_sample, replace=False)
            centres = mesh.triangles_center[indices]
            normals = mesh.face_normals[indices]
            origins = centres + normals * 0.001
            _, ray_idx, _ = mesh.ray.intersects_location(origins, -normals)
            self_hits = len(set(ray_idx))
            if self_hits > n_sample * 0.1:
                report.add(RepairIssue(
                    'self_intersection', 'error', self_hits,
                    f"~{self_hits} potential self-intersecting faces detected",
                    False, ""))
        except Exception:
            pass

    # ════════════════════════════════════════════════════════════════
    # AGGRESSIVE / PRINT-READY EXTRA PASSES
    # These run only when explicitly requested because they can be slow
    # or rewrite topology in ways that are overkill for a clean model.
    # ════════════════════════════════════════════════════════════════
    if aggressive or print_ready:
        from core import mesh_quality as mq

        # Extra 1: Self-intersection cleanup (manifold3d round-trip)
        try:
            mesh, msg = mq.fix_self_intersections(mesh)
            if "skipped" not in msg and "failed" not in msg:
                report.add(RepairIssue(
                    'self_intersection', 'error', 1,
                    "Self-intersecting geometry present",
                    True, msg))
        except Exception as e:
            report.add(RepairIssue('self_intersection', 'info', 0,
                                    f"Self-intersect pass error: {e}"))

        # Extra 2: Split non-manifold edges
        try:
            mesh, msg = mq.fix_non_manifold_edges(mesh)
            if "No non-manifold" not in msg:
                report.add(RepairIssue(
                    'non_manifold', 'error', 1,
                    "Non-manifold edges present",
                    True, msg))
        except Exception as e:
            report.add(RepairIssue('non_manifold', 'info', 0,
                                    f"Non-manifold pass error: {e}"))

        # Extra 3: Sliver triangles
        try:
            mesh, msg = mq.fix_sliver_triangles(mesh, max_aspect=60.0)
            if "No sliver" not in msg:
                report.add(RepairIssue(
                    'slivers', 'warning', 1,
                    "High-aspect-ratio triangles",
                    True, msg))
        except Exception as e:
            report.add(RepairIssue('slivers', 'info', 0,
                                    f"Sliver pass error: {e}"))

        # Extra 4: Tiny floating vertex clusters
        try:
            mesh, msg = mq.fix_floating_vertex_clusters(mesh, max_cluster_faces=4)
            if "No floating" not in msg:
                report.add(RepairIssue(
                    'floaters', 'warning', 1,
                    "Tiny floating geometry",
                    True, msg))
        except Exception as e:
            pass

    # Print-ready only: scan for thin walls and REPORT (does not fix — that
    # requires solid-thickening which changes the model silhouette).
    if print_ready:
        from core import mesh_quality as mq
        try:
            info = mq.analyse_thin_walls(mesh, min_thickness=min_wall_mm)
            if info.get('pct_thin', 0) > 0.5:
                report.add(RepairIssue(
                    'thin_walls', 'warning',
                    len(info.get('thin_faces', [])),
                    f"{info['pct_thin']:.1f}% of sampled walls are thinner "
                    f"than {min_wall_mm} mm (min seen: {info['min_seen']:.2f} mm). "
                    "These will under-extrude — thicken in CAD or use vase mode.",
                    False, ""))
        except Exception:
            pass

    # === Final status ===
    report.triangles_after = len(mesh.faces)
    report.vertices_after = len(mesh.vertices)
    report.is_watertight = mesh.is_watertight

    # Health score
    score = 100
    for issue in report.issues:
        if issue.severity == 'error' and not issue.fixed:
            score -= 25
        elif issue.severity == 'warning' and not issue.fixed:
            score -= 10
    if not mesh.is_watertight:
        score -= 20
    report.health_score = max(0, min(100, score))

    return mesh, report


def quick_diagnose(mesh: trimesh.Trimesh) -> Dict:
    """
    Quick diagnosis without fixing anything.
    Returns a dict of issues found.
    """
    result = {
        'triangles': len(mesh.faces),
        'vertices': len(mesh.vertices),
        'is_watertight': mesh.is_watertight,
        'is_winding_consistent': mesh.is_winding_consistent,
        'body_count': 1,
        'degenerate_faces': 0,
        'duplicate_faces': 0,
        'issues': [],
    }

    try:
        areas = mesh.area_faces
        result['degenerate_faces'] = int(np.sum(areas < 1e-10))
        if result['degenerate_faces'] > 0:
            result['issues'].append(f"{result['degenerate_faces']} degenerate faces")
    except Exception:
        pass

    try:
        unique = np.unique(np.sort(mesh.faces, axis=1), axis=0)
        result['duplicate_faces'] = len(mesh.faces) - len(unique)
        if result['duplicate_faces'] > 0:
            result['issues'].append(f"{result['duplicate_faces']} duplicate faces")
    except Exception:
        pass

    try:
        components = mesh.split(only_watertight=False)
        result['body_count'] = len(components)
        if len(components) > 1:
            result['issues'].append(f"{len(components)} separate bodies/shells")
    except Exception:
        pass

    if not mesh.is_watertight:
        result['issues'].append("Not watertight (has holes)")
    if not mesh.is_winding_consistent:
        result['issues'].append("Inconsistent face winding")

    return result


def simplify_mesh(mesh: trimesh.Trimesh,
                   target_ratio: float = 0.5,
                   preserve_topology: bool = True) -> Tuple[trimesh.Trimesh, Dict]:
    """
    Reduce polygon count while preserving shape.

    target_ratio: fraction of faces to keep (0.5 = half)

    Returns (simplified_mesh, stats)
    """
    before = len(mesh.faces)
    target = max(100, int(before * target_ratio))

    try:
        # Try fast_simplification first (quadric decimation)
        simplified = mesh.simplify_quadric_decimation(target)
        after = len(simplified.faces)
        return simplified, {
            'before': before, 'after': after,
            'reduction_pct': round((1 - after / before) * 100, 1),
            'method': 'quadric_decimation',
        }
    except Exception:
        pass

    try:
        # Fallback: vertex clustering
        from trimesh.remesh import subdivide
        # Can't easily decimate with just trimesh — return original with note
        return mesh, {
            'before': before, 'after': before,
            'reduction_pct': 0,
            'method': 'none (install fast_simplification for decimation)',
        }
    except Exception:
        return mesh, {'before': before, 'after': before, 'reduction_pct': 0, 'method': 'failed'}


def fix_with_pymeshfix(mesh: trimesh.Trimesh) -> Tuple[trimesh.Trimesh, str]:
    """
    Use PyMeshFix for deep repair — handles self-intersections
    and complex topology that trimesh's built-in repair can't fix.
    """
    try:
        import pymeshfix
        tin = pymeshfix.MeshFix(mesh.vertices, mesh.faces)
        tin.repair()
        pv_mesh = tin.mesh
        verts = np.array(pv_mesh.points)
        faces_raw = pv_mesh.faces.reshape(-1, 4)[:, 1:]
        fixed = trimesh.Trimesh(vertices=verts, faces=faces_raw)
        before = len(mesh.faces)
        after = len(fixed.faces)
        wt = fixed.is_watertight
        return fixed, (f"PyMeshFix: {before} → {after} faces, "
                       f"{'watertight ✓' if wt else 'improved but not watertight'}")
    except ImportError:
        return mesh, "PyMeshFix not installed (pip install pymeshfix pyvista)"
    except Exception as e:
        return mesh, f"PyMeshFix error: {e}"


# ═══════════════════════════════════════════════════════════
# INDIVIDUAL REPAIR STEPS — each can be run and undone separately
# ═══════════════════════════════════════════════════════════

def fix_degenerate_faces(mesh: trimesh.Trimesh) -> Tuple[trimesh.Trimesh, str]:
    """Remove zero-area triangles. Returns (mesh, description)."""
    areas = mesh.area_faces
    bad = int(np.sum(areas < 1e-10))
    if bad == 0:
        return mesh, "No degenerate faces found."
    mask = areas >= 1e-10
    mesh.update_faces(mask)
    mesh.remove_unreferenced_vertices()
    return mesh, f"Removed {bad} degenerate faces."


def fix_duplicate_faces(mesh: trimesh.Trimesh) -> Tuple[trimesh.Trimesh, str]:
    """Remove duplicate triangles. Returns (mesh, description)."""
    before = len(mesh.faces)
    try:
        # Try newer trimesh API
        unique = np.unique(np.sort(mesh.faces, axis=1), axis=0, return_index=True)[1]
        if len(unique) < before:
            mesh.update_faces(np.sort(unique))
    except Exception:
        try:
            mesh.remove_duplicate_faces()
        except Exception:
            pass
    removed = before - len(mesh.faces)
    if removed == 0:
        return mesh, "No duplicate faces found."
    return mesh, f"Removed {removed} duplicate faces."


def fix_normals(mesh: trimesh.Trimesh) -> Tuple[trimesh.Trimesh, str]:
    """Fix face normals and winding. Returns (mesh, description)."""
    was_consistent = mesh.is_winding_consistent
    trimesh.repair.fix_winding(mesh)
    trimesh.repair.fix_normals(mesh, multibody=True)
    trimesh.repair.fix_inversion(mesh)
    if was_consistent:
        return mesh, "Normals verified and ensured outward-facing."
    return mesh, "Fixed inconsistent normals and winding direction."


def fix_holes(mesh: trimesh.Trimesh) -> Tuple[trimesh.Trimesh, str]:
    """Fill holes in the mesh. Returns (mesh, description)."""
    was_wt = mesh.is_watertight
    if was_wt:
        return mesh, "Mesh is already watertight — no holes to fill."
    trimesh.repair.fill_holes(mesh)
    if mesh.is_watertight:
        return mesh, "Filled holes — mesh is now watertight."
    return mesh, "Attempted hole filling — some holes may remain."


def fix_merge_vertices(mesh: trimesh.Trimesh) -> Tuple[trimesh.Trimesh, str]:
    """Merge near-duplicate vertices. Returns (mesh, description)."""
    before = len(mesh.vertices)
    mesh.merge_vertices()
    mesh.remove_unreferenced_vertices()
    merged = before - len(mesh.vertices)
    if merged == 0:
        return mesh, "No duplicate vertices found."
    return mesh, f"Merged {merged} close vertices."


def fix_remove_shells(mesh: trimesh.Trimesh,
                       min_ratio: float = 0.01) -> Tuple[trimesh.Trimesh, str]:
    """Remove small disconnected shells. Returns (mesh, description)."""
    components = mesh.split(only_watertight=False)
    if len(components) <= 1:
        return mesh, "Only 1 body — no shells to remove."
    components.sort(key=lambda c: len(c.faces), reverse=True)
    total = sum(len(c.faces) for c in components)
    min_faces = max(10, int(total * min_ratio))
    keep = [c for c in components if len(c.faces) >= min_faces]
    removed = len(components) - len(keep)
    if removed == 0:
        return mesh, f"All {len(components)} bodies are large enough — none removed."
    result = trimesh.util.concatenate(keep)
    return result, f"Removed {removed} small shells ({len(components)} → {len(keep)} bodies)."
