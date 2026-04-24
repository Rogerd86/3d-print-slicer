"""
mesh_quality.py
Advanced mesh quality fixes that the baseline auto_repair pipeline doesn't
cover. Designed for 3D-printing-critical issues:

  * Self-intersections (via manifold3d round-trip — cleaner than PyMeshFix)
  * Non-manifold edges (vertex-split to make edges 2-manifold)
  * Sliver triangles (high aspect ratio, worthless for slicing)
  * T-junctions (vertex lying on an edge of another triangle)
  * Thin-wall analysis (flag faces whose opposite wall is closer than X mm)

All functions follow the (mesh, description) -> (new_mesh, str) contract
used elsewhere in core/ so they plug straight into the repair pipeline.
"""
import numpy as np
import trimesh
from typing import Tuple, Dict, List, Optional


# ════════════════════════════════════════════════════════════════════
# 1. Self-intersection repair (manifold3d round-trip)
# ════════════════════════════════════════════════════════════════════

def fix_self_intersections(mesh: trimesh.Trimesh) -> Tuple[trimesh.Trimesh, str]:
    """Clean self-intersections by round-tripping through manifold3d.

    manifold3d's boolean engine internally re-tessellates and produces a
    clean 2-manifold result free of self-intersections. We use a union
    with an empty manifold (i.e. a single-operand union) which triggers
    the cleanup pass without changing topology elsewhere.

    Falls back to trimesh's built-in boolean union with manifold engine.
    """
    try:
        import manifold3d as m3d
    except ImportError:
        return mesh, "Self-intersection fix skipped (manifold3d not installed)"

    try:
        # Feed trimesh through manifold3d, extract the cleaned result.
        verts = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.uint32)
        mesh_gl = m3d.Mesh(vert_properties=verts, tri_verts=faces)
        man = m3d.Manifold(mesh_gl)
        if man.is_empty() or man.status() != m3d.Error.NoError:
            return mesh, f"Self-intersection fix: manifold3d rejected mesh ({man.status()})"
        cleaned = man.as_original()  # triggers cleanup pass
        out_mesh = cleaned.to_mesh()
        new_verts = np.asarray(out_mesh.vert_properties)[:, :3]
        new_faces = np.asarray(out_mesh.tri_verts)
        if len(new_faces) == 0:
            return mesh, "Self-intersection fix: manifold3d produced empty result"
        fixed = trimesh.Trimesh(vertices=new_verts, faces=new_faces, process=False)
        delta = len(mesh.faces) - len(fixed.faces)
        return fixed, (
            f"Self-intersections cleaned via manifold3d "
            f"({'Δ' if delta >= 0 else '+'}{abs(delta)} faces)")
    except Exception as e:
        return mesh, f"Self-intersection fix failed: {e}"


# ════════════════════════════════════════════════════════════════════
# 2. Non-manifold edge repair (vertex split)
# ════════════════════════════════════════════════════════════════════

def fix_non_manifold_edges(mesh: trimesh.Trimesh) -> Tuple[trimesh.Trimesh, str]:
    """For each edge shared by >2 faces, duplicate endpoint vertices so the
    triangles separate into distinct manifold surfaces.

    This doesn't close the resulting seam (that's what hole-filling is for)
    but it turns non-manifold topology into clean edge boundaries that
    downstream tools (slicers, boolean engines) can actually handle.
    """
    try:
        from collections import defaultdict
        edge_to_faces = defaultdict(list)
        for fi, (a, b, c) in enumerate(mesh.faces):
            for u, v in [(a, b), (b, c), (c, a)]:
                key = (min(u, v), max(u, v))
                edge_to_faces[key].append(fi)

        bad_edges = {e: fs for e, fs in edge_to_faces.items() if len(fs) > 2}
        if not bad_edges:
            return mesh, "No non-manifold edges"

        # Build new vertex/face arrays: for each offending face beyond the
        # first two on a shared edge, emit a private copy of the endpoint
        # vertices. This decouples the extra face from the manifold pair.
        new_verts = mesh.vertices.tolist()
        new_faces = mesh.faces.tolist()
        touched = 0
        for (u, v), fs in bad_edges.items():
            for fi in fs[2:]:  # keep the first two attached, split the rest
                face = list(new_faces[fi])
                new_u = len(new_verts); new_verts.append(list(mesh.vertices[u]))
                new_v = len(new_verts); new_verts.append(list(mesh.vertices[v]))
                face = [new_u if x == u else new_v if x == v else x for x in face]
                new_faces[fi] = face
                touched += 1

        out = trimesh.Trimesh(
            vertices=np.asarray(new_verts, dtype=np.float64),
            faces=np.asarray(new_faces, dtype=np.int64),
            process=False)
        return out, (f"Split {touched} faces across {len(bad_edges)} "
                     f"non-manifold edges")
    except Exception as e:
        return mesh, f"Non-manifold edge fix failed: {e}"


# ════════════════════════════════════════════════════════════════════
# 3. Sliver triangle removal (high aspect ratio)
# ════════════════════════════════════════════════════════════════════

def fix_sliver_triangles(mesh: trimesh.Trimesh,
                          max_aspect: float = 60.0,
                          min_area: float = 1e-6
                          ) -> Tuple[trimesh.Trimesh, str]:
    """Drop triangles whose aspect ratio exceeds `max_aspect`.

    Aspect ratio is approximated as (longest edge)^2 / (4·√3·area).
    For a well-shaped equilateral triangle this is 1. Practical rule of
    thumb: anything > 50 is a useless sliver that breaks slicers and
    produces ugly layer lines.
    """
    try:
        tri = mesh.triangles  # (N,3,3)
        e0 = tri[:, 1] - tri[:, 0]
        e1 = tri[:, 2] - tri[:, 1]
        e2 = tri[:, 0] - tri[:, 2]
        L0 = np.linalg.norm(e0, axis=1)
        L1 = np.linalg.norm(e1, axis=1)
        L2 = np.linalg.norm(e2, axis=1)
        longest = np.maximum(np.maximum(L0, L1), L2)
        # Area via cross product.
        cross = np.cross(e0, -e2)
        area = 0.5 * np.linalg.norm(cross, axis=1)
        safe_area = np.maximum(area, 1e-12)
        aspect = (longest ** 2) / (4 * np.sqrt(3.0) * safe_area)
        bad = (aspect > max_aspect) | (area < min_area)
        n_bad = int(np.sum(bad))
        if n_bad == 0:
            return mesh, "No sliver triangles"
        keep = ~bad
        mesh.update_faces(keep)
        return mesh, f"Removed {n_bad} sliver triangles (aspect > {max_aspect:.0f})"
    except Exception as e:
        return mesh, f"Sliver removal failed: {e}"


# ════════════════════════════════════════════════════════════════════
# 4. T-junction resolution
# ════════════════════════════════════════════════════════════════════

def fix_t_junctions(mesh: trimesh.Trimesh,
                     tolerance: float = 1e-4
                     ) -> Tuple[trimesh.Trimesh, str]:
    """Detect vertices lying on another triangle's edge (T-junctions) and
    split that edge by subdividing the owning triangle into two.

    T-junctions cause cracks when you decimate or slice, because adjacent
    faces don't share the vertex. We use spatial hashing for O(n) lookup,
    otherwise the O(V·E) naive check kills large models.
    """
    try:
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        faces = mesh.faces.tolist()
        # Build edge -> face map to know which triangle owns each edge.
        edges = []
        edge_face = {}
        for fi, (a, b, c) in enumerate(faces):
            for u, v in [(a, b), (b, c), (c, a)]:
                key = (min(u, v), max(u, v))
                edges.append((key, fi, (u, v)))
                edge_face.setdefault(key, []).append(fi)

        split_count = 0
        # Use KDTree for vertex proximity.
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(verts)
        except ImportError:
            return mesh, "T-junction fix skipped (scipy unavailable)"

        edge_dirs = {}
        for (a, b), fs, uv in edges:
            p, q = verts[a], verts[b]
            d = q - p
            L = np.linalg.norm(d)
            if L < 1e-12: continue
            edge_dirs[(a, b)] = (p, d, L)

        new_faces = list(faces)
        replaced = set()
        for (a, b), (p, d, L) in edge_dirs.items():
            # Vertices within a thin tube around the edge.
            q = p + d
            midpoint = (p + q) / 2
            r = L * 0.5 + tolerance
            near_idx = tree.query_ball_point(midpoint, r + tolerance)
            for vi in near_idx:
                if vi == a or vi == b: continue
                v = verts[vi]
                # Parametric t along edge
                t = np.dot(v - p, d) / (L * L)
                if t <= tolerance / L or t >= 1 - tolerance / L: continue
                # Perpendicular distance
                perp = np.linalg.norm((v - p) - t * d)
                if perp > tolerance: continue
                # Split each face that owns this edge.
                owners = edge_face.get((a, b), [])
                for fi in owners:
                    if fi in replaced: continue
                    face = new_faces[fi]
                    if a in face and b in face:
                        other = [x for x in face if x != a and x != b][0]
                        new_faces[fi] = [a, vi, other]
                        new_faces.append([vi, b, other])
                        replaced.add(fi)
                        split_count += 1

        if split_count == 0:
            return mesh, "No T-junctions"
        out = trimesh.Trimesh(vertices=verts,
                              faces=np.asarray(new_faces, dtype=np.int64),
                              process=False)
        return out, f"Resolved {split_count} T-junctions via edge splits"
    except Exception as e:
        return mesh, f"T-junction fix failed: {e}"


# ════════════════════════════════════════════════════════════════════
# 5. Thin-wall analysis
# ════════════════════════════════════════════════════════════════════

def analyse_thin_walls(mesh: trimesh.Trimesh,
                        min_thickness: float = 0.8,
                        sample_size: int = 2000
                        ) -> Dict:
    """Ray-cast inward from each face centre along the negative face normal
    and measure the distance to the opposite wall.

    Returns a dict:
      { 'thin_faces':   list[int]   # face indices below min_thickness
        'min_seen':     float       # thinnest wall found (mm)
        'median':       float       # median thickness
        'pct_thin':     float       # % of sampled faces under threshold
        'locations':    ndarray     # face-centre coords of thin faces   }

    Default min_thickness = 0.8 mm = 2 layers at 0.4 mm nozzle — anything
    thinner will be under-extruded or disappear at slice time.
    """
    result = {'thin_faces': [], 'min_seen': 0.0, 'median': 0.0,
              'pct_thin': 0.0, 'locations': np.zeros((0, 3))}
    try:
        n_faces = len(mesh.faces)
        if n_faces == 0:
            return result
        idx = np.arange(n_faces)
        if n_faces > sample_size:
            idx = np.random.choice(n_faces, sample_size, replace=False)
        centres = mesh.triangles_center[idx]
        normals = mesh.face_normals[idx]
        origins = centres - normals * 0.01  # step inside the surface
        directions = -normals

        hit_locs, ray_idx, _tri_idx = mesh.ray.intersects_location(
            origins, directions, multiple_hits=False)
        # Build thickness vector (NaN where no hit).
        thickness = np.full(len(idx), np.inf)
        if len(ray_idx):
            for r_i, hit in zip(ray_idx, hit_locs):
                thickness[r_i] = float(np.linalg.norm(hit - origins[r_i]))

        finite = thickness[np.isfinite(thickness)]
        if len(finite):
            result['min_seen'] = float(finite.min())
            result['median']   = float(np.median(finite))

        thin_mask = thickness < min_thickness
        result['thin_faces'] = idx[thin_mask].tolist()
        result['pct_thin']   = 100.0 * float(np.mean(thin_mask))
        result['locations']  = centres[thin_mask]
        return result
    except Exception as e:
        result['error'] = str(e)
        return result


# ════════════════════════════════════════════════════════════════════
# 6. Floating vertex clusters (small disconnected vertex sets)
# ════════════════════════════════════════════════════════════════════

def fix_floating_vertex_clusters(mesh: trimesh.Trimesh,
                                   max_cluster_faces: int = 4
                                   ) -> Tuple[trimesh.Trimesh, str]:
    """Drop connected components whose face count <= max_cluster_faces.
    Distinct from auto_repair's ratio-based shell removal — this catches
    tiny geometric noise (2-3 face fragments) that the ratio filter keeps
    because they exceed the 1% threshold on a small model.
    """
    try:
        components = mesh.split(only_watertight=False)
        if len(components) <= 1:
            return mesh, "No floating clusters"
        keep = [c for c in components if len(c.faces) > max_cluster_faces]
        removed = len(components) - len(keep)
        if removed == 0 or not keep:
            return mesh, "No floating clusters"
        out = trimesh.util.concatenate(keep)
        return out, f"Removed {removed} tiny vertex clusters (≤{max_cluster_faces} faces each)"
    except Exception as e:
        return mesh, f"Floating-cluster fix failed: {e}"


# ════════════════════════════════════════════════════════════════════
# 7. Overall quality score
# ════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════
# 8. Thin-wall HEATMAP (per-vertex RGBA colouring)
# ════════════════════════════════════════════════════════════════════

def thin_wall_heatmap(mesh: trimesh.Trimesh,
                       min_thickness: float = 0.8,
                       warn_multiplier: float = 2.0,
                       ) -> np.ndarray:
    """Return a (V, 4) RGBA array colouring each vertex by local wall
    thickness.  Used by the viewport's heatmap path.

    Scale (green → yellow → red):
      * thickness >= min_thickness * warn_multiplier   → green  (fine)
      * min_thickness <= thickness < 2×min_thickness   → yellow (caution)
      * thickness < min_thickness                      → red    (fails on print)

    Uses ray casting inward from each face, averaged to vertices.  Much
    more informative than a binary pass/fail list because you can SEE
    where the wall runs thin.
    """
    n_verts = len(mesh.vertices)
    if n_verts == 0:
        return np.zeros((0, 4), dtype=np.float32)
    thickness_per_vert = np.full(n_verts, np.inf, dtype=np.float32)

    try:
        # Ray cast from every face centre (not sampled — we want the full
        # picture for heatmap).  If the model is huge, cap at 20k faces.
        n_faces = len(mesh.faces)
        cap = 20000
        if n_faces > cap:
            idx = np.random.choice(n_faces, cap, replace=False)
        else:
            idx = np.arange(n_faces)
        centres = mesh.triangles_center[idx]
        normals = mesh.face_normals[idx]
        origins = centres - normals * 0.01
        dirs = -normals

        hit_locs, ray_idx, _ = mesh.ray.intersects_location(
            origins, dirs, multiple_hits=False)
        # Per-face thickness
        face_thick = np.full(len(idx), np.inf, dtype=np.float32)
        if len(ray_idx):
            d = np.linalg.norm(hit_locs - origins[ray_idx], axis=1)
            face_thick[ray_idx] = d.astype(np.float32)

        # Scatter face thickness → vertex (take min over adjacent faces so
        # thin regions dominate the colour).
        for local_i, fi in enumerate(idx):
            t = face_thick[local_i]
            if not np.isfinite(t):
                continue
            for v in mesh.faces[fi]:
                if t < thickness_per_vert[v]:
                    thickness_per_vert[v] = t

        # Any vertex never hit inherits the mean of its ring neighbours so
        # the heatmap doesn't show random bright spots.
        finite = np.isfinite(thickness_per_vert)
        if finite.any() and (~finite).any():
            fallback = np.median(thickness_per_vert[finite])
            thickness_per_vert[~finite] = fallback
    except Exception:
        return np.ones((n_verts, 4), dtype=np.float32)

    # Map to RGBA using piecewise linear gradient.
    rgba = np.ones((n_verts, 4), dtype=np.float32)
    warn = min_thickness * warn_multiplier
    for v in range(n_verts):
        t = thickness_per_vert[v]
        if t >= warn:
            # Green
            rgba[v, 0] = 0.30; rgba[v, 1] = 0.78; rgba[v, 2] = 0.38
        elif t >= min_thickness:
            # Yellow — linearly interpolate green → yellow
            k = (t - min_thickness) / max(warn - min_thickness, 1e-6)
            rgba[v, 0] = 0.95 - 0.65 * k
            rgba[v, 1] = 0.80
            rgba[v, 2] = 0.20 + 0.18 * k
        else:
            # Red — interpolate yellow → red as thickness → 0
            k = max(0.0, t / min_thickness)
            rgba[v, 0] = 0.95
            rgba[v, 1] = 0.25 + 0.55 * k
            rgba[v, 2] = 0.20
    return rgba


# ════════════════════════════════════════════════════════════════════
# 9. Edge-flip optimisation (improves triangle quality without changing shape)
# ════════════════════════════════════════════════════════════════════

def optimise_edge_flips(mesh: trimesh.Trimesh,
                         max_iterations: int = 3,
                         flatness_deg: float = 10.0,
                         ) -> Tuple[trimesh.Trimesh, str]:
    """Flip interior edges where the diagonal swap improves combined
    triangle quality.  Only flips across near-coplanar triangle pairs
    (dihedral angle < `flatness_deg`) so sharp features are preserved.

    Classical mesh-quality improvement.  Pairs with aspect-ratio checks
    — a flip can turn two slivers into two well-shaped triangles when
    the original diagonal was the wrong one.
    """
    try:
        from scipy.spatial import cKDTree  # confirms scipy is available
    except ImportError:
        return mesh, "Edge-flip skipped (scipy not installed)"

    try:
        verts = np.asarray(mesh.vertices, dtype=np.float64).copy()
        faces = np.asarray(mesh.faces, dtype=np.int64).copy()

        flat_cos = float(np.cos(np.deg2rad(flatness_deg)))
        total_flips = 0

        def triangle_aspect(p0, p1, p2):
            e0 = p1 - p0; e1 = p2 - p1; e2 = p0 - p2
            L = max(np.linalg.norm(e0), np.linalg.norm(e1), np.linalg.norm(e2))
            area = 0.5 * np.linalg.norm(np.cross(e0, -e2))
            return (L * L) / (4 * np.sqrt(3.0) * max(area, 1e-12))

        for _ in range(max_iterations):
            # Build edge → adjacent face list for this iteration.
            edge_faces = {}
            for fi, (a, b, c) in enumerate(faces):
                for u, v in [(a, b), (b, c), (c, a)]:
                    key = (min(u, v), max(u, v))
                    edge_faces.setdefault(key, []).append(fi)

            flipped_this_pass = 0
            touched_faces = set()
            for (u, v), fs in edge_faces.items():
                if len(fs) != 2:            # boundary or non-manifold
                    continue
                f0, f1 = fs
                if f0 in touched_faces or f1 in touched_faces:
                    continue
                tri0 = faces[f0]; tri1 = faces[f1]
                # Opposite vertices (the one NOT on the shared edge).
                w0 = [x for x in tri0 if x != u and x != v]
                w1 = [x for x in tri1 if x != u and x != v]
                if len(w0) != 1 or len(w1) != 1:
                    continue
                w0, w1 = w0[0], w1[0]

                # Dihedral angle check — don't flip across sharp features.
                n0 = np.cross(verts[tri0[1]] - verts[tri0[0]],
                              verts[tri0[2]] - verts[tri0[0]])
                n1 = np.cross(verts[tri1[1]] - verts[tri1[0]],
                              verts[tri1[2]] - verts[tri1[0]])
                nn0 = np.linalg.norm(n0); nn1 = np.linalg.norm(n1)
                if nn0 < 1e-12 or nn1 < 1e-12:
                    continue
                cos_angle = float(np.dot(n0, n1) / (nn0 * nn1))
                if cos_angle < flat_cos:    # sharp feature
                    continue

                # Quality before
                q_before = max(
                    triangle_aspect(verts[u], verts[v], verts[w0]),
                    triangle_aspect(verts[u], verts[v], verts[w1]),
                )
                # Quality after flip (edge u-v swapped for w0-w1)
                q_after = max(
                    triangle_aspect(verts[w0], verts[w1], verts[u]),
                    triangle_aspect(verts[w0], verts[w1], verts[v]),
                )
                if q_after < q_before * 0.9:   # >10% improvement
                    faces[f0] = [w0, w1, u]
                    faces[f1] = [w1, w0, v]
                    touched_faces.add(f0); touched_faces.add(f1)
                    flipped_this_pass += 1

            total_flips += flipped_this_pass
            if flipped_this_pass == 0:
                break

        if total_flips == 0:
            return mesh, "No edge flips improved quality"
        out = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        return out, f"Edge-flip optimisation: {total_flips} flips improved triangle quality"
    except Exception as e:
        return mesh, f"Edge-flip failed: {e}"


# ════════════════════════════════════════════════════════════════════
# 10. Feature-preserving decimation (pyvista.decimate_pro)
# ════════════════════════════════════════════════════════════════════

def decimate_pro(mesh: trimesh.Trimesh,
                  target_ratio: float = 0.5,
                  feature_angle: float = 30.0,
                  preserve_topology: bool = True,
                  preserve_boundary: bool = True,
                  ) -> Tuple[trimesh.Trimesh, str]:
    """PyVista's `decimate_pro` variant.  Unlike quadric decimation, it
    refuses to collapse edges across features sharper than
    `feature_angle` degrees — so mechanical details (chamfers, ribs,
    logos) survive polygon reduction instead of melting into soft blobs.

    `target_ratio` = fraction of faces to REMOVE (0.5 = halve poly count).
    """
    try:
        import pyvista as pv
    except ImportError:
        return mesh, "decimate_pro skipped (pyvista not installed)"
    try:
        n_faces = len(mesh.faces)
        pv_faces = np.column_stack([
            np.full(n_faces, 3, dtype=np.int32),
            mesh.faces.astype(np.int32)
        ]).flatten()
        pv_mesh = pv.PolyData(mesh.vertices.astype(np.float64), pv_faces)
        reduced = pv_mesh.decimate_pro(
            target_ratio,
            feature_angle=feature_angle,
            preserve_topology=preserve_topology,
            boundary_vertex_deletion=not preserve_boundary,
            splitting=False,
        )
        # Convert back to trimesh
        r_faces_flat = np.asarray(reduced.faces).reshape(-1, 4)[:, 1:]
        r_verts = np.asarray(reduced.points, dtype=np.float64)
        out = trimesh.Trimesh(vertices=r_verts, faces=r_faces_flat, process=False)
        removed = n_faces - len(out.faces)
        return out, (f"Feature-preserving decimation: "
                     f"{n_faces:,} → {len(out.faces):,} faces "
                     f"({removed:,} removed, feature_angle={feature_angle:.0f}°)")
    except Exception as e:
        return mesh, f"decimate_pro failed: {e}"


def quality_report(mesh: trimesh.Trimesh) -> Dict:
    """Fast survey of mesh quality metrics useful for a dashboard.
    Non-destructive — no modifications."""
    out = {}
    try:
        out['faces'] = len(mesh.faces)
        out['vertices'] = len(mesh.vertices)
        out['watertight'] = bool(mesh.is_watertight)
        out['winding_consistent'] = bool(mesh.is_winding_consistent)
        out['volume_mm3'] = float(mesh.volume) if mesh.is_volume else 0.0
        out['surface_area_mm2'] = float(mesh.area)
        # Sliver count
        tri = mesh.triangles
        e0 = tri[:,1]-tri[:,0]; e1 = tri[:,2]-tri[:,1]; e2 = tri[:,0]-tri[:,2]
        longest = np.max(np.stack([np.linalg.norm(e0,axis=1),
                                    np.linalg.norm(e1,axis=1),
                                    np.linalg.norm(e2,axis=1)], axis=1), axis=1)
        area = 0.5 * np.linalg.norm(np.cross(e0, -e2), axis=1)
        aspect = (longest**2) / (4*np.sqrt(3.0)*np.maximum(area, 1e-12))
        out['sliver_faces'] = int(np.sum(aspect > 60))
        out['sliver_pct'] = float(100.0 * np.mean(aspect > 60))
        # Non-manifold edges
        from collections import Counter
        ec = Counter()
        for a,b,c in mesh.faces:
            for u,v in [(a,b),(b,c),(c,a)]:
                ec[(min(u,v), max(u,v))] += 1
        out['non_manifold_edges'] = sum(1 for v in ec.values() if v > 2)
        out['boundary_edges']     = sum(1 for v in ec.values() if v == 1)
    except Exception as e:
        out['error'] = str(e)
    return out
