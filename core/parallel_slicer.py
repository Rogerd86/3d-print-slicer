"""
parallel_slicer.py
Multi-threaded slicing — runs mesh cuts in parallel using ProcessPoolExecutor.
Falls back to sequential if parallel fails.

Key insight: trimesh slice_mesh_plane is CPU-bound and releases the GIL,
so ThreadPoolExecutor gives real speedup. ProcessPoolExecutor is used for
the full auto-slice to avoid memory issues with large meshes.
"""
import numpy as np
import trimesh
from trimesh.intersections import slice_mesh_plane
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Callable, Optional
import threading


def parallel_slice_all(source_mesh: trimesh.Trimesh,
                        cut_definitions: list,
                        progress_cb: Optional[Callable] = None,
                        max_workers: int = 4) -> List[trimesh.Trimesh]:
    """
    Slice a mesh with multiple independent cuts in parallel.
    Each cut is independent — slices the same source mesh.
    Returns list of result meshes (one per cut).
    
    progress_cb: called with (completed, total) after each cut finishes.
    """
    results = [None] * len(cut_definitions)
    total = len(cut_definitions)
    completed = 0
    lock = threading.Lock()

    def do_cut(idx, cut):
        normal = cut.get_normal()
        origin = cut.get_origin()
        try:
            result = slice_mesh_plane(source_mesh.copy(), normal, origin, cap=True)
            return idx, result
        except Exception as e:
            print(f"Parallel cut {idx} error: {e}")
            return idx, None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(do_cut, i, cut): i
                   for i, cut in enumerate(cut_definitions)}
        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result
            with lock:
                completed += 1
                if progress_cb:
                    progress_cb(completed, total)

    return results


def parallel_auto_slice(root_mesh: trimesh.Trimesh,
                         cut_size: float,
                         progress_cb: Optional[Callable] = None,
                         max_workers: int = 4) -> List[trimesh.Trimesh]:
    """
    Auto-slice a mesh into a grid of cut_size pieces.
    Cuts each axis repeatedly until every piece fits within cut_size.
    Parallelises across parts within each axis pass.
    """
    current_parts = [root_mesh.copy()]
    axes_indices = [0, 1, 2]

    for ax_idx in axes_indices:
        # Keep slicing this axis until all parts fit
        changed = True
        max_passes = 50
        passes = 0
        while changed and passes < max_passes:
            changed = False
            passes += 1
            next_parts = []
            total_this_pass = len(current_parts)

            def slice_one_part(part):
                lo = float(part.bounds[0][ax_idx])
                hi = float(part.bounds[1][ax_idx])
                if (hi - lo) <= cut_size * 1.05:
                    return [part]  # already fits

                cut_pos = lo + cut_size
                normal = np.zeros(3); normal[ax_idx] = 1.0
                origin = np.zeros(3); origin[ax_idx] = cut_pos
                try:
                    low  = slice_mesh_plane(part, normal, origin, cap=True)
                    high = slice_mesh_plane(part, -normal, origin, cap=True)
                    result = []
                    if low  is not None and len(low.faces)  > 0: result.append(low)
                    if high is not None and len(high.faces) > 0: result.append(high)
                    return result if result else [part]
                except Exception as e:
                    print(f"Parallel cut error: {e}")
                    return [part]

            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(slice_one_part, p): i
                           for i, p in enumerate(current_parts)}
                for fut in as_completed(futures):
                    result_list = fut.result()
                    next_parts.extend(result_list)
                    if len(result_list) > 1:
                        changed = True

            current_parts = next_parts
            if progress_cb:
                progress_cb(ax_idx * 10 + passes, 30)

    if progress_cb:
        progress_cb(30, 30)

    return [p for p in current_parts if p is not None and len(p.faces) > 0]


def parallel_export_parts(parts, export_fn: Callable, max_workers: int = 4,
                           progress_cb: Optional[Callable] = None):
    """
    Export multiple parts in parallel.
    export_fn(part, index) -> filepath string
    """
    results = {}
    total = len(parts)
    completed = 0
    lock = threading.Lock()

    def do_export(idx, part):
        try:
            path = export_fn(part, idx)
            return idx, path, None
        except Exception as e:
            return idx, None, str(e)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(do_export, i, p): i for i, p in enumerate(parts)}
        for future in as_completed(futures):
            idx, path, err = future.result()
            results[idx] = path
            with lock:
                completed += 1
                if progress_cb:
                    progress_cb(completed, total)

    return [results.get(i) for i in range(total)]
