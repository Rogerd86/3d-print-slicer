"""
grid_numbering.py
Smart spatial numbering for sliced parts.

Instead of sequential numbering (001, 002, 003...), assigns numbers based
on spatial position: left→right, front→back, bottom→top.

This makes assembly intuitive — part 1 is always bottom-left-front,
and numbers increase predictably so you know where each piece goes.
"""
import numpy as np
from typing import List


def renumber_parts_spatially(parts, base_label: str = "Body") -> List[str]:
    """
    Renumber parts based on their spatial position.

    Ordering: X (left→right) → Y (front→back) → Z (bottom→top)
    Parts are sorted by Z first (layers), then Y, then X within each layer.

    Returns list of new labels in the same order as input parts.
    """
    if not parts:
        return []

    # Get centroid of each part
    centroids = []
    for p in parts:
        try:
            c = (p.mesh.bounds[0] + p.mesh.bounds[1]) / 2.0
            centroids.append(c)
        except Exception:
            centroids.append(np.zeros(3))

    centroids = np.array(centroids)

    # Sort by Z (bottom→top), then Y (front→back), then X (left→right)
    # Use rounding to group parts in the same layer/row
    z_vals = centroids[:, 2]
    y_vals = centroids[:, 1]
    x_vals = centroids[:, 0]

    # Determine grid resolution (round to nearest 10mm to group layers)
    z_range = z_vals.max() - z_vals.min() if len(z_vals) > 1 else 1
    y_range = y_vals.max() - y_vals.min() if len(y_vals) > 1 else 1
    x_range = x_vals.max() - x_vals.min() if len(x_vals) > 1 else 1

    # Use 5% of range as bucket size (groups nearby parts into same layer)
    z_bucket = max(1, z_range * 0.05)
    y_bucket = max(1, y_range * 0.05)
    x_bucket = max(1, x_range * 0.05)

    # Create sort key: (z_bucket, y_bucket, x_bucket) for each part
    sort_keys = []
    for i in range(len(parts)):
        zk = int(z_vals[i] / z_bucket)
        yk = int(y_vals[i] / y_bucket)
        xk = int(x_vals[i] / x_bucket)
        sort_keys.append((zk, yk, xk, i))

    # Sort and assign numbers
    sort_keys.sort()
    labels = [""] * len(parts)
    for rank, (_, _, _, orig_idx) in enumerate(sort_keys):
        labels[orig_idx] = f"{base_label}-{rank + 1:03d}"

    return labels


def get_position_description(part) -> str:
    """
    Get a human-readable position description for a part.
    e.g., "bottom-left-front", "top-right-back"
    """
    try:
        c = (part.mesh.bounds[0] + part.mesh.bounds[1]) / 2.0
        # These will be relative — caller should provide model bounds
        return f"({c[0]:.0f}, {c[1]:.0f}, {c[2]:.0f})"
    except Exception:
        return "(?)"


def get_assembly_order(parts) -> List[dict]:
    """
    Generate assembly order with spatial descriptions.
    Returns list of {part, label, position, layer, description}
    sorted in recommended assembly order (bottom-up, inside-out).
    """
    if not parts:
        return []

    centroids = []
    for p in parts:
        try:
            c = (p.mesh.bounds[0] + p.mesh.bounds[1]) / 2.0
            centroids.append(c)
        except Exception:
            centroids.append(np.zeros(3))

    centroids = np.array(centroids)

    # Determine spatial regions
    bounds_min = centroids.min(axis=0)
    bounds_max = centroids.max(axis=0)
    mid = (bounds_min + bounds_max) / 2.0

    order = []
    for i, p in enumerate(parts):
        c = centroids[i]
        # Position words
        x_pos = "left" if c[0] < mid[0] else "right"
        y_pos = "front" if c[1] < mid[1] else "back"
        z_pos = "bottom" if c[2] < mid[2] else "top"

        # Layer number (z-axis grouping)
        z_range = max(1, bounds_max[2] - bounds_min[2])
        layer = int((c[2] - bounds_min[2]) / z_range * 3) + 1  # 1-3

        order.append({
            'part': p,
            'label': p.label,
            'position': f"{z_pos}-{x_pos}-{y_pos}",
            'layer': layer,
            'centroid': c,
            'description': f"Layer {layer}, {z_pos} {x_pos} {y_pos}",
        })

    # Sort: bottom layers first, then front-to-back, then left-to-right
    order.sort(key=lambda o: (o['centroid'][2], o['centroid'][1], o['centroid'][0]))
    return order
