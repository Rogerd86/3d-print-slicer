"""
connector_shapes.py
Generate connector geometry for assembly alignment.

Shapes:
  - D-Shape: half-round, prevents rotation (parts only go one way)
  - Pyramid: tapered square peg, self-centering
  - Terrace: stepped profile, large bonding area + self-alignment
  - Square: square hole/peg, prevents rotation (simpler than D-shape)

All functions return a trimesh mesh positioned at origin, aligned along +Z.
Caller must rotate and translate to the correct face position.
"""
import numpy as np
import trimesh
from typing import Optional


def make_d_shape(radius: float = 5.0, depth: float = 15.0,
                  tolerance: float = 0.0, sections: int = 24) -> trimesh.Trimesh:
    """
    D-shaped connector (half-round).
    Flat side prevents rotation — parts only assemble one way.
    """
    # Build a full cylinder, then cut it in half
    cyl = trimesh.creation.cylinder(
        radius=radius + tolerance, height=depth, sections=sections)
    # Slice off one side to make D-shape (cut along Y=0 plane)
    from trimesh.intersections import slice_mesh_plane
    d_shape = slice_mesh_plane(cyl, np.array([0, 1, 0]), np.array([0, 0, 0]), cap=True)
    if d_shape is None or len(d_shape.faces) == 0:
        return cyl  # fallback to full cylinder
    return d_shape


def make_pyramid(base_size: float = 8.0, depth: float = 12.0,
                  taper: float = 0.7, tolerance: float = 0.0) -> trimesh.Trimesh:
    """
    Pyramid/tapered square peg — self-centering during assembly.
    taper: ratio of top to bottom (0.7 = 70% size at tip).
    """
    bs = base_size + tolerance
    ts = bs * taper  # top (inner) size

    # 8 vertices: 4 at base, 4 at top
    verts = np.array([
        # Base (z=0)
        [-bs/2, -bs/2, 0], [bs/2, -bs/2, 0],
        [bs/2, bs/2, 0], [-bs/2, bs/2, 0],
        # Top (z=depth)
        [-ts/2, -ts/2, depth], [ts/2, -ts/2, depth],
        [ts/2, ts/2, depth], [-ts/2, ts/2, depth],
    ], dtype=np.float64)

    # 12 triangular faces (2 per side + 2 for top + 2 for bottom)
    faces = np.array([
        # Bottom
        [0, 2, 1], [0, 3, 2],
        # Top
        [4, 5, 6], [4, 6, 7],
        # Front
        [0, 1, 5], [0, 5, 4],
        # Right
        [1, 2, 6], [1, 6, 5],
        # Back
        [2, 3, 7], [2, 7, 6],
        # Left
        [3, 0, 4], [3, 4, 7],
    ])

    return trimesh.Trimesh(vertices=verts, faces=faces)


def make_terrace(width: float = 10.0, depth: float = 12.0,
                  steps: int = 3, tolerance: float = 0.0) -> trimesh.Trimesh:
    """
    Terrace/stepped connector — stair-step profile.
    Each step provides a shelf for alignment + bonding area.
    """
    w = width + tolerance
    step_depth = depth / steps
    step_width = w / steps

    all_boxes = []
    for i in range(steps):
        # Each step is smaller than the last (pyramid of steps)
        current_w = w - i * step_width
        if current_w < 2.0:
            current_w = 2.0
        z_start = i * step_depth
        box = trimesh.creation.box([current_w, current_w, step_depth])
        box.apply_translation([0, 0, z_start + step_depth / 2])
        all_boxes.append(box)

    if not all_boxes:
        return trimesh.creation.box([w, w, depth])

    return trimesh.util.concatenate(all_boxes)


def make_square_peg(size: float = 6.0, depth: float = 12.0,
                     tolerance: float = 0.0) -> trimesh.Trimesh:
    """
    Square peg/hole — prevents rotation like D-shape but simpler geometry.
    """
    s = size + tolerance
    box = trimesh.creation.box([s, s, depth])
    return box


def make_connector(shape: str, **kwargs) -> Optional[trimesh.Trimesh]:
    """
    Factory function — create any connector shape by name.

    shape: 'd_shape', 'pyramid', 'terrace', 'square'
    Returns mesh centred at origin, extending along +Z.
    """
    makers = {
        'd_shape': make_d_shape,
        'pyramid': make_pyramid,
        'terrace': make_terrace,
        'square': make_square_peg,
    }
    maker = makers.get(shape)
    if maker is None:
        print(f"Unknown connector shape: {shape}")
        return None
    try:
        return maker(**kwargs)
    except Exception as e:
        print(f"Connector creation error ({shape}): {e}")
        return None
