"""
tolerance_test.py
Generate a small test block with the exact joint geometry at multiple tolerance
values so the user can find the right fit before printing all parts.

Exports a single STL with multiple test blocks side by side, each labelled
with its tolerance value.
"""
import numpy as np
import trimesh
import os
from typing import Optional
from core.boolean_ops import boolean_difference


def generate_tolerance_test(joint_type: str = 'round_dowel',
                             base_radius: float = 5.0,
                             rect_width: float = 3.0,
                             rect_height: float = 12.0,
                             magnet_diameter: float = 6.0,
                             magnet_depth: float = 3.2,
                             tolerances: list = None,
                             block_size: float = 30.0,
                             block_depth: float = 15.0) -> Optional[trimesh.Trimesh]:
    """
    Generate a tolerance test print.

    Creates a row of blocks, each with a hole/slot at a different tolerance.
    Also creates matching male test pieces (pins/keys).

    joint_type: 'round_dowel', 'rect_dowel', 'magnet', 'snap_fit'
    tolerances: list of mm values to test. Default: [0.1, 0.15, 0.2, 0.25, 0.3, 0.4]

    Returns a single mesh with all test blocks combined.
    """
    if tolerances is None:
        tolerances = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]

    all_parts = []
    spacing = block_size + 5  # gap between blocks

    for i, tol in enumerate(tolerances):
        x_offset = i * spacing

        # Base block
        block = trimesh.creation.box([block_size, block_size, block_depth])
        block.apply_translation([x_offset, 0, block_depth / 2])

        if joint_type == 'round_dowel':
            # Cylindrical hole at the given tolerance
            r = base_radius + tol
            hole = trimesh.creation.cylinder(radius=r, height=block_depth + 2, sections=32)
            hole.apply_translation([x_offset, 0, block_depth / 2])
            try:
                result = boolean_difference([block, hole])
                if result is not None and len(result.faces) > 0:
                    block = result
            except Exception:
                pass  # fallback: just the block without the hole

        elif joint_type == 'rect_dowel':
            w = rect_width + tol
            h = rect_height + tol
            slot = trimesh.creation.box([w, h, block_depth + 2])
            slot.apply_translation([x_offset, 0, block_depth / 2])
            try:
                result = boolean_difference([block, slot])
                if result is not None and len(result.faces) > 0:
                    block = result
            except Exception:
                pass

        elif joint_type == 'magnet':
            r = (magnet_diameter / 2) + tol
            depth = magnet_depth + 0.2  # slight extra depth
            pocket = trimesh.creation.cylinder(radius=r, height=depth, sections=32)
            pocket.apply_translation([x_offset, 0, depth / 2])
            try:
                result = boolean_difference([block, pocket])
                if result is not None and len(result.faces) > 0:
                    block = result
            except Exception:
                pass

        elif joint_type == 'snap_fit':
            # Rectangular slot for a snap clip
            w = 5 + tol
            h = 8 + tol
            slot = trimesh.creation.box([w, h, block_depth + 2])
            slot.apply_translation([x_offset, 0, block_depth / 2])
            try:
                result = boolean_difference([block, slot])
                if result is not None and len(result.faces) > 0:
                    block = result
            except Exception:
                pass

        # Add tolerance label as a small embossed text
        # Simple approach: a thin raised bar whose width encodes the tolerance
        bar_w = tol * 50  # visual indicator
        label_bar = trimesh.creation.box([max(1, bar_w), block_size * 0.8, 0.8])
        label_bar.apply_translation([x_offset, 0, block_depth + 0.4])
        block = trimesh.util.concatenate([block, label_bar])

        all_parts.append(block)

    if not all_parts:
        return None

    # Combine all blocks
    combined = trimesh.util.concatenate(all_parts)

    # Also add matching male test pieces (pins/keys) below the blocks
    male_parts = []
    for i, tol in enumerate(tolerances):
        x_offset = i * spacing
        y_offset = -block_size - 10  # below the blocks

        if joint_type == 'round_dowel':
            pin = trimesh.creation.cylinder(
                radius=base_radius, height=block_depth * 0.8, sections=32)
            pin.apply_translation([x_offset, y_offset, block_depth * 0.4])
            male_parts.append(pin)

        elif joint_type == 'rect_dowel':
            key = trimesh.creation.box([rect_width, rect_height, block_depth * 0.8])
            key.apply_translation([x_offset, y_offset, block_depth * 0.4])
            male_parts.append(key)

        elif joint_type == 'magnet':
            # No male piece needed — magnets are the male piece
            pass

        elif joint_type == 'snap_fit':
            clip = trimesh.creation.box([5, 8, block_depth * 0.8])
            clip.apply_translation([x_offset, y_offset, block_depth * 0.4])
            male_parts.append(clip)

    if male_parts:
        combined = trimesh.util.concatenate([combined] + male_parts)

    return combined


def export_tolerance_test(output_dir: str,
                           joint_type: str = 'round_dowel',
                           base_radius: float = 5.0,
                           rect_width: float = 3.0,
                           rect_height: float = 12.0,
                           magnet_diameter: float = 6.0,
                           magnet_depth: float = 3.2,
                           tolerances: list = None) -> Optional[str]:
    """
    Generate and export a tolerance test STL.
    Returns the file path or None on failure.
    """
    mesh = generate_tolerance_test(
        joint_type=joint_type,
        base_radius=base_radius,
        rect_width=rect_width,
        rect_height=rect_height,
        magnet_diameter=magnet_diameter,
        magnet_depth=magnet_depth,
        tolerances=tolerances)

    if mesh is None:
        return None

    type_name = joint_type.replace('_', '-')
    tol_str = "-".join(f"{t:.2f}" for t in (tolerances or [0.1,0.15,0.2,0.25,0.3,0.4]))
    filename = f"tolerance_test_{type_name}.stl"
    filepath = os.path.join(output_dir, filename)

    try:
        os.makedirs(output_dir, exist_ok=True)
        mesh.export(filepath, file_type='stl')

        # Write info file
        info_path = os.path.join(output_dir, f"tolerance_test_{type_name}_info.txt")
        with open(info_path, 'w') as f:
            f.write(f"3D Print Slicer Tolerance Test — {joint_type}\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"Joint type: {joint_type}\n")
            if joint_type == 'round_dowel':
                f.write(f"Base radius: {base_radius}mm\n")
            elif joint_type == 'rect_dowel':
                f.write(f"Slot: {rect_width}×{rect_height}mm\n")
            elif joint_type == 'magnet':
                f.write(f"Magnet: {magnet_diameter}mm dia × {magnet_depth}mm deep\n")
            f.write(f"\nTest blocks (left to right):\n")
            for i, tol in enumerate(tolerances or [0.1,0.15,0.2,0.25,0.3,0.4]):
                f.write(f"  Block {i+1}: tolerance = {tol:.2f}mm\n")
            f.write(f"\nInstructions:\n")
            f.write(f"  1. Print this test piece\n")
            f.write(f"  2. Try fitting the pin/key into each hole\n")
            f.write(f"  3. Find the tightest fit that still goes in smoothly\n")
            f.write(f"  4. Use that tolerance value in 3D Print Slicer\n")

        return filepath
    except Exception as e:
        print(f"Tolerance test export error: {e}")
        return None
