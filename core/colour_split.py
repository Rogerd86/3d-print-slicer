"""
colour_split.py — detect colour/material regions and split into separate parts.
"""
import numpy as np
import trimesh
from typing import List, Tuple, Optional

def detect_colour_regions(mesh, tolerance=20):
    try:
        fc = mesh.visual.face_colors
        if fc is None or len(fc) == 0:
            return []
        unique = np.unique(fc[:, :3], axis=0)
        if len(unique) <= 1:
            return []
        binned = (fc[:, :3] // tolerance) * tolerance
        unique_binned = np.unique(binned, axis=0)
        if len(unique_binned) <= 1:
            return []
        regions = []
        for uc in unique_binned:
            mask = np.all(binned == uc, axis=1)
            face_indices = np.where(mask)[0]
            if len(face_indices) < 4:
                continue
            avg_colour = fc[face_indices].mean(axis=0).astype(np.uint8)
            regions.append((face_indices, avg_colour))
        return regions
    except Exception as e:
        print(f"Colour detection error: {e}")
        return []

def has_colour_data(mesh):
    try:
        fc = mesh.visual.face_colors
        if fc is None or len(fc) == 0: return False
        return len(np.unique(fc[:, :3], axis=0)) > 1
    except Exception:
        return False

def has_multi_geometry(obj):
    return isinstance(obj, trimesh.Scene) and len(obj.geometry) > 1

def split_by_colour(mesh, part_tree, base_label="Body", tolerance=20):
    regions = detect_colour_regions(mesh, tolerance)
    if len(regions) < 2:
        return False, "No distinct colour regions found."
    from core.part_tree import Part
    Part._color_counter = 0
    root = Part(mesh.copy(), base_label)
    part_tree.root = root
    part_tree.selected_part = root
    part_tree._undo_stack = []
    for i, (face_indices, avg_colour) in enumerate(regions):
        try:
            sub = mesh.submesh([face_indices], append=True)
            if sub is None or len(sub.faces) == 0: continue
            r, g, b = int(avg_colour[0]), int(avg_colour[1]), int(avg_colour[2])
            label = f"{base_label}-{_colour_name(r,g,b)}"
            child = Part(sub, label, parent=root)
            child._source_colour = (r/255, g/255, b/255)
            root.children.append(child)
        except Exception as e:
            print(f"Region {i} error: {e}")
    if not root.children:
        return False, "Could not extract colour regions."
    part_tree.selected_part = root.children[0]
    names = ", ".join(c.label.split('-')[-1] for c in root.children[:5])
    extra = f" +{len(root.children)-5} more" if len(root.children) > 5 else ""
    return True, f"Split into {len(root.children)} colour regions: {names}{extra}"

def split_scene_by_geometry(scene, part_tree, base_label="Body"):
    geometries = [(k,v) for k,v in scene.geometry.items() if isinstance(v, trimesh.Trimesh) and len(v.faces) > 0]
    if len(geometries) < 2:
        return False, "Only one geometry in scene."
    from core.part_tree import Part
    Part._color_counter = 0
    all_meshes = [m for _, m in geometries]
    combined = trimesh.util.concatenate(all_meshes)
    root = Part(combined, base_label)
    part_tree.root = root; part_tree.selected_part = root; part_tree._undo_stack = []
    for name, mesh in geometries:
        label = f"{base_label}-{name}"
        child = Part(mesh.copy(), label, parent=root)
        try:
            fc = mesh.visual.face_colors
            if fc is not None and len(fc) > 0:
                avg = fc.mean(axis=0).astype(np.uint8)
                child._source_colour = (avg[0]/255, avg[1]/255, avg[2]/255)
        except Exception:
            pass
        root.children.append(child)
    part_tree.selected_part = root.children[0] if root.children else root
    return True, f"Split into {len(root.children)} geometry groups."

def _colour_name(r, g, b):
    mx, mn = max(r,g,b), min(r,g,b)
    if mx < 40: return "Black"
    if mn > 200: return "White"
    if mx - mn < 30:
        v = (r+g+b)//3
        return "DarkGrey" if v < 100 else ("Grey" if v < 180 else "LightGrey")
    if r > g and r > b: return "Orange" if g > 120 else "Red"
    if g > r and g > b: return "Yellow" if r > 150 else "Green"
    if b > r and b > g: return "Purple" if r > g*1.2 else "Blue"
    if r > 180 and g > 180: return "Yellow"
    if r > 180 and b > 180: return "Magenta"
    if g > 180 and b > 180: return "Cyan"
    return f"Col{r:02x}{g:02x}{b:02x}"
