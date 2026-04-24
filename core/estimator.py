"""
Print time and filament estimator for sliced parts.
Rough estimates based on part volume, wall count, and printer speed.
Not a substitute for a real slicer estimate, but useful for planning.
"""
import numpy as np
import trimesh
from typing import List, Dict
from core.slicer import SlicedPart


# Filament density g/cm³
FILAMENT_DENSITY = {
    'PLA': 1.24,
    'PETG': 1.27,
    'ABS': 1.05,
    'ASA': 1.07,
    'TPU': 1.21,
}

# Bambu Lab P2S typical speeds mm/s
P2S_OUTER_WALL_SPEED = 100
P2S_INNER_WALL_SPEED = 250
P2S_INFILL_SPEED = 300
P2S_TRAVEL_SPEED = 500


def estimate_part(
    part: SlicedPart,
    wall_thickness_mm: float = 3.0,
    infill_pct: float = 15.0,
    layer_height_mm: float = 0.2,
    material: str = 'PETG'
) -> Dict:
    """
    Estimate print time and filament for a single part.
    Returns dict with time_minutes, filament_g, filament_m.
    """
    mesh = part.mesh
    extents = mesh.extents  # x, y, z mm

    try:
        # --- Volume estimates ---
        # Shell volume: surface area * wall thickness
        surface_area_mm2 = mesh.area
        shell_volume_mm3 = surface_area_mm2 * wall_thickness_mm

        # Infill volume: bounding box interior minus shell
        bbox_volume_mm3 = float(np.prod(extents))
        interior_volume_mm3 = max(0, bbox_volume_mm3 - shell_volume_mm3)
        infill_volume_mm3 = interior_volume_mm3 * (infill_pct / 100.0)

        total_printed_mm3 = shell_volume_mm3 + infill_volume_mm3

        # Convert to filament length (1.75mm diameter filament)
        filament_radius_mm = 0.875
        filament_area_mm2 = np.pi * filament_radius_mm ** 2
        filament_length_mm = total_printed_mm3 / filament_area_mm2
        filament_length_m = filament_length_mm / 1000.0

        density = FILAMENT_DENSITY.get(material, 1.24)
        filament_g = (total_printed_mm3 / 1000.0) * density  # cm³ * g/cm³

        # --- Time estimate ---
        # Number of layers
        n_layers = int(np.ceil(extents[2] / layer_height_mm))

        # Perimeter length per layer (rough: 2 * perimeter of bounding box cross section)
        perimeter_per_layer_mm = 2 * (extents[0] + extents[1])

        # Wall passes (number of walls based on thickness)
        n_walls = max(1, int(wall_thickness_mm / (layer_height_mm * 2)))
        outer_wall_lines = n_layers * perimeter_per_layer_mm
        inner_wall_lines = n_layers * perimeter_per_layer_mm * max(0, n_walls - 1)

        # Infill lines per layer
        line_spacing_mm = layer_height_mm * 2
        infill_area_mm2 = extents[0] * extents[1]
        infill_lines_per_layer = (infill_area_mm2 * infill_pct / 100.0) / line_spacing_mm
        total_infill_mm = infill_lines_per_layer * n_layers

        # Time calculations
        t_outer = outer_wall_lines / P2S_OUTER_WALL_SPEED
        t_inner = inner_wall_lines / P2S_INNER_WALL_SPEED
        t_infill = total_infill_mm / P2S_INFILL_SPEED

        # Add 20% overhead for travel, Z moves, etc.
        total_seconds = (t_outer + t_inner + t_infill) * 1.2
        total_minutes = total_seconds / 60.0

        # Weight = filament grams (already computed)
        weight_g = round(filament_g, 1)
        # Centroid for assembly CoG
        try:
            centroid = mesh.centroid.tolist()
        except Exception:
            centroid = [0, 0, 0]

        return {
            'label': part.label,
            'dimensions_mm': (round(float(extents[0]), 1),
                              round(float(extents[1]), 1),
                              round(float(extents[2]), 1)),
            'time_minutes': round(total_minutes, 1),
            'time_str': _format_time(total_minutes),
            'filament_g': round(filament_g, 1),
            'filament_m': round(filament_length_m, 2),
            'weight_g': weight_g,
            'centroid': centroid,
            'n_layers': n_layers,
            'orientation_hint': _orientation_hint(extents),
        }

    except Exception as e:
        return {
            'label': part.label,
            'dimensions_mm': (0, 0, 0),
            'time_minutes': 0,
            'time_str': 'N/A',
            'filament_g': 0,
            'filament_m': 0,
            'weight_g': 0,
            'centroid': [0, 0, 0],
            'n_layers': 0,
            'orientation_hint': '',
        }


def estimate_all(
    parts: List[SlicedPart],
    wall_thickness_mm: float = 3.0,
    infill_pct: float = 15.0,
    layer_height_mm: float = 0.2,
    material: str = 'PETG'
) -> List[Dict]:
    return [estimate_part(p, wall_thickness_mm, infill_pct, layer_height_mm, material)
            for p in parts]


def total_summary(estimates: List[Dict]) -> Dict:
    total_time = sum(e['time_minutes'] for e in estimates)
    total_g = sum(e['filament_g'] for e in estimates)
    total_m = sum(e['filament_m'] for e in estimates)
    total_weight = sum(e.get('weight_g', e['filament_g']) for e in estimates)
    parts_needing_rotation = sum(1 for e in estimates if e['orientation_hint'])

    # Compute assembly centre of gravity (weighted average of centroids)
    cog = [0.0, 0.0, 0.0]
    if total_weight > 0:
        for e in estimates:
            w = e.get('weight_g', 0)
            c = e.get('centroid', [0, 0, 0])
            for i in range(3):
                cog[i] += w * c[i]
        cog = [round(c / total_weight, 1) for c in cog]

    return {
        'total_time_str': _format_time(total_time),
        'total_filament_g': round(total_g, 1),
        'total_filament_m': round(total_m, 2),
        'total_filament_spools': round(total_g / 1000.0, 2),
        'total_weight_g': round(total_weight, 1),
        'centre_of_gravity': cog,
        'parts_needing_rotation': parts_needing_rotation,
        'part_count': len(estimates),
    }


def _orientation_hint(extents) -> str:
    """
    Suggest rotation if the part would print better in a different orientation.
    A part is flagged if its Z (print height) is much larger than X or Y,
    meaning it's tall and thin — lay it flat for better strength.
    """
    x, y, z = extents
    if z > 0 and (z / max(x, y, 1)) > 2.5:
        # Very tall relative to footprint — suggest rotating
        if x >= y:
            return "Rotate: lay flat on Y axis (−90° X)"
        else:
            return "Rotate: lay flat on X axis (−90° Y)"
    return ""


def _format_time(minutes: float) -> str:
    if minutes < 1:
        return "< 1 min"
    h = int(minutes // 60)
    m = int(minutes % 60)
    if h == 0:
        return f"{m}m"
    return f"{h}h {m}m"
