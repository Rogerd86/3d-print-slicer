"""
Project save/load: serialise the full 3D Print Slicer session to JSON.
Saves: source file path, resize dimensions, all cut planes + lock states,
       build plate settings, joint settings, hollow shell settings.
"""
import json
import os
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.slicer import Slicer
    from core.mesh_handler import MeshHandler

PROJECT_VERSION = "1.1"


def save_project(path: str, mesh_handler, slicer, settings: dict) -> bool:
    """
    Save the current session to a .ksproject JSON file.

    settings dict should contain:
        wall_thickness, infill_pct, layer_height, material,
        joint_type, joint_size, tolerance, export_format
    """
    try:
        data = {
            'version': PROJECT_VERSION,
            'source_file': mesh_handler.file_path or '',
            'resize': {
                'x': mesh_handler.get_dimensions_mm()[0],
                'y': mesh_handler.get_dimensions_mm()[1],
                'z': mesh_handler.get_dimensions_mm()[2],
                'scale_factor': mesh_handler.scale_factor,
            },
            'build_plate': {
                'x': slicer.build_plate_x,
                'y': slicer.build_plate_y,
                'z': slicer.build_plate_z,
            },
            'default_cut_size': slicer.default_cut_size,
            'cut_planes': [
                {
                    'axis': p.axis,
                    'auto_position': p.auto_position,
                    'manual_position': p.manual_position,
                    'pinned': p.pinned,
                }
                for p in slicer.cut_planes
            ],
            'settings': settings,
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        return True

    except Exception as e:
        print(f"Save error: {e}")
        return False


def load_project(path: str) -> Optional[dict]:
    """
    Load a .ksproject file. Returns the raw dict for the caller to apply.
    Returns None on failure.
    """
    try:
        with open(path, 'r') as f:
            data = json.load(f)

        # Version compatibility check
        ver = data.get('version', '1.0')
        if ver not in ('1.0', '1.1'):
            print(f"Warning: project version {ver} may not be fully compatible")

        return data
    except Exception as e:
        print(f"Load error: {e}")
        return None


def apply_project(data: dict, mesh_handler, slicer) -> dict:
    """
    Apply loaded project data to mesh_handler and slicer.
    Returns the settings dict for the UI to apply.

    Note: does NOT reload the mesh — that must be done by the caller
    since the mesh file needs to exist on disk.
    """
    from core.slicer import CutPlane

    # Build plate
    bp = data.get('build_plate', {})
    slicer.build_plate_x = bp.get('x', 256.0)
    slicer.build_plate_y = bp.get('y', 256.0)
    slicer.build_plate_z = bp.get('z', 256.0)
    slicer.default_cut_size = data.get('default_cut_size', 150.0)

    # Restore cut planes with lock states
    slicer.cut_planes = []
    for cp_data in data.get('cut_planes', []):
        plane = CutPlane(cp_data['axis'], cp_data['auto_position'])
        plane.manual_position = cp_data.get('manual_position')
        plane.pinned = cp_data.get('pinned', False)
        slicer.cut_planes.append(plane)

    return data.get('settings', {})
