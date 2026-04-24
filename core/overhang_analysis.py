"""
overhang_analysis.py
Analyse per-part overhang severity to flag parts that need supports.

Overhang angle: angle between face normal and -Z (gravity direction).
Faces with overhang angle > threshold need support material.

Returns per-part analysis:
  - overhang_pct: % of surface area with overhangs
  - max_overhang_angle: steepest overhang in degrees
  - needs_support: True if overhang_pct > threshold
  - suggestion: text recommendation
"""
import numpy as np
import trimesh
from typing import Dict, List


def analyse_overhang(mesh: trimesh.Trimesh,
                      overhang_threshold: float = 45.0) -> Dict:
    """
    Analyse overhang severity for a single mesh.

    overhang_threshold: faces steeper than this (degrees from vertical)
                        are flagged as needing support. Default 45°.

    Returns dict with overhang_pct, max_angle, needs_support, suggestion.
    """
    try:
        normals = mesh.face_normals
        areas = mesh.area_faces
        total_area = max(np.sum(areas), 1e-9)

        # Gravity direction is -Z
        gravity = np.array([0, 0, -1.0])

        # Dot product of each face normal with gravity
        # Positive dot = face points downward (potential overhang)
        dots = np.dot(normals, gravity)

        # Overhang angle: angle between face normal and -Z
        # A face pointing straight down has dot=1.0 → angle=0° (worst overhang)
        # A face pointing straight up has dot=-1.0 → angle=180° (no overhang)
        # A face at 45° from vertical: dot ≈ 0.7 → angle ≈ 45°

        # Convert to overhang angle (degrees from horizontal downward)
        angles = np.degrees(np.arccos(np.clip(dots, -1, 1)))
        # angles: 0° = face points straight down, 90° = face is vertical, 180° = face points up

        # Overhanging faces: angle < (90 - overhang_threshold) means steep downward overhang
        # Or equivalently: the face normal points more than threshold degrees below horizontal
        overhang_mask = angles < (90 - overhang_threshold)

        overhang_area = np.sum(areas[overhang_mask])
        overhang_pct = (overhang_area / total_area) * 100.0

        max_angle = float(90 - np.min(angles)) if len(angles) > 0 else 0.0
        max_angle = max(0, max_angle)

        needs_support = overhang_pct > 5.0  # >5% overhang area

        if needs_support:
            if overhang_pct > 30:
                suggestion = "Heavy supports needed — consider rotating this part"
            elif overhang_pct > 15:
                suggestion = "Moderate supports — try tree supports"
            else:
                suggestion = "Light supports — auto supports should work"
        else:
            suggestion = "Prints clean — no supports needed"

        return {
            'overhang_pct': round(overhang_pct, 1),
            'max_angle': round(max_angle, 1),
            'needs_support': needs_support,
            'suggestion': suggestion,
            'overhang_area_mm2': round(float(overhang_area), 1),
            'total_area_mm2': round(float(total_area), 1),
        }

    except Exception as e:
        return {
            'overhang_pct': 0,
            'max_angle': 0,
            'needs_support': False,
            'suggestion': f"Analysis error: {e}",
            'overhang_area_mm2': 0,
            'total_area_mm2': 0,
        }


def analyse_all_parts(parts, overhang_threshold: float = 45.0) -> List[Dict]:
    """Analyse overhang for all parts. Returns list of analysis dicts."""
    results = []
    for part in parts:
        result = analyse_overhang(part.mesh, overhang_threshold)
        result['label'] = part.label
        results.append(result)
    return results


def overhang_summary(analyses: List[Dict]) -> Dict:
    """Summary across all parts."""
    n_support = sum(1 for a in analyses if a['needs_support'])
    avg_pct = np.mean([a['overhang_pct'] for a in analyses]) if analyses else 0
    return {
        'total_parts': len(analyses),
        'parts_needing_support': n_support,
        'avg_overhang_pct': round(float(avg_pct), 1),
    }
