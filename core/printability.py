"""
printability.py
Combined printability check — runs all analyses on each part and
produces a single pass/warn/fail report.

Checks:
  - Exceeds build volume
  - Thin walls
  - Excessive overhangs
  - Too small to be useful
  - Weak bond surfaces
"""
import numpy as np
from typing import Dict, List, Tuple


def check_printability(part,
                        build_volume: Tuple[float, float, float] = (256, 256, 256),
                        min_wall: float = 1.5,
                        min_dimension: float = 5.0) -> Dict:
    """
    Run all printability checks on a single part.

    Returns dict with:
      label: part name
      issues: list of {severity, check, message}
      score: 0-100 overall printability score
      colour: 'green', 'yellow', or 'red'
    """
    issues = []
    mesh = part.mesh

    # 1. Build volume check
    extents = mesh.extents
    for i, (dim, limit, axis) in enumerate(zip(extents, build_volume, ['X', 'Y', 'Z'])):
        if dim > limit:
            issues.append({
                'severity': 'error',
                'check': 'build_volume',
                'message': f"{axis} dimension {dim:.0f}mm exceeds build volume ({limit:.0f}mm)"
            })

    # 2. Too small check
    min_ext = float(np.min(extents))
    if min_ext < min_dimension:
        issues.append({
            'severity': 'warning',
            'check': 'too_small',
            'message': f"Smallest dimension is {min_ext:.1f}mm — may be too fragile"
        })

    # 3. Overhang check
    try:
        from core.overhang_analysis import analyse_overhang
        oh = analyse_overhang(mesh)
        if oh.get('overhang_pct', 0) > 30:
            issues.append({
                'severity': 'warning',
                'check': 'overhang',
                'message': f"{oh['overhang_pct']:.0f}% overhang — needs supports ({oh['suggestion']})"
            })
        elif oh.get('overhang_pct', 0) > 10:
            issues.append({
                'severity': 'info',
                'check': 'overhang',
                'message': f"{oh['overhang_pct']:.0f}% overhang — minor supports may help"
            })
    except Exception:
        pass

    # 4. Wall thickness check
    try:
        from core.wall_thickness import analyse_wall_thickness
        wt = analyse_wall_thickness(mesh, min_wall, n_samples=500)
        if wt.get('pct_thin', 0) > 15:
            issues.append({
                'severity': 'error',
                'check': 'thin_wall',
                'message': f"{wt['pct_thin']:.0f}% of walls < {min_wall}mm (min: {wt['min_found_mm']}mm)"
            })
        elif wt.get('pct_thin', 0) > 3:
            issues.append({
                'severity': 'warning',
                'check': 'thin_wall',
                'message': f"Some thin walls: {wt['min_found_mm']}mm — use 3+ walls in slicer"
            })
    except Exception:
        pass

    # 5. Bond surface check
    try:
        from core.bond_analysis import analyse_bond_surfaces
        bonds = analyse_bond_surfaces(mesh)
        weak = [b for b in bonds if b['bond_quality'] < 30]
        if weak:
            issues.append({
                'severity': 'warning',
                'check': 'weak_bond',
                'message': f"{len(weak)} weak bond face(s) — add dowels or reposition cut"
            })
    except Exception:
        pass

    # 6. Triangle count sanity
    n_tris = len(mesh.faces)
    if n_tris < 20:
        issues.append({
            'severity': 'error',
            'check': 'degenerate',
            'message': f"Only {n_tris} triangles — mesh may be degenerate"
        })

    # Compute score
    errors = sum(1 for i in issues if i['severity'] == 'error')
    warnings = sum(1 for i in issues if i['severity'] == 'warning')
    score = max(0, 100 - errors * 30 - warnings * 10)
    if errors > 0:
        colour = 'red'
    elif warnings > 0:
        colour = 'yellow'
    else:
        colour = 'green'

    return {
        'label': part.label,
        'issues': issues,
        'score': score,
        'colour': colour,
        'n_errors': errors,
        'n_warnings': warnings,
    }


def check_all_parts(parts,
                     build_volume=(256, 256, 256),
                     min_wall=1.5) -> List[Dict]:
    """Run printability check on all parts."""
    return [check_printability(p, build_volume, min_wall) for p in parts]


def printability_summary(results: List[Dict]) -> str:
    """Human-readable summary."""
    reds = sum(1 for r in results if r['colour'] == 'red')
    yellows = sum(1 for r in results if r['colour'] == 'yellow')
    greens = sum(1 for r in results if r['colour'] == 'green')
    lines = []
    if reds:
        labels = ", ".join(r['label'] for r in results if r['colour'] == 'red')[:60]
        lines.append(f"🔴 {reds} parts with errors: {labels}")
    if yellows:
        lines.append(f"🟡 {yellows} parts with warnings")
    if greens:
        lines.append(f"🟢 {greens} parts ready to print")
    return "\n".join(lines) if lines else "No parts to check."
