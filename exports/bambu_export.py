"""
bambu_export.py
Export sliced parts as a Bambu Studio-compatible 3MF file.

Creates a properly structured 3MF with:
- All parts as separate meshes with unique names
- Parts arranged on virtual print plates (256x256mm for P2S)
- Each plate gets its own build item group
- Multi-material support: parts with colours get material assignments
- File opens directly in Bambu Studio with parts pre-arranged

Bambu Studio 3MF structure:
  3D/3dmodel.model     - mesh data + build items
  _rels/.rels          - relationships
  [Content_Types].xml  - MIME types
"""
import os
import io
import zipfile
import numpy as np
import trimesh
from typing import List, Tuple, Optional
from xml.etree import ElementTree as ET


# Bambu 3MF namespaces
NS_3MF  = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
NS_REL  = "http://schemas.openxmlformats.org/package/2006/relationships"
NS_CONT = "http://schemas.openxmlformats.org/package/2006/content-types"
NS_BAMBU = "http://schemas.bambulab.com/package/2021"


def pack_onto_plates(parts_dims: List[Tuple[float, float]],
                      plate_w: float = 256.0,
                      plate_h: float = 256.0,
                      margin: float = 8.0) -> List[List[dict]]:
    """
    Pack parts onto print plates using row-based bin packing with rotation.
    Returns list of plates, each plate is a list of dicts:
      {idx, x, y, w, h, rotated}
    """
    plates = []
    remaining = list(enumerate(parts_dims))

    while remaining:
        plate = []
        x, y, row_h = margin, margin, 0
        packed = []

        for i, (w, h) in remaining:
            placed = False
            for pw, ph in [(w, h), (h, w)]:  # try both orientations
                if pw > plate_w - 2*margin or ph > plate_h - 2*margin:
                    continue
                if x + pw + margin > plate_w:
                    x = margin; y += row_h + margin; row_h = 0
                if y + ph + margin <= plate_h:
                    plate.append({'idx': i, 'x': x, 'y': y,
                                  'w': pw, 'h': ph, 'rotated': (pw != w)})
                    x += pw + margin; row_h = max(row_h, ph)
                    packed.append(i); placed = True; break

        if not packed:
            # Part too big for any plate — place alone on oversized plate
            i, (w, h) = remaining[0]
            plate.append({'idx': i, 'x': margin, 'y': margin,
                          'w': w, 'h': h, 'rotated': False})
            packed.append(i)

        plates.append(plate)
        remaining = [(i, d) for i, d in remaining if i not in packed]

    return plates


def export_bambu_3mf(parts: List[trimesh.Trimesh],
                      labels: List[str],
                      output_path: str,
                      plate_w: float = 256.0,
                      plate_h: float = 256.0,
                      auto_orient: bool = True,
                      part_colours: Optional[List[Optional[Tuple[float,float,float]]]] = None
                      ) -> Tuple[bool, str, List[List[str]]]:
    """
    Export parts to a Bambu Studio-compatible 3MF file.

    Parts are:
    1. Auto-oriented (largest flat face down) if auto_orient=True
    2. Packed onto plates of plate_w x plate_h mm
    3. Written to a single 3MF that Bambu Studio opens with all plates ready

    part_colours: optional list of (r, g, b) float tuples (0-1) per part.
                  None entries = no colour assigned.

    Returns (success, message, plate_assignments)
    plate_assignments[i] = list of part labels on plate i
    """
    try:
        # Auto-orient parts
        oriented_parts = []
        for mesh in parts:
            if auto_orient:
                mesh = _orient_flat_down(mesh)
            oriented_parts.append(mesh)

        # Get XY footprint of each part
        dims_xy = []
        for mesh in oriented_parts:
            e = mesh.extents
            dims_xy.append((float(e[0]) + 2, float(e[1]) + 2))  # +2mm clearance

        # Pack onto plates
        plates = pack_onto_plates(dims_xy, plate_w, plate_h)

        # Build 3MF XML
        model_xml = _build_3mf_model(oriented_parts, labels, plates,
                                      plate_w, plate_h, part_colours)
        rels_xml  = _build_rels()
        types_xml = _build_content_types()

        # Write ZIP
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('3D/3dmodel.model', model_xml)
            zf.writestr('_rels/.rels', rels_xml)
            zf.writestr('[Content_Types].xml', types_xml)

        with open(output_path, 'wb') as f:
            f.write(buf.getvalue())

        plate_assignments = []
        for plate in plates:
            plate_assignments.append([labels[item['idx']] for item in plate])

        n_coloured = sum(1 for c in (part_colours or []) if c is not None)
        colour_msg = f" ({n_coloured} with materials)" if n_coloured > 0 else ""
        msg = (f"Exported {len(parts)} parts{colour_msg} across {len(plates)} plates\n"
               f"Open {os.path.basename(output_path)} in Bambu Studio")
        return True, msg, plate_assignments

    except Exception as e:
        return False, f"Bambu export error: {e}", []


def _orient_flat_down(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Rotate mesh so its largest flat face is on the bottom (Z=0).
    This minimises supports for most printed parts.
    """
    try:
        if not hasattr(mesh, 'facets') or len(mesh.facets) == 0:
            return mesh

        # Find largest facet
        best = int(np.argmax(mesh.facets_area))
        normal = mesh.facets_normal[best]

        # Rotate so this normal points DOWN (-Z)
        target = np.array([0., 0., -1.])
        if np.allclose(normal, target, atol=0.01):
            return mesh  # already flat down

        # Compute rotation
        rot = trimesh.geometry.align_vectors(normal, target)
        oriented = mesh.copy()
        oriented.apply_transform(rot)

        # Translate so bottom is at Z=0
        oriented.apply_translation([0, 0, -oriented.bounds[0][2]])
        return oriented

    except Exception:
        return mesh


def _mesh_to_3mf_vertices_triangles(mesh: trimesh.Trimesh,
                                      offset_x: float, offset_y: float
                                      ) -> Tuple[List[str], List[str]]:
    """Convert mesh to 3MF vertex/triangle XML strings with XY offset applied."""
    verts = mesh.vertices.copy()
    verts[:, 0] += offset_x
    verts[:, 1] += offset_y

    v_lines = []
    for v in verts:
        v_lines.append(f'<vertex x="{v[0]:.4f}" y="{v[1]:.4f}" z="{v[2]:.4f}"/>')

    t_lines = []
    for face in mesh.faces:
        t_lines.append(f'<triangle v1="{face[0]}" v2="{face[1]}" v3="{face[2]}"/>')

    return v_lines, t_lines


def _mesh_to_3mf_triangles_with_material(mesh: trimesh.Trimesh,
                                           offset_x: float, offset_y: float,
                                           material_id: int
                                           ) -> Tuple[List[str], List[str]]:
    """Convert mesh to 3MF XML with material assignment on all triangles."""
    verts = mesh.vertices.copy()
    verts[:, 0] += offset_x
    verts[:, 1] += offset_y

    v_lines = []
    for v in verts:
        v_lines.append(f'<vertex x="{v[0]:.4f}" y="{v[1]:.4f}" z="{v[2]:.4f}"/>')

    t_lines = []
    for face in mesh.faces:
        t_lines.append(
            f'<triangle v1="{face[0]}" v2="{face[1]}" v3="{face[2]}" '
            f'pid="{material_id}" p1="0"/>')

    return v_lines, t_lines


def _build_3mf_model(parts, labels, plates, plate_w, plate_h,
                      part_colours=None) -> str:
    """Build the 3dmodel.model XML for a multi-plate Bambu 3MF."""

    # Collect unique materials from part colours
    materials = []  # list of (name, hex_color)
    part_material_idx = {}  # part_index -> material resource id
    if part_colours:
        colour_map = {}  # hex -> material_index
        for pi, colour in enumerate(part_colours):
            if colour is None:
                continue
            r, g, b = colour
            hex_col = f"#{int(r*255):02X}{int(g*255):02X}{int(b*255):02X}"
            if hex_col not in colour_map:
                colour_map[hex_col] = len(materials)
                materials.append((labels[pi].split('-')[-1], hex_col))
            part_material_idx[pi] = colour_map[hex_col]

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<model unit="millimeter" xml:lang="en-US" xmlns="{NS_3MF}"'
        f' xmlns:m="http://schemas.microsoft.com/3dmanufacturing/material/2015/02">',
        '  <resources>',
    ]

    # Write material definitions if we have colours
    if materials:
        lines.append('    <m:basematerials id="1">')
        for name, hex_col in materials:
            lines.append(f'      <m:base name="{name}" displaycolor="{hex_col}"/>')
        lines.append('    </m:basematerials>')

    obj_id = 2 if materials else 1  # start after material resource
    obj_ids = []  # (part_idx, obj_id)

    for pi, part in enumerate(parts):
        # Find where this part sits on its plate
        placement = None
        for plate in plates:
            for item in plate:
                if item['idx'] == pi:
                    placement = item
                    break
            if placement:
                break

        ox = placement['x'] if placement else 0.0
        oy = placement['y'] if placement else 0.0

        # Use material-aware triangles if this part has a colour
        if pi in part_material_idx:
            mat_idx = part_material_idx[pi]
            v_lines, t_lines = _mesh_to_3mf_triangles_with_material(
                part, ox, oy, 1)  # pid=1 is the basematerials resource
            # Override p1 to use the correct material index
            t_lines_fixed = []
            for tl in t_lines:
                tl = tl.replace('p1="0"', f'p1="{mat_idx}"')
                t_lines_fixed.append(tl)
            t_lines = t_lines_fixed
        else:
            v_lines, t_lines = _mesh_to_3mf_vertices_triangles(part, ox, oy)

        lines.append(f'    <object id="{obj_id}" name="{labels[pi]}" type="model">')
        lines.append('      <mesh>')
        lines.append('        <vertices>')
        lines.extend(f'          {v}' for v in v_lines)
        lines.append('        </vertices>')
        lines.append('        <triangles>')
        lines.extend(f'          {t}' for t in t_lines)
        lines.append('        </triangles>')
        lines.append('      </mesh>')
        lines.append('    </object>')

        obj_ids.append((pi, obj_id))
        obj_id += 1

    lines.append('  </resources>')
    lines.append('  <build>')

    for pi, oid in obj_ids:
        lines.append(f'    <item objectid="{oid}"/>')

    lines.append('  </build>')
    lines.append('</model>')

    return '\n'.join(lines)


def _build_rels() -> str:
    return '''<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Target="/3D/3dmodel.model" Id="rel0"
    Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>'''


def _build_content_types() -> str:
    return '''<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels"
    ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="model"
    ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
</Types>'''
