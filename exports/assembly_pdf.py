"""
assembly_pdf.py
Generate an assembly guide PDF for sliced parts.

Creates:
  - Cover page: project name, part count, total filament estimate
  - Assembly overview: all parts shown as numbered grid
  - Per-plate pages: which parts go on which print plate
  - Connection guide: which parts join to which, with joint types
  - Part index: alphabetical list with dimensions
"""
import os
import numpy as np
from typing import List, Optional
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, PageBreak, HRFlowable)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.graphics.shapes import Drawing, Rect, String, Circle, Line
from reportlab.graphics import renderPDF


# Colour scheme
COL_PRIMARY   = colors.HexColor('#1565C0')
COL_ACCENT    = colors.HexColor('#F57F17')
COL_DARK      = colors.HexColor('#1a1d2a')
COL_LIGHT     = colors.HexColor('#e8eaf6')
COL_GREY      = colors.HexColor('#9E9E9E')
COL_GREEN     = colors.HexColor('#2E7D32')
COL_WHITE     = colors.white


def generate_assembly_pdf(output_path: str,
                           parts,   # list of Part objects
                           plate_assignments: List[List[str]],
                           project_name: str = "3D Print Slicer Build",
                           printer_name: str = "Bambu Lab P2S",
                           total_filament_g: float = 0.0,
                           total_time_str: str = "—",
                           material: str = "PETG") -> bool:
    """
    Generate a full assembly guide PDF.
    parts: list of Part objects with .label, .mesh, .get_dimensions()
    plate_assignments: [[label, label, ...], [...], ...]  (from bambu_export)
    """
    try:
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=15*mm, leftMargin=15*mm,
            topMargin=15*mm, bottomMargin=15*mm,
            title=f"{project_name} — Assembly Guide",
            author="3D Print Slicer"
        )

        styles = getSampleStyleSheet()
        story = []

        # ── Cover page ────────────────────────────────────────────
        story.extend(_cover_page(project_name, printer_name,
                                   len(parts), len(plate_assignments),
                                   total_filament_g, total_time_str, styles))
        story.append(PageBreak())

        # ── Parts overview table ──────────────────────────────────
        story.extend(_parts_overview(parts, styles))
        story.append(PageBreak())

        # ── Plate assignments ─────────────────────────────────────
        if plate_assignments:
            story.extend(_plate_pages(plate_assignments, parts, styles))
            story.append(PageBreak())

        # ── Connection guide ──────────────────────────────────────
        story.extend(_connection_guide(parts, styles))
        story.append(PageBreak())

        # ── Print settings tips ───────────────────────────────────
        story.extend(_print_tips(styles, material))

        doc.build(story)
        return True

    except Exception as e:
        print(f"PDF generation error: {e}")
        return False


def _cover_page(project_name, printer, n_parts, n_plates,
                 filament_g, time_str, styles):
    elems = []

    # Title block
    title_style = ParagraphStyle('title', fontSize=28, fontName='Helvetica-Bold',
                                  textColor=COL_PRIMARY, alignment=TA_CENTER,
                                  spaceAfter=6)
    sub_style = ParagraphStyle('sub', fontSize=14, fontName='Helvetica',
                                textColor=COL_GREY, alignment=TA_CENTER, spaceAfter=4)
    elems.append(Spacer(1, 20*mm))
    elems.append(Paragraph("⬡ KARTSLICER", ParagraphStyle('brand', fontSize=11,
        fontName='Helvetica-Bold', textColor=COL_ACCENT, alignment=TA_CENTER)))
    elems.append(Spacer(1, 4*mm))
    elems.append(Paragraph(project_name, title_style))
    elems.append(Paragraph("Assembly Guide", sub_style))
    elems.append(Spacer(1, 8*mm))
    elems.append(HRFlowable(width="100%", thickness=2, color=COL_PRIMARY))
    elems.append(Spacer(1, 10*mm))

    # Stats grid
    stats = [
        ["Total Parts", str(n_parts)],
        ["Print Plates", str(n_plates)],
        ["Estimated Filament", f"{filament_g:.0f}g" if filament_g else "—"],
        ["Estimated Print Time", time_str],
        ["Printer", printer],
    ]
    stat_style = ParagraphStyle('stat_key', fontSize=10, textColor=COL_GREY)
    stat_val_style = ParagraphStyle('stat_val', fontSize=16,
                                     fontName='Helvetica-Bold', textColor=COL_DARK)
    tdata = []
    for key, val in stats:
        tdata.append([Paragraph(key, stat_style), Paragraph(val, stat_val_style)])

    t = Table(tdata, colWidths=[80*mm, 90*mm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), COL_LIGHT),
        ('ROWBACKGROUNDS', (0,0), (-1,-1), [COL_LIGHT, COL_WHITE]),
        ('GRID', (0,0), (-1,-1), 0.5, COL_GREY),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    elems.append(t)
    elems.append(Spacer(1, 12*mm))

    # Assembly tips
    tip_style = ParagraphStyle('tip', fontSize=9, textColor=COL_DARK,
                                leftIndent=5, spaceAfter=4)
    elems.append(Paragraph("<b>Before you start:</b>", ParagraphStyle('tiptitle',
        fontSize=11, fontName='Helvetica-Bold', textColor=COL_PRIMARY, spaceAfter=4)))
    tips = [
        "Dry-fit all parts before applying adhesive to check alignment.",
        "Sand the cut faces lightly (120 grit) for a flush join.",
        "Use body filler / spot putty to hide seam lines after assembly.",
        "For dowel joins: insert rod from one side before the glue sets.",
        "Print with 4 walls and 15% infill for body panel rigidity.",
        "PETG or ASA recommended for outdoor use (UV/heat resistance).",
    ]
    for tip in tips:
        elems.append(Paragraph(f"• {tip}", tip_style))

    return elems


def _parts_overview(parts, styles):
    elems = []
    h1 = ParagraphStyle('h1', fontSize=16, fontName='Helvetica-Bold',
                          textColor=COL_PRIMARY, spaceAfter=6)
    elems.append(Paragraph("Parts List", h1))
    elems.append(HRFlowable(width="100%", thickness=1, color=COL_LIGHT))
    elems.append(Spacer(1, 4*mm))

    # Table of all parts
    header = ["#", "Part Name", "X (mm)", "Y (mm)", "Z (mm)", "Joint"]
    rows = [header]
    for i, part in enumerate(parts):
        dims = part.get_dimensions()
        joint = getattr(part, '_joint_type', 'flat').replace('_', ' ').title()
        rows.append([
            str(i+1),
            part.label,
            f"{dims[0]:.1f}",
            f"{dims[1]:.1f}",
            f"{dims[2]:.1f}",
            joint,
        ])

    col_w = [12*mm, 55*mm, 22*mm, 22*mm, 22*mm, 30*mm]
    t = Table(rows, colWidths=col_w)
    header_bg = COL_PRIMARY
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), header_bg),
        ('TEXTCOLOR', (0,0), (-1,0), COL_WHITE),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [COL_WHITE, COL_LIGHT]),
        ('GRID', (0,0), (-1,-1), 0.3, COL_GREY),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('ALIGN', (1,0), (1,-1), 'LEFT'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
    ]))
    elems.append(t)
    return elems


def _plate_pages(plate_assignments, parts, styles):
    """One section showing plate assignments."""
    elems = []
    h1 = ParagraphStyle('h1', fontSize=16, fontName='Helvetica-Bold',
                          textColor=COL_PRIMARY, spaceAfter=6)
    elems.append(Paragraph("Print Plates", h1))
    elems.append(HRFlowable(width="100%", thickness=1, color=COL_LIGHT))
    elems.append(Spacer(1, 4*mm))

    # Build label→dimensions map
    label_dims = {p.label: p.get_dimensions() for p in parts}

    for plate_idx, plate_labels in enumerate(plate_assignments):
        plate_title = ParagraphStyle('pt', fontSize=11, fontName='Helvetica-Bold',
                                      textColor=COL_ACCENT, spaceAfter=3)
        elems.append(Paragraph(f"Plate {plate_idx+1}  ({len(plate_labels)} parts)", plate_title))

        rows = [["Part", "X mm", "Y mm", "Z mm"]]
        for lbl in plate_labels:
            dims = label_dims.get(lbl, (0,0,0))
            rows.append([lbl, f"{dims[0]:.0f}", f"{dims[1]:.0f}", f"{dims[2]:.0f}"])

        t = Table(rows, colWidths=[70*mm, 28*mm, 28*mm, 28*mm])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), COL_ACCENT),
            ('TEXTCOLOR', (0,0), (-1,0), COL_WHITE),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 8),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [COL_WHITE, COL_LIGHT]),
            ('GRID', (0,0), (-1,-1), 0.3, COL_GREY),
            ('ALIGN', (1,0), (-1,-1), 'CENTER'),
            ('TOPPADDING', (0,0), (-1,-1), 3),
            ('BOTTOMPADDING', (0,0), (-1,-1), 3),
            ('LEFTPADDING', (0,0), (-1,-1), 4),
        ]))
        elems.append(t)
        elems.append(Spacer(1, 4*mm))

    return elems


def _connection_guide(parts, styles):
    elems = []
    h1 = ParagraphStyle('h1', fontSize=16, fontName='Helvetica-Bold',
                          textColor=COL_PRIMARY, spaceAfter=6)
    elems.append(Paragraph("Assembly Order", h1))
    elems.append(HRFlowable(width="100%", thickness=1, color=COL_LIGHT))
    elems.append(Spacer(1, 4*mm))

    body_style = ParagraphStyle('body', fontSize=9, textColor=COL_DARK, spaceAfter=3)
    elems.append(Paragraph(
        "Parts are numbered in assembly order (001 = bottom-left-front, "
        "numbers increase along X then Y then Z). "
        "Adjacent parts share cut faces — connect them in numerical order for easiest assembly.",
        body_style))
    elems.append(Spacer(1, 4*mm))

    # Simple numbered list
    for i, part in enumerate(parts):
        dims = part.get_dimensions()
        joint = getattr(part, '_joint_type', 'flat').replace('_',' ').title()
        elems.append(Paragraph(
            f"<b>{part.label}</b>  —  {dims[0]:.0f}×{dims[1]:.0f}×{dims[2]:.0f}mm  |  Joint: {joint}",
            ParagraphStyle('pi', fontSize=8, textColor=COL_DARK,
                            leftIndent=5, spaceAfter=2)))

    return elems


def _print_tips(styles, material: str = "PETG"):
    elems = []
    h1 = ParagraphStyle('h1', fontSize=16, fontName='Helvetica-Bold',
                          textColor=COL_PRIMARY, spaceAfter=6)
    elems.append(Paragraph("Print & Finishing Tips", h1))
    elems.append(HRFlowable(width="100%", thickness=1, color=COL_LIGHT))
    elems.append(Spacer(1, 4*mm))

    tip_style = ParagraphStyle('tip', fontSize=9, textColor=COL_DARK,
                                leftIndent=8, spaceAfter=5)
    mat = material.upper()

    # Print settings
    plate_tip = {
        'PETG': "Plate adhesive: Bambu Cool Plate or smooth PEI (release agent helps)",
        'PLA':  "Plate adhesive: Bambu Cool Plate or PEI — PLA sticks easily",
        'ASA':  "Plate adhesive: Textured PEI. Enclose printer to prevent warping",
        'ABS':  "Plate adhesive: Textured PEI + glue stick. MUST enclose printer — ABS warps badly",
    }.get(mat, "Plate adhesive: Bambu Cool Plate or smooth PEI")

    sections = [
        (f"Print Settings — {mat}", [
            "Layer height: 0.2mm standard, 0.16mm for better surface quality",
            "Walls: 4 outer walls for panel rigidity",
            "Infill: 15% gyroid — strong, lightweight, print-time efficient",
            "Support: Enable for any face > 45° overhang",
            plate_tip,
        ]),
    ]

    # Material-specific bonding section
    bond_tips = {
        'PETG': [
            "Primary bond: Plastic weld with MEK solvent — strongest PETG bond",
            "Alternative: 2-part epoxy (5-min or 30-min depending on alignment time needed)",
            "Quick tack: CA glue + activator, then reinforce with epoxy bead on inside",
            "PETG does not dissolve in acetone — don't try acetone welding",
            "Sand cut faces with 120 grit before bonding — rougher = better adhesion",
        ],
        'PLA': [
            "Primary bond: CA glue (superglue) — fast and strong on PLA",
            "Alternative: 2-part epoxy for load-bearing joins",
            "PLA can be lightly softened with a soldering iron for spot-welding",
            "Avoid heat — PLA softens at 55-60°C (don't leave in car!)",
            "Sand with 120 grit before gluing",
        ],
        'ASA': [
            "Primary bond: Acetone welding — ASA dissolves in acetone for chemical weld",
            "Apply thin coat of acetone to both faces, press together, hold 30sec",
            "Alternative: MEK solvent weld (stronger than acetone)",
            "Epoxy for structural joints where acetone can't reach",
            "ASA is UV-resistant — best for outdoor use",
        ],
        'ABS': [
            "Primary bond: Acetone welding — ABS dissolves readily in acetone",
            "Make ABS slurry (ABS scraps + acetone) for gap-filling paste",
            "Apply slurry to both faces, press, clamp for 2+ hours",
            "Acetone vapor smoothing can hide seam lines after joining",
            "Caution: ABS warps easily — print enclosed with consistent temperature",
        ],
    }
    sections.append((f"Bonding {mat} Parts", bond_tips.get(mat, bond_tips['PETG'])))

    # Finishing section — material-specific
    finish_tips = {
        'PETG': [
            "Filler primer: 2-3 coats, wet-sand 400 → 800 → 1200 grit",
            "PETG is harder to sand than PLA — use wet sanding technique",
            "Seams: body filler (Bondo) or 2-part epoxy filler, sand flush",
            "Top coat: 2K automotive clear or spray paint for durability",
            "PETG resists heat to ~80°C — safe in vehicles",
        ],
        'PLA': [
            "Filler primer: 2-3 coats, sand between coats with 400 grit",
            "PLA sands easily — start at 220 grit for rough shaping",
            "Layer lines: fill with spot putty, sand smooth, re-prime",
            "Caution: PLA softens with heat — avoid heat gun finishing",
            "Top coat: spray paint (acrylic or enamel) — avoid solvent-based paints",
        ],
        'ASA': [
            "Acetone vapor smoothing eliminates layer lines entirely",
            "Hang parts in sealed container with acetone-soaked cloth for 30-60min",
            "Sand with 400 grit after smoothing for paint prep",
            "UV-resistant — doesn't yellow outdoors like PLA",
            "Top coat: automotive 2K paint for best outdoor durability",
        ],
        'ABS': [
            "Acetone vapor smoothing gives glass-smooth finish",
            "Brief exposure (2-5 min) for light smoothing, longer for full gloss",
            "ABS slurry fills gaps and seams — sand smooth after drying",
            "Prime with filler primer, wet-sand 400 → 800",
            "Top coat: any paint system works — ABS accepts paint well",
        ],
    }
    sections.append((f"Finishing {mat} Parts", finish_tips.get(mat, finish_tips['PETG'])))

    # Common tips
    sections.append(("General Assembly Tips", [
        "Always dry-fit before any adhesive — check alignment with pins/dowels",
        "Assemble in order: interior supports first, then exterior panels",
        "Sand seam lines from inside with 120 grit for glue surface prep",
        "Clamp or tape parts while adhesive cures — don't rush",
        "Sanding sequence: 220 → 400 → 800 → 1200 for mirror finish",
        "Final step: wet-sand with 1500 grit + polishing compound for show finish",
    ]))

    for title, tips in sections:
        elems.append(Paragraph(f"<b>{title}</b>",
                                ParagraphStyle('sh', fontSize=11,
                                               fontName='Helvetica-Bold',
                                               textColor=COL_ACCENT, spaceAfter=3)))
        for tip in tips:
            elems.append(Paragraph(f"&bull; {tip}", tip_style))
        elems.append(Spacer(1, 4*mm))

    return elems
