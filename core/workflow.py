"""
workflow.py
Guided workflow definitions for the step-by-step wizard.

Each workflow step has:
  - name: short title
  - description: what it does and when to use it
  - action: function to call
  - can_skip: whether the user can skip this step
  - undo_label: what to show in undo history
"""
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, field


@dataclass
class WorkflowStep:
    """A single step in a guided workflow."""
    name: str
    description: str
    section_key: str = ""       # which sidebar section to expand
    can_skip: bool = True
    auto_run: bool = False      # run automatically without asking
    undo_label: str = ""


# ═══════════════════════════════════════════════════════
# WORKFLOW: Import → Repair → Resize → Slice → Export
# ═══════════════════════════════════════════════════════

MAIN_WORKFLOW = [
    WorkflowStep(
        name="1. Import or Generate Model",
        description=(
            "Load a 3D model file (STL, OBJ, 3MF) or generate one from a photo using AI.\n\n"
            "SUPPORTED FORMATS:\n"
            "• STL — most common for 3D printing\n"
            "• OBJ — supports colours and materials\n"
            "• 3MF — Bambu Studio native format\n\n"
            "AI GENERATION:\n"
            "• Take 1-4 photos of your object\n"
            "• AI creates a 3D model automatically\n"
            "• Requires NVIDIA GPU (run on your home PC)"
        ),
        section_key="import",
    ),
    WorkflowStep(
        name="2. Auto-Repair Mesh",
        description=(
            "Fix common problems that prevent clean slicing.\n\n"
            "WHAT GETS FIXED:\n"
            "• Holes in the surface → filled\n"
            "• Inverted faces (inside-out) → flipped\n"
            "• Duplicate/degenerate triangles → removed\n"
            "• Floating debris → removed\n"
            "• Non-manifold edges → reported\n\n"
            "This runs automatically on import if 'Auto-repair' is checked.\n"
            "You can also run individual repairs manually."
        ),
        section_key="repair",
        auto_run=True,
        undo_label="repair",
    ),
    WorkflowStep(
        name="3. Improve Mesh Quality (Optional)",
        description=(
            "Improve the mesh surface for better print results.\n\n"
            "THREE OPTIONS:\n"
            "• ISOTROPIC REMESH — makes all triangles even-sized\n"
            "  Best for: scans, AI meshes with uneven triangles\n\n"
            "• ADAPTIVE REMESH — more detail on curves, less on flats\n"
            "  Best for: car bodies, organic shapes\n\n"
            "• SUBDIVIDE — adds triangles everywhere (4x per pass)\n"
            "  Best for: very low-poly models that look blocky\n\n"
            "SKIP THIS if your model already looks smooth."
        ),
        section_key="repair",
        can_skip=True,
        undo_label="remesh",
    ),
    WorkflowStep(
        name="4. Resize to Target Dimensions",
        description=(
            "Scale the model to the size you want to print.\n\n"
            "• Set target dimensions in mm for X, Y, Z\n"
            "• Use 'Uniform' to keep proportions (recommended)\n"
            "• Check the % change indicators to see how much you're scaling\n\n"
            "TIP: Measure your real-world target first.\n"
            "For a go-kart body, measure the chassis and add clearance."
        ),
        section_key="resize",
        undo_label="resize",
    ),
    WorkflowStep(
        name="5. Slice into Printable Pieces",
        description=(
            "Cut the model into pieces that fit your printer.\n\n"
            "AUTO-SLICE:\n"
            "• Set part size (default matches your printer profile)\n"
            "• Click 'Preview Cuts' to see where cuts will go\n"
            "• Adjust individual cuts by dragging in the viewport\n"
            "• Click 'Apply All Cuts' to split the model\n\n"
            "MANUAL CUT:\n"
            "• Use the Quick Cut bar below the viewport\n"
            "• Right-click a part → 'Cut This Part'\n\n"
            "CUT MODES:\n"
            "• Full — straight through\n"
            "• Angled — tilted cut plane\n"
            "• Groove — zigzag teeth for mechanical interlock\n"
            "• Natural — follows surface creases to hide seams"
        ),
        section_key="slice",
        undo_label="slice",
    ),
    WorkflowStep(
        name="6. Add Connectors (Optional)",
        description=(
            "Add alignment features so parts fit together during assembly.\n\n"
            "CONNECTOR TYPES:\n"
            "• Round Dowel — steel rod holes\n"
            "• D-Shape — prevents rotation\n"
            "• Pyramid — self-centering taper\n"
            "• Terrace — stepped, large bonding area\n"
            "• Magnet — press-fit magnet pockets\n"
            "• Square — square peg, prevents rotation\n"
            "• Snap-Fit — clip that clicks together\n"
            "• Dovetail — mechanical wedge interlock\n\n"
            "SKIP THIS if you plan to just glue the parts together."
        ),
        section_key="joints",
        can_skip=True,
    ),
    WorkflowStep(
        name="7. Export for Printing",
        description=(
            "Export all parts ready for your slicer.\n\n"
            "EXPORT OPTIONS:\n"
            "• STL/OBJ/3MF — individual part files\n"
            "• Bambu .3mf — all parts packed onto plates, opens in Bambu Studio\n"
            "• Assembly PDF — numbered guide with tips\n\n"
            "BEFORE EXPORTING:\n"
            "• Run 'Printability Check' to verify all parts are OK\n"
            "• Run 'Bond Analysis' to check joint strength\n"
            "• Check 'Emboss part numbers' for assembly labelling"
        ),
        section_key="export",
    ),
]


# Feature info database — used by the info buttons throughout the UI
FEATURE_INFO = {
    'auto_repair': {
        'title': 'Auto-Repair Mesh',
        'text': (
            "Automatically fixes common mesh problems:\n\n"
            "✓ Degenerate faces (zero-area triangles)\n"
            "✓ Duplicate faces\n"
            "✓ Orphan vertices\n"
            "✓ Inconsistent normals\n"
            "✓ Holes in the surface\n"
            "✓ Floating debris shells\n\n"
            "Runs automatically when you import a model.\n"
            "Each fix can be undone individually."
        ),
    },
    'isotropic_remesh': {
        'title': 'Isotropic Remesh',
        'text': (
            "Makes all triangles roughly the same size.\n\n"
            "USE WHEN: Mesh has uneven triangles (common in scans and AI models).\n"
            "RESULT: Smoother, more predictable print quality.\n"
            "CAUTION: May change total triangle count."
        ),
    },
    'adaptive_remesh': {
        'title': 'Adaptive Remesh',
        'text': (
            "Smart remeshing — more detail on curves, less on flats.\n\n"
            "USE WHEN: Printing organic shapes like car bodies.\n"
            "RESULT: Best surface quality with minimal waste.\n"
            "CAUTION: May increase triangle count on complex models."
        ),
    },
    'subdivide': {
        'title': 'Subdivide Mesh',
        'text': (
            "Splits each triangle into 4 smaller ones.\n\n"
            "USE WHEN: Model looks blocky (under 1000 triangles).\n"
            "RESULT: 4x triangles per pass.\n"
            "CAUTION: 2 passes = 16x, 3 passes = 64x. Can make files huge."
        ),
    },
    'groove_cut': {
        'title': 'Groove / Zigzag Cut',
        'text': (
            "Creates interlocking teeth on the cut face.\n\n"
            "Parts mechanically lock together before gluing.\n"
            "Prevents sliding during assembly.\n\n"
            "PARAMETERS:\n"
            "• Teeth count — how many teeth\n"
            "• Depth — how deep the teeth go\n"
            "• Width — how wide each tooth is"
        ),
    },
    'natural_cut': {
        'title': 'Natural Cut (Follow Crease)',
        'text': (
            "Moves the cut to follow a natural crease/panel line.\n\n"
            "The seam hides along an existing edge in the model,\n"
            "making it nearly invisible after assembly and finishing.\n\n"
            "RUN 'Show Seam Heatmap' first to detect creases.\n"
            "Then use Natural cut mode — it auto-snaps to the nearest crease."
        ),
    },
    'face_labels': {
        'title': 'Face Labels (A1/A2 Matching)',
        'text': (
            "Embosses matching labels on both sides of each joint.\n\n"
            "A1 connects to A2, B1 to B2, etc.\n"
            "Makes assembly obvious — you always know which pieces go together.\n\n"
            "Click 'Preview Labels' to see them in the viewport first."
        ),
    },
    'tolerance_test': {
        'title': 'Tolerance Test Print',
        'text': (
            "Generates a small test piece with your connector at 6 tolerances.\n\n"
            "Print it, test the fit at each hole, then use the best tolerance.\n"
            "Saves wasting filament on a full model that doesn't fit.\n\n"
            "Each printer is slightly different — this calibrates YOUR printer."
        ),
    },
    'manifold_booleans': {
        'title': 'Manifold3D Boolean Engine',
        'text': (
            "Uses Microsoft's Manifold library for mesh boolean operations.\n\n"
            "This is what PrusaSlicer uses internally.\n"
            "Guarantees correct results for:\n"
            "• Dowel hole subtraction\n"
            "• Groove teeth generation\n"
            "• Dovetail cutting\n"
            "• Any connector that modifies the mesh"
        ),
    },
    'pymeshfix': {
        'title': 'Deep Repair (PyMeshFix)',
        'text': (
            "Advanced mesh repair that handles problems basic repair can't fix:\n\n"
            "✓ Self-intersecting faces\n"
            "✓ Complex non-manifold topology\n"
            "✓ Tangled internal geometry\n\n"
            "Use this AFTER basic auto-repair if the mesh is still broken.\n"
            "PyMeshFix uses a different algorithm that's more aggressive."
        ),
    },
    'partcrafter': {
        'title': 'PartCrafter (Pre-Separated Parts)',
        'text': (
            "AI that generates a 3D model with parts ALREADY SEPARATED.\n\n"
            "For a car photo, you get: body, wheels, bumper, etc.\n"
            "as individual meshes — no manual splitting needed.\n\n"
            "Needs ~12GB VRAM. NeurIPS 2025 research.\n"
            "Best for: complex objects with distinct components."
        ),
    },
    'text_to_3d': {
        'title': 'Text to 3D (OpenSCAD)',
        'text': (
            "Describe a shape in words and get a 3D model.\n\n"
            "Examples:\n"
            "  'box 100x80x60 with rounded corners'\n"
            "  'cylinder 50mm diameter 100mm tall'\n\n"
            "Uses OpenSCAD — parametric models you can edit.\n"
            "No GPU needed, runs on CPU.\n"
            "Install OpenSCAD from openscad.org first."
        ),
    },
}
