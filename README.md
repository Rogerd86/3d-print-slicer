# 3D Print Slicer

**Model preparation, splitting, repair and AI generation for desktop 3D printers.**

Take any mesh — imported STL/OBJ/3MF, a scan, or a photo you ran through AI —
and turn it into a bag of print-ready parts sized for your printer, with
snap-fit or dowel joints already modelled in.

Built for a Bambu Lab P1S / P2S workflow but works for any FDM printer with
a known build volume.

---

## What it does

### Import
- **STL / OBJ / 3MF** direct import
- Drag-and-drop onto the viewport
- **Generate 3D from photo** via TRELLIS 2 / SAM 3D / PartCrafter / TripoSR
- **Generate 3D from text** via OpenSCAD (CPU only, no GPU)
- Colour / material splitting for multi-body 3MF files

### Clean up
- **Full Auto-Repair** — degenerate faces, duplicates, winding, normals, hole
  filling, shell pruning, non-manifold detection
- **Print-Ready Repair** — everything above plus:
  - Self-intersection cleanup via manifold3d round-trip
  - Non-manifold edge splitting
  - Sliver triangle removal (aspect ratio > 60)
  - Thin-wall scan with pass/fail report
- **Thin-Wall Heatmap** — paints the mesh red/yellow/green by local wall
  thickness so you can see problems rather than get a list
- **Edge-Flip Optimisation** — improves triangle aspect ratios without
  changing the silhouette
- **Decimation** — both classical quadric and feature-preserving
  `decimate_pro` (preserves sharp edges)
- **Remeshing** — isotropic and adaptive (curvature-aware)
- **Smoothing** — Taubin / Laplacian / Humphrey with cut-face locking

### Slice
- **Auto-slice** — automatically partitions the mesh into build-volume-sized
  pieces using parallel workers
- **Manual cuts** with five modes:
  - Full (planar)
  - Free / Angled
  - Section (bounded rectangle)
  - Groove / Zigzag (interlocking teeth)
  - Natural (follow detected surface creases)
- **Quick Cut bar** under the viewport — two-way-synced with the Advanced
  Options sidebar so the preview always matches the cut
- **Undo** history across every edit
- **Seam heatmap** — visualises where cuts will be least visible
- **Natural-crease snapping** on the cut plane gizmo

### Connect
- **Joint types**:
  - Flat (glue)
  - Dowel holes (steel-pin locating)
  - Dovetail slots (mechanical lock)
  - Rectangular peg-and-socket
- **Manual dowel placement** — click the face, drag the sphere along the
  surface, release to commit
- **Connector-shape library** — D-shape, pyramid, terrace, square

### Inspect
- **Part tree** with per-part visibility, solo, wireframe toggle
- **Show Adjacent** — hides everything except the selected part and its
  cut-face neighbours
- **Rotate Object mode** — hold R and drag to rotate the selected part
  around its centroid, independent of camera
- **Reset Orientation** — snap parts back to assembly position
- **Fixed view presets** — Front / Back / Left / Right / Top / Bottom / Iso
- **Explode slider** — dry-fit preview
- **Measure tool**, **build volume overlay**, **seam heatmap**

### Export
- **STL / OBJ / 3MF** with manifest
- **Bambu Studio .3mf** with plates pre-packed
- **Assembly guide PDF** — per-material finishing tips (PLA/PETG/ASA)
- **Print estimate** — filament grams, print time, weight, cost

---

## Requirements

| | Minimum | Recommended |
|---|---|---|
| Python | 3.10 | 3.11 or 3.12 (3.13 has wheel gaps) |
| RAM | 8 GB | 16 GB |
| GPU (optional) | — | NVIDIA 8 GB+ VRAM (for AI backends) |
| OS | Windows 10, macOS 12, Ubuntu 22.04 | Windows 11 |

Core dependencies (auto-installed by the run script):
`numpy`, `PyQt5`, `trimesh`, `scipy`, `rtree`, `networkx`, `pyvista`,
`pyvistaqt`, `qtawesome`, `pymeshfix`, `PyOpenGL`, `manifold3d` (optional).

---

## Installation

### Windows
```bat
run_windows.bat
```
Installs everything and launches the app. Re-run any time — it only
installs missing packages.

### macOS / Linux
```bash
chmod +x run_linux_mac.sh
./run_linux_mac.sh
```

### Manual
```bash
pip install -r requirements.txt
python main.py
```

### AI backends (optional)
Only needed if you want "Generate 3D from Photo". Run:
```bat
setup_ai_backends.bat
```
Interactive — prompts for each backend so you can skip ones you don't
need. Needs Git + NVIDIA GPU + ~10 GB disk for model weights.

---

## Typical workflow

1. **Import** your model.
2. **Print-Ready Repair** → fixes manifold/intersections in one click.
3. **Thin-Wall Heatmap** → spot any walls that'll under-extrude.
4. **Auto-Slice** → the app computes cut planes sized to your build plate.
5. **Set Joint** on each cut face → pick dowels / dovetails / flat.
6. **Preview Assembly (Dry-Fit)** → spins up the assembled view.
7. **Export for Bambu Studio** → one `.3mf` with plates packed and ready.
8. **Assembly Guide PDF** → print it, stick it in the box of parts.

---

## Keyboard / mouse

| Action | Shortcut |
|---|---|
| Import | `Ctrl+O` |
| Save project | `Ctrl+S` |
| Export | `Ctrl+E` |
| Undo | `Ctrl+Z` |
| Toggle visibility of selected | `Space` |
| Delete selected cut plane | `Del` |
| Hold to rotate the selected part | `R` + LMB drag |
| Exit any mode / reset view | `Esc` |
| Focus on selected part | `F` |

---

## Folder layout

```
kart_slicer/
├── main.py                     # entry point
├── core/                       # mesh, cut, repair, AI, export logic
│   ├── auto_repair.py
│   ├── mesh_quality.py         # advanced repair passes
│   ├── cut_definition.py
│   ├── part_tree.py
│   ├── ai_generate.py          # 4 image-to-3D backends
│   └── ...
├── ui/
│   ├── main_window.py          # Qt main window
│   └── viewport.py             # PyVista 3D viewport
├── exports/                    # STL / 3MF / PDF writers
├── run_windows.bat
├── run_linux_mac.sh
├── setup_ai_backends.bat
└── requirements.txt
```

---

## License

MIT — do what you want. Attribution appreciated.

---

## Roadmap

- [ ] Signed-distance thin-wall overlay (not just ray samples)
- [ ] Custom snap-fit clip geometry library
- [ ] Per-part optimal print-orientation solver
- [ ] GCode-level support-material estimation
- [ ] Embossed part-number labels on the back face
- [ ] Export preset cloud sync
