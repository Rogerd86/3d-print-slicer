"""3D Print Slicer Main Window — Part-tree cutting system."""
import os
import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QSplitter,
    QLabel, QPushButton, QFileDialog, QDoubleSpinBox, QSpinBox,
    QGroupBox, QComboBox, QScrollArea, QFrame, QTableWidget,
    QTableWidgetItem, QHeaderView, QProgressBar, QMessageBox,
    QTabWidget, QCheckBox, QStatusBar, QAbstractItemView,
    QTreeWidget, QTreeWidgetItem, QSlider, QShortcut, QToolButton,
    QSizePolicy, QMenu, QAction
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt5.QtGui import QColor, QFont, QKeySequence, QCursor, QIcon

# qtawesome — Font Awesome icons for a consistent professional look.
# Falls back to emoji text if the package is unavailable.
try:
    import qtawesome as qta
    _HAS_QTA = True
except Exception:
    qta = None
    _HAS_QTA = False


# Mapping from semantic icon name -> (qtawesome name, emoji fallback)
_ICONS = {
    'import':     ('fa5s.file-import',       '📂'),
    'photo':      ('fa5s.image',             '🖼'),
    'text':       ('fa5s.font',              '📝'),
    'cut':        ('fa5s.cut',               '✂'),
    'repair':     ('fa5s.wrench',            '🔧'),
    'diagnose':   ('fa5s.stethoscope',       '🩺'),
    'deep':       ('fa5s.tools',             '🔩'),
    'magnet':     ('fa5s.magnet',            '🧲'),
    'flat':       ('fa5s.level-down-alt',    '⤵'),
    'eye':        ('fa5s.eye',               '👁'),
    'save':       ('fa5s.save',              '💾'),
    'box':        ('fa5s.box',               '📦'),
    'pdf':        ('fa5s.file-pdf',          '📄'),
    'rotate':     ('fa5s.sync-alt',          '⟳'),
    'resize':     ('fa5s.expand-arrows-alt', '📐'),
    'heatmap':    ('fa5s.thermometer-half',  '🌡'),
    'test':       ('fa5s.vial',              '🧪'),
    'brush':      ('fa5s.paint-brush',       '🖌'),
    'measure':    ('fa5s.ruler',             '📏'),
    'preview':    ('fa5s.eye',               '👁'),
    'adjacent':   ('fa5s.project-diagram',   '🔗'),
    'reset':      ('fa5s.undo',              '↺'),
}


def _qicon(name, color='#c0d4f0'):
    """Return a QIcon for the given semantic name. Falls back to empty QIcon if qta missing."""
    if not _HAS_QTA:
        return QIcon()
    qta_name, _ = _ICONS.get(name, (None, ''))
    if not qta_name:
        return QIcon()
    try:
        return qta.icon(qta_name, color=color)
    except Exception:
        return QIcon()


def _icon_label(name, text):
    """Return a label suffix: when qta is missing, prepend emoji to preserve visual cue."""
    if _HAS_QTA:
        return text  # icon is set separately via setIcon
    emoji = _ICONS.get(name, ('', ''))[1]
    return f"{emoji} {text}".strip()


class CollapsibleSection(QWidget):
    """A collapsible sidebar section with a clickable header."""
    def __init__(self, title, parent=None, expanded=False):
        super().__init__(parent)
        self._expanded = expanded
        self._title = title

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 4, 0, 0)  # top margin = gap between sections
        self._layout.setSpacing(0)

        # Header button — taller, more prominent
        self._header = QToolButton()
        self._header.setText(f"{'▾' if expanded else '▸'}  {title}")
        self._header.setObjectName("sectionHeader")
        self._header.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self._header.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._header.setFixedHeight(34)
        self._header.setCheckable(True)
        self._header.setChecked(expanded)
        self._header.clicked.connect(self._toggle)
        self._layout.addWidget(self._header)

        # Content area
        self._content = QWidget()
        self._content.setObjectName("sectionContent")
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(12, 8, 10, 10)
        self._content_layout.setSpacing(6)
        self._content.setVisible(expanded)
        self._layout.addWidget(self._content)

    def content_layout(self):
        return self._content_layout

    def expand(self):
        if not self._expanded and self._header.isEnabled():
            self._toggle()

    def collapse(self):
        if self._expanded:
            self._toggle()

    def set_stage_state(self, state):
        """state: 'locked' | 'available' | 'active' | 'done'
        Controls the visual style and whether the section can be opened."""
        self._stage_state = state
        if state == 'locked':
            self._header.setEnabled(False)
            self._header.setProperty('stageState', 'locked')
            if self._expanded:
                self._toggle()  # force collapse
        else:
            self._header.setEnabled(True)
            self._header.setProperty('stageState', state)
        # Refresh the stylesheet to apply the property
        self._header.style().unpolish(self._header)
        self._header.style().polish(self._header)

    def _toggle(self):
        self._expanded = not self._expanded
        self._content.setVisible(self._expanded)
        self._header.setText(f"{'▾' if self._expanded else '▸'}  {self._title}")
        self._header.setChecked(self._expanded)

from ui.viewport import Viewport3D
from core.mesh_handler import MeshHandler
from core.part_tree import PartTree, Part
from core.cut_definition import CutDefinition
from core.mesh_repair import repair_mesh
from core.hollow import make_hollow_simple, estimate_material_saved
from core.estimator import estimate_all, total_summary
from core.project import save_project, load_project, apply_project
from core.connector_pins import add_pins_to_parts, generate_separate_pin
from core.colour_split import (has_colour_data, has_multi_geometry,
                                split_by_colour, split_scene_by_geometry)
from exports.exporter import Exporter


class AutoSliceWorker(QThread):
    finished  = pyqtSignal()
    progress  = pyqtSignal(int, int)   # completed, total
    error     = pyqtSignal(str)

    def __init__(self, tree, cut_size):
        super().__init__()
        self.tree = tree
        self.cut_size = cut_size

    def run(self):
        try:
            # Use parallel slicer for speed — falls back gracefully
            from core.parallel_slicer import parallel_auto_slice
            root_mesh = self.tree.root.mesh.copy()

            def cb(done, total):
                self.progress.emit(done, total)

            result_meshes = parallel_auto_slice(
                root_mesh, self.cut_size,
                progress_cb=cb, max_workers=4)

            # Load results back into part tree
            from core.part_tree import Part
            self.tree.root.children = []
            Part._color_counter = 1
            base = self.tree.root.label
            for i, mesh in enumerate(result_meshes):
                if mesh is not None and len(mesh.faces) > 0:
                    child = Part(mesh, f"{base}-{i+1:03d}", parent=self.tree.root)
                    self.tree.root.children.append(child)
            self.tree._undo_stack = []
            self.finished.emit()

        except Exception as e:
            self.error.emit(str(e))


class RepairWorker(QThread):
    finished = pyqtSignal(object, object)
    error    = pyqtSignal(str)

    def __init__(self, mesh):
        super().__init__()
        self._mesh = mesh

    def run(self):
        try:
            self.finished.emit(*repair_mesh(self._mesh))
        except Exception as e:
            self.error.emit(str(e))


class AIGenerateWorker(QThread):
    """Run AI mesh generation off the UI thread."""
    finished = pyqtSignal(str, str)     # obj_path, status_message
    error    = pyqtSignal(str)

    def __init__(self, image_paths, output_dir, backend='auto'):
        super().__init__()
        self.image_paths = image_paths
        self.output_dir = output_dir
        self.backend = backend

    def run(self):
        try:
            from core.ai_generate import generate_mesh
            obj_path, msg = generate_mesh(
                self.image_paths, self.output_dir, self.backend)
            if obj_path:
                self.finished.emit(obj_path, msg)
            else:
                self.error.emit(msg)
        except Exception as e:
            self.error.emit(str(e))


class TaskWorker(QThread):
    """Generic worker — runs any callable off the UI thread."""
    finished = pyqtSignal(object)  # result
    error = pyqtSignal(str)

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self._func = func
        self._args = args
        self._kwargs = kwargs

    def run(self):
        try:
            result = self._func(*self._args, **self._kwargs)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class LoadWorker(QThread):
    """Load a 3D model file off the UI thread."""
    finished = pyqtSignal(bool, str)    # success, file_path
    error    = pyqtSignal(str)

    def __init__(self, mesh_handler, file_path):
        super().__init__()
        self._handler = mesh_handler
        self._path = file_path

    def run(self):
        try:
            ok = self._handler.load(self._path)
            self.finished.emit(ok, self._path)
        except Exception as e:
            self.error.emit(str(e))


class ExportWorker(QThread):
    """Runs heavy export operations off the UI thread."""
    finished = pyqtSignal(int, str)    # n_files, out_path
    progress = pyqtSignal(int, int)
    error    = pyqtSignal(str)

    def __init__(self, leaves, out_dir, fmt, do_hollow, wall,
                 do_pins, pin_radius, do_numbers):
        super().__init__()
        self.leaves     = leaves
        self.out_dir    = out_dir
        self.fmt        = fmt
        self.do_hollow  = do_hollow
        self.wall       = wall
        self.do_pins    = do_pins
        self.pin_radius = pin_radius
        self.do_numbers = do_numbers

    def run(self):
        try:
            from core.slicer import SlicedPart
            from core.dowel_joints import (apply_dowels_to_pair, DowelConfig,
                                            _distribute_dowel_positions,
                                            _make_cutters, _boolean_subtract)
            from core.connector_pins import _find_pin_positions, _add_male_pins
            from core.part_numbers import add_part_number_to_mesh
            from core.hollow import make_hollow_simple
            from concurrent.futures import ThreadPoolExecutor, as_completed
            import threading
            import numpy as np

            total = len(self.leaves)
            done_count = 0
            lock = threading.Lock()

            def process_one(idx_part):
                idx, p = idx_part
                mesh = p.mesh.copy()

                if self.do_hollow:
                    h, ok_h = make_hollow_simple(mesh, self.wall)
                    if ok_h: mesh = h

                joint_type = getattr(p, '_joint_type', 'flat')
                joint_config = getattr(p, '_joint_configs', {})

                if joint_type in ('round_dowel','rect_dowel') and 'dowel' in joint_config:
                    try:
                        cfg = joint_config['dowel']
                        bounds = mesh.bounds
                        dims = bounds[1] - bounds[0]
                        ax = int(np.argmin(dims))
                        normal = np.zeros(3); normal[ax] = 1.0
                        origin = mesh.centroid.copy()
                        positions = _distribute_dowel_positions(mesh, normal, origin, cfg)
                        if positions:
                            ca = _make_cutters(positions, normal, cfg,
                                               depth=cfg.depth_a, into_a=True)
                            mesh = _boolean_subtract(mesh, ca)
                            cb = _make_cutters(positions, normal, cfg,
                                               depth=cfg.depth_b, into_a=False)
                            mesh = _boolean_subtract(mesh, cb)
                    except Exception as e:
                        print(f"Dowel error {p.label}: {e}")

                elif self.do_pins and joint_type == 'flat':
                    try:
                        bounds = mesh.bounds
                        dims = bounds[1] - bounds[0]
                        ax = int(np.argmin(dims))
                        normal = np.zeros(3); normal[ax] = 1.0
                        origin = mesh.centroid.copy()
                        origin[ax] = bounds[1][ax]
                        pos = _find_pin_positions(mesh, normal, origin, 2,
                                                  self.pin_radius*3)
                        if pos:
                            mesh = _add_male_pins(mesh, pos, normal,
                                                  self.pin_radius, 6.0)
                    except Exception as e:
                        print(f"Pin error {p.label}: {e}")

                if self.do_numbers:
                    try:
                        mesh = add_part_number_to_mesh(mesh, idx + 1)
                    except Exception as e:
                        print(f"Number error {p.label}: {e}")

                sp = SlicedPart(mesh, (idx,0,0), p.label)
                sp.joint_type = joint_type
                return idx, sp

            parts_out = [None] * total
            with ThreadPoolExecutor(max_workers=4) as ex:
                futs = {ex.submit(process_one, (i,p)): i
                        for i,p in enumerate(self.leaves)}
                for fut in as_completed(futs):
                    idx, sp = fut.result()
                    parts_out[idx] = sp
                    with lock:
                        done_count += 1
                        self.progress.emit(done_count, total)

            parts_out = [p for p in parts_out if p is not None]

            from exports.exporter import Exporter
            exp = Exporter()
            exp.set_output_dir(self.out_dir)
            exported = exp.export_all(parts_out, fmt=self.fmt,
                joint_type='flat', joint_size=5.0, tolerance=0.3)

            self.finished.emit(len(exported), self.out_dir)

        except Exception as e:
            self.error.emit(str(e))


class SeamWorker(QThread):
    finished = pyqtSignal(object, list, list)  # scores_rgba, crease_lines, suggestions
    error    = pyqtSignal(str)

    def __init__(self, mesh):
        super().__init__()
        self._mesh = mesh

    def run(self):
        try:
            from core.seam_analysis import (compute_seam_scores, scores_to_rgba,
                                             find_natural_seams, suggest_cut_positions)
            scores   = compute_seam_scores(self._mesh)
            rgba     = scores_to_rgba(scores)
            seams    = find_natural_seams(self._mesh)
            suggests = suggest_cut_positions(self._mesh)
            self.finished.emit(rgba, seams, suggests)
        except Exception as e:
            self.error.emit(str(e))


class SmoothWorker(QThread):
    finished = pyqtSignal(list)   # list of (part_id, smoothed_mesh, stats)
    progress = pyqtSignal(int, int)
    error    = pyqtSignal(str)

    def __init__(self, parts, method, iterations, strength, preserve_cuts):
        super().__init__()
        self.parts         = parts
        self.method        = method
        self.iterations    = iterations
        self.strength      = strength
        self.preserve_cuts = preserve_cuts

    def run(self):
        try:
            from core.mesh_smoother import smooth_part
            from concurrent.futures import ThreadPoolExecutor, as_completed
            import threading

            results = []
            done = 0
            total = len(self.parts)
            lock = threading.Lock()

            def do_one(part):
                smoothed, stats = smooth_part(
                    part.mesh,
                    method=self.method,
                    iterations=self.iterations,
                    strength=self.strength,
                    preserve_cut_faces=self.preserve_cuts)
                return part.id, smoothed, stats

            with ThreadPoolExecutor(max_workers=4) as ex:
                futs = {ex.submit(do_one, p): p for p in self.parts}
                for fut in as_completed(futs):
                    pid, smoothed, stats = fut.result()
                    results.append((pid, smoothed, stats))
                    with lock:
                        done += 1
                        self.progress.emit(done, total)

            self.finished.emit(results)

        except Exception as e:
            self.error.emit(str(e))


class BambuExportWorker(QThread):
    finished = pyqtSignal(bool, str, list)
    error    = pyqtSignal(str)

    def __init__(self, leaves, output_path, plate_w, plate_h):
        super().__init__()
        self.leaves      = leaves
        self.output_path = output_path
        self.plate_w     = plate_w
        self.plate_h     = plate_h

    def run(self):
        try:
            from exports.bambu_export import export_bambu_3mf
            meshes = [p.mesh for p in self.leaves]
            labels = [p.label for p in self.leaves]
            # Extract colours from parts (if available from colour split)
            colours = []
            for p in self.leaves:
                c = getattr(p, '_source_colour', None)
                colours.append(c)
            has_colours = any(c is not None for c in colours)
            ok, msg, plates = export_bambu_3mf(
                meshes, labels, self.output_path,
                self.plate_w, self.plate_h, auto_orient=True,
                part_colours=colours if has_colours else None)
            self.finished.emit(ok, msg, plates)
        except Exception as e:
            self.error.emit(str(e))


class PdfExportWorker(QThread):
    finished = pyqtSignal(bool, str)
    error    = pyqtSignal(str)

    def __init__(self, leaves, output_path, plate_assignments,
                 project_name, printer_name,
                 total_filament_g=0.0, total_time_str="—",
                 material="PETG"):
        super().__init__()
        self.leaves           = leaves
        self.output_path      = output_path
        self.plate_assignments = plate_assignments
        self.project_name     = project_name
        self.printer_name     = printer_name
        self.total_filament_g = total_filament_g
        self.total_time_str   = total_time_str
        self.material         = material

    def run(self):
        try:
            from exports.assembly_pdf import generate_assembly_pdf
            ok = generate_assembly_pdf(
                self.output_path,
                self.leaves,
                self.plate_assignments,
                project_name=self.project_name,
                printer_name=self.printer_name,
                total_filament_g=self.total_filament_g,
                total_time_str=self.total_time_str,
                material=self.material)
            msg = self.output_path if ok else "PDF generation failed"
            self.finished.emit(ok, msg)
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Print Slicer — 3D App for 3D Printer")
        self.resize(1500, 940)
        self.mesh_handler = MeshHandler()
        self.part_tree = PartTree()
        self.exporter = Exporter()
        self._lock_signals = False
        self._measure_mode = False
        self._measure_points = []
        self._original_title = "3D Print Slicer — 3D App for 3D Printer"
        self._hover_highlight_id = None
        self._hover_was_hidden = False
        self._hover_timer = QTimer()
        self._hover_timer.setSingleShot(True)
        self._hover_timer.timeout.connect(self._end_hover_preview)
        self._build_ui()
        self._apply_theme()
        self._connect_signals()
        self._update_cut_preview()
        # Initial workflow stage + context bar
        self._set_workflow_stage('import')
        self._update_context_bar()
        # Enable drag-and-drop
        self.setAcceptDrops(True)

    # ═══════════════════════════════════════════════════════
    # BUILD UI
    # ═══════════════════════════════════════════════════════

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0,0,0,0); root.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        # ── Left sidebar ──────────────────────────────────
        left = QWidget(); left.setFixedWidth(360); left.setObjectName("leftPanel")
        ll = QVBoxLayout(left); ll.setContentsMargins(0,0,0,0); ll.setSpacing(0)

        title_bar = QWidget(); title_bar.setObjectName("titleBar"); title_bar.setFixedHeight(52)
        tb = QHBoxLayout(title_bar); tb.setContentsMargins(14,0,14,0); tb.setSpacing(8)
        tl = QLabel("⬡ 3D PRINT SLICER"); tl.setObjectName("appTitle"); tb.addWidget(tl); tb.addStretch()
        self.btn_save = QPushButton("Save"); self.btn_load = QPushButton("Load")
        self.btn_undo = QPushButton("↩ Undo")
        for b in [self.btn_save, self.btn_load, self.btn_undo]:
            b.setObjectName("toolbarBtn"); b.setFixedHeight(26); tb.addWidget(b)
        self.btn_undo.setEnabled(False)
        ll.addWidget(title_bar)

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame); scroll.setObjectName("sidebarScroll")
        sidebar = QWidget(); sidebar.setObjectName("sidebarContent")
        self._sl = QVBoxLayout(sidebar); self._sl.setContentsMargins(10,10,10,10); self._sl.setSpacing(8)

        # Parts tree at the top (collapsible)
        self._build_parts_tree_section()

        # Workflow progress indicator
        self._build_workflow_indicator()

        self._build_import_section()
        self._build_resize_section()
        self._build_repair_section()
        self._build_smoothing_section()
        self._build_auto_slice_section()
        self._build_seam_section()
        self._build_cut_section()
        self._build_joints_section()
        self._build_hollow_section()
        self._build_export_section()
        self._sl.addStretch()

        scroll.setWidget(sidebar); ll.addWidget(scroll)
        splitter.addWidget(left)

        # ── Right panel ───────────────────────────────────
        right = QWidget(); right.setObjectName("rightPanel")
        rl = QVBoxLayout(right); rl.setContentsMargins(0,0,0,0); rl.setSpacing(0)

        # Toolbar
        toolbar = QWidget(); toolbar.setObjectName("toolbar"); toolbar.setFixedHeight(40)
        tb2 = QHBoxLayout(toolbar); tb2.setContentsMargins(12,0,12,0); tb2.setSpacing(8)
        self.lbl_status = QLabel("No model loaded"); self.lbl_status.setObjectName("statusLabel")
        tb2.addWidget(self.lbl_status); tb2.addStretch()
        self.chk_wireframe = QCheckBox("Wireframe")
        self.chk_sel_wire = QCheckBox("Sel Wire")
        self.chk_sel_wire.setToolTip("Show wireframe overlay on the selected part")
        self.chk_preview = QCheckBox("Preview Cut"); self.chk_preview.setChecked(True)
        self.btn_reset_view = QPushButton("Reset View"); self.btn_reset_view.setObjectName("toolbarBtn")
        self.btn_workflow = QPushButton("? Guide"); self.btn_workflow.setObjectName("toolbarBtn")
        self.btn_workflow.setToolTip("Show step-by-step workflow guide")

        # Explode slider
        tb2.addWidget(QLabel("Explode:"))
        self.slider_explode = QSlider(Qt.Horizontal)
        self.slider_explode.setRange(0, 100); self.slider_explode.setValue(0)
        self.slider_explode.setFixedWidth(100)
        self.slider_explode.setToolTip("Pull parts apart to inspect assembly (0%=assembled, 100%=exploded)")
        tb2.addWidget(self.slider_explode)

        # Snap angle selector
        tb2.addWidget(QLabel("Snap:"))
        self.combo_snap = QComboBox()
        self.combo_snap.addItems(["5°", "10°", "15°", "30°", "45°", "Off"])
        self.combo_snap.setCurrentIndex(2)  # default 15°
        self.combo_snap.setFixedSize(55, 26)
        self.combo_snap.setToolTip("Rotation ring snap angle.\nHold Shift for free rotation.")
        tb2.addWidget(self.combo_snap)

        self.chk_build_vol = QCheckBox("Build Vol")
        self.chk_build_vol.setToolTip("Show/hide printer build volume outline in viewport")
        for w in [self.chk_wireframe, self.chk_sel_wire, self.chk_preview, self.chk_build_vol, self.btn_reset_view, self.btn_workflow]:
            tb2.addWidget(w)
        rl.addWidget(toolbar)

        # ── Fixed-view preset row ─────────────────────────────
        # Lets the user snap the camera to orthographic presets (like Fusion 360 /
        # Blender numpad). Combines well with hold-R-to-rotate-object: lock the
        # camera to Front, then rotate the part itself.
        view_bar = QWidget(); view_bar.setObjectName("viewBar"); view_bar.setFixedHeight(30)
        vb = QHBoxLayout(view_bar); vb.setContentsMargins(10, 2, 10, 2); vb.setSpacing(4)
        lbl_v = QLabel("View"); lbl_v.setObjectName("viewBarLabel")
        vb.addWidget(lbl_v)
        self._view_buttons = {}
        for key, label, tip in [
            ('front',  'Front',  'Look along +Y (front)'),
            ('back',   'Back',   'Look along -Y (back)'),
            ('left',   'Left',   'Look along +X (left)'),
            ('right',  'Right',  'Look along -X (right)'),
            ('top',    'Top',    'Look straight down (top)'),
            ('bottom', 'Bot',    'Look straight up (bottom)'),
            ('iso',    'Iso',    'Isometric 3/4 view'),
        ]:
            b = QPushButton(label)
            b.setObjectName("viewPresetBtn")
            b.setFixedHeight(22); b.setFixedWidth(42)
            b.setToolTip(tip)
            b.setCheckable(True)
            b.clicked.connect(lambda _=False, k=key: self._set_view_preset(k))
            vb.addWidget(b)
            self._view_buttons[key] = b
        vb.addSpacing(10)
        # Rotate-object mode toggle — when active, LMB-drag rotates the
        # selected part around its centroid instead of orbiting the camera.
        self.btn_rotate_obj = QPushButton("Rotate Object  (hold R)")
        self.btn_rotate_obj.setObjectName("viewPresetBtn")
        self.btn_rotate_obj.setFixedHeight(22)
        self.btn_rotate_obj.setCheckable(True)
        self.btn_rotate_obj.setToolTip(
            "Toggle: drag on viewport rotates the SELECTED PART around its centroid.\n"
            "Shortcut: hold R while dragging for momentary rotate mode.")
        self.btn_rotate_obj.clicked.connect(self._toggle_rotate_object_mode)
        vb.addWidget(self.btn_rotate_obj)
        # Reset Orientation — undo any manual rotations applied via
        # rotate-object mode so parts snap back to their assembly position.
        # Enabled only when at least one part has been manually rotated.
        self.btn_reset_orient = QPushButton("Reset Orientation")
        self.btn_reset_orient.setObjectName("viewPresetBtn")
        self.btn_reset_orient.setFixedHeight(22)
        self.btn_reset_orient.setToolTip(
            "Undo all manual part rotations and snap every part back to its\n"
            "assembly position (how it was when you last sliced).\n"
            "Works per-part if one is selected, otherwise resets all.")
        if _HAS_QTA:
            self.btn_reset_orient.setIcon(_qicon('reset'))
            self.btn_reset_orient.setIconSize(QSize(13, 13))
        self.btn_reset_orient.clicked.connect(self._reset_orientation)
        vb.addWidget(self.btn_reset_orient)

        # "Normal View" — escape hatch for stuck modes. Clears rotate-object,
        # select-faces, manual dowel, measure, wireframe overlays, heatmap
        # and restores the solid-surface view. ESC does the same thing.
        self.btn_normal_view = QPushButton("Normal View  (Esc)")
        self.btn_normal_view.setObjectName("viewPresetBtn")
        self.btn_normal_view.setFixedHeight(22)
        self.btn_normal_view.setToolTip(
            "Get me out of whatever mode I'm in:\n"
            "  • Exits rotate-object / select-faces / manual-dowel / measure\n"
            "  • Turns off wireframe + selection-wireframe overlays\n"
            "  • Clears thin-wall heatmap\n"
            "  • Restores solid shading on every part\n"
            "Keyboard: Esc")
        self.btn_normal_view.clicked.connect(self._reset_all_interaction_modes)
        vb.addWidget(self.btn_normal_view)

        vb.addStretch()
        # Mode indicator — always visible so you can see what the LMB drag
        # will do before you do it. Colour-coded: grey=normal, amber=special.
        self.lbl_mode = QLabel("MODE: Normal")
        self.lbl_mode.setObjectName("viewBarMode")
        self.lbl_mode.setToolTip("Current viewport interaction mode. Press Esc to return to Normal.")
        vb.addWidget(self.lbl_mode)
        # Persist cam hint label — tells you what mode you're in
        self.lbl_view_hint = QLabel("")
        self.lbl_view_hint.setObjectName("viewBarHint")
        vb.addWidget(self.lbl_view_hint)
        rl.addWidget(view_bar)

        # Context bar — adaptive action buttons based on selection
        self._build_context_bar(rl)

        # Viewport takes the full right side now (tree is in the sidebar)
        self.viewport = Viewport3D()
        rl.addWidget(self.viewport, 1)

        # Quick-cut toolbar (always visible below viewport)
        cut_bar = QWidget(); cut_bar.setObjectName("cutBar"); cut_bar.setFixedHeight(38)
        cb_lay = QHBoxLayout(cut_bar); cb_lay.setContentsMargins(8,0,8,0); cb_lay.setSpacing(6)
        cb_lay.addWidget(QLabel("Quick Cut:"))
        self.combo_qc_axis = QComboBox(); self.combo_qc_axis.addItems(["X","Y","Z"])
        self.combo_qc_axis.setFixedSize(45, 26)
        cb_lay.addWidget(self.combo_qc_axis)
        self.spin_qc_pos = QDoubleSpinBox(); self.spin_qc_pos.setRange(-9999,9999)
        self.spin_qc_pos.setSuffix(" mm"); self.spin_qc_pos.setDecimals(1)
        self.spin_qc_pos.setSingleStep(5); self.spin_qc_pos.setFixedSize(110, 26)
        cb_lay.addWidget(self.spin_qc_pos)
        self.btn_qc_m10 = QPushButton("-10"); self.btn_qc_m10.setObjectName("nudgeBtn"); self.btn_qc_m10.setFixedSize(32,26)
        self.btn_qc_m1  = QPushButton("-1");  self.btn_qc_m1.setObjectName("nudgeBtn");  self.btn_qc_m1.setFixedSize(28,26)
        self.btn_qc_p1  = QPushButton("+1");  self.btn_qc_p1.setObjectName("nudgeBtn");  self.btn_qc_p1.setFixedSize(28,26)
        self.btn_qc_p10 = QPushButton("+10"); self.btn_qc_p10.setObjectName("nudgeBtn"); self.btn_qc_p10.setFixedSize(32,26)
        for b in [self.btn_qc_m10, self.btn_qc_m1, self.btn_qc_p1, self.btn_qc_p10]:
            cb_lay.addWidget(b)
        self.combo_qc_mode = QComboBox(); self.combo_qc_mode.addItems(["Full","Angled","Section","Groove","Natural"])
        self.combo_qc_mode.setFixedSize(70, 26)
        cb_lay.addWidget(self.combo_qc_mode)
        self.btn_qc_cut = QPushButton("Cut"); self.btn_qc_cut.setObjectName("cutBtn")
        self._set_btn_icon(self.btn_qc_cut, 'cut', color='#e0ffe8')
        self.btn_qc_cut.setFixedSize(70, 28); self.btn_qc_cut.setEnabled(False)
        cb_lay.addWidget(self.btn_qc_cut)
        cb_lay.addStretch()
        rl.addWidget(cut_bar)

        # Bottom tabs
        self.bottom_tabs = QTabWidget(); self.bottom_tabs.setObjectName("bottomTabs"); self.bottom_tabs.setFixedHeight(170)
        self._build_parts_tab()
        self._build_estimate_tab()
        self._build_history_tab()
        self._build_help_tab()
        rl.addWidget(self.bottom_tabs)

        splitter.addWidget(right)
        splitter.setSizes([360, 1140])

        self.status_bar = QStatusBar(); self.status_bar.setObjectName("statusBar"); self.setStatusBar(self.status_bar)
        self.progress_bar = QProgressBar(); self.progress_bar.setVisible(False); self.progress_bar.setFixedWidth(200)
        self.status_bar.addPermanentWidget(self.progress_bar)

    def _build_context_bar(self, parent_layout):
        """Adaptive action bar — shows buttons relevant to current selection.
        Sits between the main toolbar and the viewport.
        """
        self.context_bar = QWidget()
        self.context_bar.setObjectName("contextBar")
        self.context_bar.setFixedHeight(38)
        self._ctx_layout = QHBoxLayout(self.context_bar)
        self._ctx_layout.setContentsMargins(8, 3, 8, 3)
        self._ctx_layout.setSpacing(4)

        # Context label — tells the user what's active
        self.lbl_context = QLabel("No model loaded")
        self.lbl_context.setObjectName("contextLabel")
        self._ctx_layout.addWidget(self.lbl_context)
        self._ctx_layout.addSpacing(12)

        # Buttons are created dynamically in _update_context_bar
        self._ctx_buttons = []

        self._ctx_layout.addStretch()
        parent_layout.addWidget(self.context_bar)

    def _set_btn_icon(self, btn, icon_name, color='#c0d4f0'):
        """Attach a qtawesome icon to an existing button (or prepend emoji fallback)."""
        if _HAS_QTA:
            btn.setIcon(_qicon(icon_name, color=color))
            btn.setIconSize(QSize(14, 14))
        else:
            emoji = _ICONS.get(icon_name, ('', ''))[1]
            if emoji:
                btn.setText(f"{emoji} {btn.text()}")
        return btn

    def _make_ctx_button(self, label, tooltip, callback, highlight=False, icon=None):
        """Create a context bar button with an optional qtawesome icon."""
        if icon:
            btn = QPushButton(_icon_label(icon, label))
            if _HAS_QTA:
                color = '#e0ffe8' if highlight else '#c0d4f0'
                btn.setIcon(_qicon(icon, color=color))
                btn.setIconSize(QSize(14, 14))
        else:
            btn = QPushButton(label)
        btn.setObjectName("ctxPrimaryBtn" if highlight else "ctxBtn")
        btn.setFixedHeight(28)
        btn.setToolTip(tooltip)
        btn.clicked.connect(callback)
        return btn

    def _clear_ctx_buttons(self):
        """Remove all dynamic context bar buttons."""
        for btn in self._ctx_buttons:
            try:
                btn.setParent(None)
                btn.deleteLater()
            except Exception:
                pass
        self._ctx_buttons = []

    def _update_context_bar(self):
        """Rebuild the context bar based on current state."""
        if not hasattr(self, '_ctx_layout'):
            return
        self._clear_ctx_buttons()

        has_mesh = self.mesh_handler.mesh is not None
        leaves = self.part_tree.get_all_leaves() if self.part_tree.root else []
        has_parts = len(leaves) > 1
        sel = self.part_tree.selected_part
        sel_is_leaf = sel is not None and sel.is_leaf

        # --- Build context label ---
        if not has_mesh:
            self.lbl_context.setText("No model loaded")
        elif sel_is_leaf:
            self.lbl_context.setText(f"Part: {sel.label}")
        elif has_parts:
            self.lbl_context.setText(f"Model: {len(leaves)} parts")
        else:
            self.lbl_context.setText("Model loaded")

        # --- Build buttons for the current context ---
        # The 'insertion point' is just before the stretch — index = len(ctx items) - 1
        def add(btn):
            self._ctx_buttons.append(btn)
            # Insert before the stretch (last item)
            self._ctx_layout.insertWidget(self._ctx_layout.count() - 1, btn)

        if not has_mesh:
            # Empty state — offer import actions
            add(self._make_ctx_button("Import Model…",
                "Import STL/OBJ/3MF file",
                self._import_model, highlight=True, icon='import'))
            if hasattr(self, 'btn_ai_generate'):
                add(self._make_ctx_button("From Photo…",
                    "Generate 3D from photo using AI",
                    self._ai_generate, icon='photo'))
            if hasattr(self, 'btn_text_to_3d'):
                add(self._make_ctx_button("From Text…",
                    "Generate 3D from text description",
                    self._text_to_3d, icon='text'))
            return

        if sel_is_leaf:
            # A specific part is selected — show part-specific actions
            add(self._make_ctx_button("Cut Part",
                "Cut this part — opens Quick Cut with this part targeted",
                lambda: self._ctx_cut_part(), icon='cut'))
            add(self._make_ctx_button("Repair",
                "Run auto-repair on this part only",
                lambda: self._run_full_repair(), icon='repair'))
            add(self._make_ctx_button("Set Joint",
                "Configure connector joint for this part",
                lambda: self._ctx_open_joints(), icon='magnet'))
            add(self._make_ctx_button("Flat Down",
                "Orient this part flat-face-down for printing",
                lambda: self._ctx_orient_part(), icon='flat'))
            # Show Adjacent — hides all parts except this one + neighbours it
            # shares a cut-face with. Toggle label reflects current mode.
            adj_on = getattr(self, '_adjacent_mode', False)
            add(self._make_ctx_button(
                "Hide Others" if adj_on else "Show Adjacent",
                "Show only this part plus the parts whose cut-faces touch it.\n"
                "Use Explode to pull them apart for inspection.",
                lambda: self._ctx_toggle_adjacent(), icon='adjacent'))
            add(self._make_ctx_button("Hide",
                "Hide this part in the viewport",
                lambda: self._ctx_hide_selected(), icon='eye'))

        elif has_parts:
            # Have parts but nothing selected — offer bulk actions
            add(self._make_ctx_button("Export All",
                "Export all parts as STL/OBJ/3MF files",
                self._export_parts, highlight=True, icon='save'))
            if hasattr(self, 'btn_bambu_export'):
                add(self._make_ctx_button("Bambu 3MF",
                    "Export as Bambu Studio 3MF with plates packed",
                    self._export_bambu, icon='box'))
            add(self._make_ctx_button("Assembly PDF",
                "Generate assembly guide PDF",
                self._export_pdf, icon='pdf'))
            add(self._make_ctx_button("Orient All",
                "Auto-orient every part flat-face-down",
                self._orient_all_parts, icon='rotate'))

        else:
            # Model loaded but not sliced yet
            add(self._make_ctx_button("Auto-Slice",
                "Automatically cut into printer-sized pieces",
                self._auto_slice, highlight=True, icon='cut'))
            add(self._make_ctx_button("Resize",
                "Change model dimensions",
                lambda: self._ctx_open_section('resize'), icon='resize'))
            add(self._make_ctx_button("Repair",
                "Fix mesh problems",
                lambda: self._run_full_repair(), icon='repair'))
            add(self._make_ctx_button("Seam Heatmap",
                "Analyse where cuts will be hidden",
                lambda: self._show_seam_heatmap() if hasattr(self, '_show_seam_heatmap') else None,
                icon='heatmap'))

    def _ctx_cut_part(self):
        """Scroll focus to the quick-cut bar — cut mode for this part."""
        self.spin_qc_pos.setFocus()
        self.status_bar.showMessage("Set cut position in the Quick Cut bar below viewport, then click Cut.")

    def _ctx_open_joints(self):
        """Expand and focus the Joints section."""
        if hasattr(self, '_sections'):
            sec = self._sections.get('joints')
            if sec and sec._header.isEnabled():
                sec.expand()

    def _ctx_open_section(self, key):
        """Expand a named sidebar section."""
        if hasattr(self, '_sections'):
            sec = self._sections.get(key)
            if sec and sec._header.isEnabled():
                sec.expand()

    def _ctx_orient_part(self):
        """Orient just the selected part flat-face-down."""
        sel = self.part_tree.selected_part
        if sel is None or not sel.is_leaf: return
        self._context_orient(sel) if hasattr(self, '_context_orient') else None

    def _ctx_hide_selected(self):
        """Hide the selected part."""
        sel = self.part_tree.selected_part
        if sel and sel.is_leaf:
            sel.visible = False
            self._refresh_tree(); self._refresh_viewport()

    def _build_parts_tree_section(self):
        """Parts tree as a collapsible section at the top of the sidebar."""
        sec, lay = self._section("Parts", expanded=True, key='parts_tree')
        # Tighter layout for the tree
        lay.setContentsMargins(6, 4, 6, 6); lay.setSpacing(4)

        # Visibility button row — All / None / Solo
        btn_row = QWidget(); brl = QHBoxLayout(btn_row)
        brl.setContentsMargins(0, 0, 0, 0); brl.setSpacing(4)
        self.btn_show_all = QPushButton("All")
        self.btn_show_all.setObjectName("visBtn"); self.btn_show_all.setFixedHeight(22)
        self.btn_hide_all = QPushButton("None")
        self.btn_hide_all.setObjectName("visBtn"); self.btn_hide_all.setFixedHeight(22)
        self.btn_solo = QPushButton("Solo")
        self.btn_solo.setObjectName("visSoloBtn"); self.btn_solo.setFixedHeight(22)
        self.btn_solo.setToolTip("Show only the selected part")
        for b in [self.btn_show_all, self.btn_hide_all, self.btn_solo]:
            brl.addWidget(b, 1)
        lay.addWidget(btn_row)

        # Tree widget
        self.parts_tree_widget = QTreeWidget()
        self.parts_tree_widget.setHeaderHidden(True)
        self.parts_tree_widget.setObjectName("partsTreeWidget")
        self.parts_tree_widget.setColumnCount(3)  # label | wireframe | eye
        self.parts_tree_widget.header().setStretchLastSection(False)
        self.parts_tree_widget.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.parts_tree_widget.header().setSectionResizeMode(1, QHeaderView.Fixed)
        self.parts_tree_widget.header().setSectionResizeMode(2, QHeaderView.Fixed)
        self.parts_tree_widget.setColumnWidth(1, 26)
        self.parts_tree_widget.setColumnWidth(2, 26)
        self.parts_tree_widget.setIndentation(12)
        self.parts_tree_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.parts_tree_widget.setMouseTracking(True)
        # Fixed reasonable height — scrolls if more parts than fit
        self.parts_tree_widget.setMinimumHeight(140)
        self.parts_tree_widget.setMaximumHeight(280)
        lay.addWidget(self.parts_tree_widget)

        # Selected part info
        self.lbl_part_info = QLabel("No part selected")
        self.lbl_part_info.setObjectName("dimLabel"); self.lbl_part_info.setWordWrap(True)
        self.lbl_part_info.setContentsMargins(4, 2, 4, 2)
        lay.addWidget(self.lbl_part_info)

    def _build_workflow_indicator(self):
        """Horizontal progress bar at top of sidebar showing workflow stages."""
        container = QWidget()
        container.setObjectName("workflowBar")
        container.setFixedHeight(56)
        vl = QVBoxLayout(container); vl.setContentsMargins(6, 4, 6, 4); vl.setSpacing(2)

        # Stage label
        self.lbl_workflow_stage = QLabel("→ Start by importing a model")
        self.lbl_workflow_stage.setObjectName("workflowStageLabel")
        self.lbl_workflow_stage.setAlignment(Qt.AlignCenter)
        vl.addWidget(self.lbl_workflow_stage)

        # Stage dots row
        dots_row = QWidget()
        dl = QHBoxLayout(dots_row); dl.setContentsMargins(0,0,0,0); dl.setSpacing(4)
        self._workflow_stages = ['import', 'prepare', 'slice', 'connect', 'export']
        stage_names = ['Import', 'Prepare', 'Slice', 'Connect', 'Export']
        self._workflow_dots = []
        for i, (key, name) in enumerate(zip(self._workflow_stages, stage_names)):
            dot = QPushButton(name)
            dot.setObjectName("workflowDot")
            dot.setProperty('stage', 'locked' if i > 0 else 'active')
            dot.setFixedHeight(22)
            dot.setCheckable(False)
            # Clicking a dot jumps to that stage (if available)
            dot.clicked.connect(lambda _, k=key: self._jump_to_stage(k))
            dl.addWidget(dot, 1)
            self._workflow_dots.append(dot)
        vl.addWidget(dots_row)
        self._sl.addWidget(container)

        # Current stage — starts at 'import'
        self._current_stage = 'import'

    def _set_workflow_stage(self, stage):
        """Update the active stage + enable/disable sidebar sections."""
        if stage not in self._workflow_stages:
            return
        self._current_stage = stage

        # Update dots
        stage_idx = self._workflow_stages.index(stage)
        for i, dot in enumerate(self._workflow_dots):
            if i < stage_idx:
                dot.setProperty('stage', 'done')
            elif i == stage_idx:
                dot.setProperty('stage', 'active')
            else:
                dot.setProperty('stage', 'locked' if not self._stage_available(i) else 'available')
            dot.style().unpolish(dot); dot.style().polish(dot)

        # Update stage label
        labels = {
            'import':  "① Start by importing a model or generating from photo",
            'prepare': "② Prepare mesh — repair, resize, improve quality",
            'slice':   "③ Slice into printable pieces",
            'connect': "④ Add connectors to align parts during assembly",
            'export':  "⑤ Export for printing",
        }
        self.lbl_workflow_stage.setText(labels.get(stage, ''))

        # Enable/disable sidebar sections based on stage
        self._update_sidebar_visibility()
        # Refresh the context bar for the new state
        self._update_context_bar()

    def _stage_available(self, stage_idx):
        """Check whether a stage is accessible given current app state."""
        has_mesh = self.mesh_handler.mesh is not None
        has_parts = bool(self.part_tree.root and self.part_tree.get_all_leaves()
                         and len(self.part_tree.get_all_leaves()) > 1)
        # 0=import always, 1=prepare needs mesh, 2=slice needs mesh,
        # 3=connect needs parts, 4=export needs parts
        if stage_idx == 0: return True
        if stage_idx in (1, 2): return has_mesh
        if stage_idx in (3, 4): return has_parts
        return False

    def _update_sidebar_visibility(self):
        """Enable/disable sidebar sections based on current workflow stage."""
        if not hasattr(self, '_sections'): return
        has_mesh = self.mesh_handler.mesh is not None
        has_parts = bool(self.part_tree.root and
                         len(self.part_tree.get_all_leaves()) > 1)

        # Section → required state (parts_tree always available)
        availability = {
            'parts_tree': True,
            'import':  True,
            'resize':  has_mesh,
            'repair':  has_mesh,
            'smooth':  has_mesh,
            'slice':   has_mesh,
            'seam':    has_mesh,
            'cut':     has_mesh,
            'joints':  has_parts,
            'hollow':  has_parts,
            'export':  has_parts,
        }
        for key, available in availability.items():
            sec = self._sections.get(key)
            if sec:
                sec.set_stage_state('available' if available else 'locked')

    def _jump_to_stage(self, stage):
        """User clicked a workflow dot — expand relevant section."""
        if not hasattr(self, '_sections'): return
        # Which section to expand for each stage
        stage_section = {
            'import':  'import',
            'prepare': 'repair',
            'slice':   'slice',
            'connect': 'joints',
            'export':  'export',
        }
        target = stage_section.get(stage)
        if target and target in self._sections:
            sec = self._sections[target]
            if hasattr(sec, '_header') and sec._header.isEnabled():
                sec.expand()

    def _section(self, title, expanded=False, key=None):
        sec = CollapsibleSection(title, expanded=expanded)
        self._sl.addWidget(sec)
        if key:
            if not hasattr(self, '_sections'):
                self._sections = {}
            self._sections[key] = sec
        return sec, sec.content_layout()

    def _build_import_section(self):
        _, lay = self._section("1 · Import Model", expanded=True, key='import')

        # Printer profile
        pr = QWidget(); prl = QHBoxLayout(pr); prl.setContentsMargins(0,0,0,0); prl.setSpacing(6)
        prl.addWidget(QLabel("Printer:"))
        self.combo_printer = QComboBox()
        from core.printer_profiles import profile_names, DEFAULT_PROFILE
        self.combo_printer.addItems(profile_names())
        self.combo_printer.setCurrentText(DEFAULT_PROFILE)
        self.combo_printer.setFixedHeight(28)
        self.combo_printer.setToolTip("Sets the build volume for auto-slice cut sizing and Bambu export plate packing")
        prl.addWidget(self.combo_printer)
        lay.addWidget(pr)

        # Custom size row (shown when Custom selected)
        self.custom_size_row = QWidget()
        csl = QHBoxLayout(self.custom_size_row); csl.setContentsMargins(0,0,0,0); csl.setSpacing(4)
        csl.addWidget(QLabel("W:"))
        self.spin_custom_x = QDoubleSpinBox(); self.spin_custom_x.setRange(50,1000); self.spin_custom_x.setValue(256); self.spin_custom_x.setSuffix("mm"); self.spin_custom_x.setFixedHeight(26)
        csl.addWidget(self.spin_custom_x)
        csl.addWidget(QLabel("D:"))
        self.spin_custom_y = QDoubleSpinBox(); self.spin_custom_y.setRange(50,1000); self.spin_custom_y.setValue(256); self.spin_custom_y.setSuffix("mm"); self.spin_custom_y.setFixedHeight(26)
        csl.addWidget(self.spin_custom_y)
        csl.addWidget(QLabel("H:"))
        self.spin_custom_z = QDoubleSpinBox(); self.spin_custom_z.setRange(50,1000); self.spin_custom_z.setValue(256); self.spin_custom_z.setSuffix("mm"); self.spin_custom_z.setFixedHeight(26)
        csl.addWidget(self.spin_custom_z)
        self.custom_size_row.setVisible(False)
        lay.addWidget(self.custom_size_row)

        self.lbl_printer_info = QLabel("")
        self.lbl_printer_info.setObjectName("dimLabel"); lay.addWidget(self.lbl_printer_info)

        self.btn_import = QPushButton("Import STL / OBJ / 3MF…")
        self.btn_import.setObjectName("primaryBtn"); self.btn_import.setFixedHeight(36)
        lay.addWidget(self.btn_import)

        # AI Generate from Photo
        self.btn_ai_generate = QPushButton("Generate 3D from Photo…")
        self._set_btn_icon(self.btn_ai_generate, 'photo')
        self.btn_ai_generate.setObjectName("colourBtn"); self.btn_ai_generate.setFixedHeight(34)
        self.btn_ai_generate.setToolTip(
            "Take a photo of any object and generate a 3D model.\n"
            "Uses AI (SAM 3D / TripoSR) running locally on your GPU.\n"
            "Supports 1-4 photos for better accuracy.\n"
            "Run setup_ai_backends.bat first to install.")
        lay.addWidget(self.btn_ai_generate)

        # Backend selector
        ai_row = QWidget(); airl = QHBoxLayout(ai_row); airl.setContentsMargins(0,0,0,0); airl.setSpacing(4)
        airl.addWidget(QLabel("AI Backend:"))
        self.combo_ai_backend = QComboBox()
        self.combo_ai_backend.addItems(["Auto (best available)",
                                         "TRELLIS 2 (best quality)",
                                         "SAM 3D Objects (excellent)",
                                         "PartCrafter (pre-separated parts)",
                                         "TripoSR (fastest)"])
        self.combo_ai_backend.setFixedHeight(24)
        airl.addWidget(self.combo_ai_backend)
        lay.addWidget(ai_row)

        # Text-to-3D
        self.btn_text_to_3d = QPushButton("Generate 3D from Text…")
        self._set_btn_icon(self.btn_text_to_3d, 'text')
        self.btn_text_to_3d.setObjectName("secondaryBtn"); self.btn_text_to_3d.setFixedHeight(30)
        self.btn_text_to_3d.setToolTip(
            "Describe a shape in text and generate a 3D model.\n"
            "Requires OpenSCAD installed (openscad.org).\n"
            "No GPU needed — runs on CPU.\n\n"
            "Examples:\n"
            "  'box 100x80x60 with rounded corners'\n"
            "  'cylinder 50mm diameter 100mm tall'\n"
            "  'sphere 40mm'")
        lay.addWidget(self.btn_text_to_3d)

        self.lbl_ai_status = QLabel(""); self.lbl_ai_status.setObjectName("dimLabel"); self.lbl_ai_status.setWordWrap(True)
        lay.addWidget(self.lbl_ai_status)

        self.lbl_file = QLabel("No file loaded"); self.lbl_file.setObjectName("dimLabel"); self.lbl_file.setWordWrap(True)
        self.lbl_mesh_info = QLabel(""); self.lbl_mesh_info.setObjectName("dimLabel"); self.lbl_mesh_info.setWordWrap(True)
        lay.addWidget(self.lbl_file); lay.addWidget(self.lbl_mesh_info)

        # Colour split
        self.btn_colour_split = QPushButton("🎨 Split by Colour / Material")
        self.btn_colour_split.setObjectName("colourBtn"); self.btn_colour_split.setFixedHeight(32)
        self.btn_colour_split.setEnabled(False)
        self.btn_colour_split.setToolTip("Split into separate parts based on colour/material.")
        lay.addWidget(self.btn_colour_split)

        # Colour tolerance
        ct_row = QWidget(); ctl = QHBoxLayout(ct_row); ctl.setContentsMargins(0,0,0,0); ctl.setSpacing(6)
        ctl.addWidget(QLabel("Colour tolerance:"))
        self.spin_colour_tol = QSpinBox()
        self.spin_colour_tol.setRange(5, 50); self.spin_colour_tol.setValue(25)
        self.spin_colour_tol.setFixedHeight(26)
        self.spin_colour_tol.setToolTip(
            "Lower = more colour regions (fine detail).\n"
            "Higher = fewer regions (merge similar colours).\n"
            "Default 25 works for most models.")
        ctl.addWidget(self.spin_colour_tol)
        lay.addWidget(ct_row)

        self.lbl_colour_info = QLabel(""); self.lbl_colour_info.setObjectName("dimLabel"); self.lbl_colour_info.setWordWrap(True)
        lay.addWidget(self.lbl_colour_info)

        # Update profile info label on start
        self._update_printer_info()

    def _build_resize_section(self):
        _, lay = self._section("2 · Resize", key='resize')
        def axis_row(lbl, color):
            row = QWidget(); rl = QHBoxLayout(row); rl.setContentsMargins(0,0,0,0); rl.setSpacing(6)
            l = QLabel(lbl); l.setFixedWidth(16); l.setStyleSheet(f"color:{color};font-weight:700;font-size:12px;")
            s = QDoubleSpinBox(); s.setRange(1,99999); s.setSuffix(" mm"); s.setDecimals(1); s.setSingleStep(5); s.setFixedHeight(28)
            p = QLabel("—"); p.setFixedWidth(55); p.setObjectName("pctLabel"); p.setAlignment(Qt.AlignRight|Qt.AlignVCenter)
            rl.addWidget(l); rl.addWidget(s,1); rl.addWidget(p)
            lay.addWidget(row)
            return s, p
        self.spin_x, self.lbl_pct_x = axis_row("X","#e05555")
        self.spin_y, self.lbl_pct_y = axis_row("Y","#50c870")
        self.spin_z, self.lbl_pct_z = axis_row("Z","#5588e8")
        self.lbl_orig = QLabel("Original: —"); self.lbl_orig.setObjectName("dimLabel"); lay.addWidget(self.lbl_orig)
        ur = QWidget(); ul = QHBoxLayout(ur); ul.setContentsMargins(0,0,0,0); ul.setSpacing(6)
        self.chk_uniform = QCheckBox("Uniform"); self.chk_uniform.setChecked(False)
        self.combo_lock_axis = QComboBox(); self.combo_lock_axis.addItems(["from X","from Y","from Z"]); self.combo_lock_axis.setFixedHeight(26); self.combo_lock_axis.setEnabled(False)
        ul.addWidget(self.chk_uniform); ul.addWidget(self.combo_lock_axis); ul.addStretch()
        lay.addWidget(ur)
        self.btn_apply_resize = QPushButton("Apply Resize"); self.btn_apply_resize.setObjectName("secondaryBtn"); self.btn_apply_resize.setFixedHeight(32)
        lay.addWidget(self.btn_apply_resize)

    def _build_repair_section(self):
        _, lay = self._section("3 · Mesh Repair & Health", key='repair')

        # Auto-fix on import toggle
        self.chk_auto_repair = QCheckBox("Auto-repair on import")
        self.chk_auto_repair.setChecked(False)
        self.chk_auto_repair.setToolTip(
            "Automatically run full repair pipeline when loading a model.\n"
            "Fixes normals, holes, degenerate faces, winding, and floating shells.")
        lay.addWidget(self.chk_auto_repair)

        # Health dashboard
        self.lbl_repair = QLabel("Import a model first.")
        self.lbl_repair.setObjectName("dimLabel"); self.lbl_repair.setWordWrap(True)
        lay.addWidget(self.lbl_repair)

        # Health score bar
        self.lbl_health_score = QLabel("")
        self.lbl_health_score.setObjectName("dimLabel")
        lay.addWidget(self.lbl_health_score)

        # Repair buttons row
        btn_row = QWidget(); brl = QHBoxLayout(btn_row); brl.setContentsMargins(0,0,0,0); brl.setSpacing(4)
        self.btn_repair = QPushButton("Full Auto-Repair")
        self._set_btn_icon(self.btn_repair, 'repair')
        self.btn_repair.setObjectName("secondaryBtn"); self.btn_repair.setFixedHeight(30)
        self.btn_repair.setEnabled(False)
        self.btn_repair.setToolTip(
            "Run the full repair pipeline:\n"
            "• Remove degenerate/duplicate faces\n"
            "• Fix normals and winding\n"
            "• Fill holes\n"
            "• Remove floating shells\n"
            "• Merge close vertices")
        brl.addWidget(self.btn_repair)
        self.btn_diagnose = QPushButton("Diagnose")
        self._set_btn_icon(self.btn_diagnose, 'diagnose')
        self.btn_diagnose.setObjectName("nudgeBtn"); self.btn_diagnose.setFixedHeight(30)
        self.btn_diagnose.setEnabled(False)
        self.btn_diagnose.setToolTip("Quick check for problems without fixing anything")
        brl.addWidget(self.btn_diagnose)
        lay.addWidget(btn_row)

        # Decimation
        dec_row = QWidget(); drl = QHBoxLayout(dec_row); drl.setContentsMargins(0,0,0,0); drl.setSpacing(6)
        drl.addWidget(QLabel("Simplify:"))
        self.spin_decimate = QDoubleSpinBox()
        self.spin_decimate.setRange(0.1, 0.9); self.spin_decimate.setValue(0.5)
        self.spin_decimate.setSingleStep(0.1); self.spin_decimate.setDecimals(1)
        self.spin_decimate.setFixedHeight(26)
        self.spin_decimate.setToolTip("Target ratio — 0.5 keeps 50% of triangles")
        drl.addWidget(self.spin_decimate)
        self.btn_decimate = QPushButton("Reduce (Quadric)")
        self.btn_decimate.setObjectName("nudgeBtn"); self.btn_decimate.setFixedHeight(26)
        self.btn_decimate.setEnabled(False)
        self.btn_decimate.setToolTip(
            "Classical quadric decimation — fastest, shape-preserving but\n"
            "tends to soften sharp features. Good for organic shapes.")
        drl.addWidget(self.btn_decimate)
        # Feature-preserving variant — uses pyvista.decimate_pro which
        # respects dihedral angles above 30° so logos, ribs, and chamfers
        # survive polygon reduction.
        self.btn_decimate_pro = QPushButton("Reduce (Feature-Preserving)")
        self.btn_decimate_pro.setObjectName("nudgeBtn"); self.btn_decimate_pro.setFixedHeight(26)
        self.btn_decimate_pro.setEnabled(False)
        self.btn_decimate_pro.setToolTip(
            "Feature-preserving decimation (pyvista.decimate_pro).\n"
            "Keeps sharp edges intact (feature angle ≥ 30°). Best for\n"
            "mechanical parts where details matter more than speed.")
        drl.addWidget(self.btn_decimate_pro)
        lay.addWidget(dec_row)

        # Subdivision
        sub_row = QWidget(); srl = QHBoxLayout(sub_row); srl.setContentsMargins(0,0,0,0); srl.setSpacing(6)
        srl.addWidget(QLabel("Subdivide:"))
        self.spin_subdivide = QSpinBox()
        self.spin_subdivide.setRange(1, 3); self.spin_subdivide.setValue(1)
        self.spin_subdivide.setFixedHeight(26)
        self.spin_subdivide.setToolTip("Each pass quadruples the triangle count.\n1 = 4x, 2 = 16x, 3 = 64x")
        srl.addWidget(self.spin_subdivide)
        self.btn_subdivide = QPushButton("Subdivide")
        self.btn_subdivide.setObjectName("nudgeBtn"); self.btn_subdivide.setFixedHeight(26)
        self.btn_subdivide.setEnabled(False)
        srl.addWidget(self.btn_subdivide)
        lay.addWidget(sub_row)

        # PyMeshFix deep repair
        self.btn_pymeshfix = QPushButton("Deep Repair (PyMeshFix)")
        self._set_btn_icon(self.btn_pymeshfix, 'deep')
        self.btn_pymeshfix.setObjectName("secondaryBtn"); self.btn_pymeshfix.setFixedHeight(28)
        self.btn_pymeshfix.setEnabled(False)
        self.btn_pymeshfix.setToolTip(
            "Deep repair using PyMeshFix — fixes self-intersections,\n"
            "complex topology, and problems that basic repair can't handle.\n"
            "Use after basic repair if mesh is still broken.")
        lay.addWidget(self.btn_pymeshfix)

        # Print-Ready Repair — runs the full baseline pipeline PLUS
        # aggressive passes: self-intersection cleanup via manifold3d,
        # non-manifold-edge splitting, sliver removal, and a thin-wall
        # scan. Best for models that will go straight to the printer.
        self.btn_print_ready = QPushButton("Print-Ready Repair")
        self._set_btn_icon(self.btn_print_ready, 'deep')
        self.btn_print_ready.setObjectName("primaryBtn"); self.btn_print_ready.setFixedHeight(32)
        self.btn_print_ready.setEnabled(False)
        self.btn_print_ready.setToolTip(
            "One-click print preparation. Runs the full auto-repair pipeline\n"
            "PLUS these aggressive cleanup passes:\n"
            "  • Self-intersection cleanup (manifold3d round-trip)\n"
            "  • Non-manifold-edge splitting\n"
            "  • Sliver triangle removal (aspect ratio > 60)\n"
            "  • Tiny floating-cluster removal\n"
            "  • Thin-wall scan — flags walls under 0.8 mm that will\n"
            "    under-extrude on a 0.4 mm nozzle.")
        lay.addWidget(self.btn_print_ready)

        # ── Advanced mesh-quality tools ──────────────────────
        # Thin-wall heatmap toggle — paints the mesh red/yellow/green
        # according to local wall thickness. Uses trimesh ray queries.
        self.btn_thin_heatmap = QPushButton("Thin-Wall Heatmap")
        self._set_btn_icon(self.btn_thin_heatmap, 'heatmap')
        self.btn_thin_heatmap.setObjectName("secondaryBtn")
        self.btn_thin_heatmap.setFixedHeight(28)
        self.btn_thin_heatmap.setCheckable(True)
        self.btn_thin_heatmap.setEnabled(False)
        self.btn_thin_heatmap.setToolTip(
            "Visualise wall thickness directly on the model.\n"
            "  Green  = safe (≥ 1.6 mm)\n"
            "  Yellow = caution (0.8–1.6 mm)\n"
            "  Red    = will fail on a 0.4 mm nozzle (< 0.8 mm)")
        lay.addWidget(self.btn_thin_heatmap)

        # Edge-flip optimisation — improves triangle aspect ratios by
        # swapping diagonals across near-coplanar pairs.  Purely a quality
        # pass; doesn't change mesh shape.
        self.btn_edge_flip = QPushButton("Optimise Triangle Quality")
        self._set_btn_icon(self.btn_edge_flip, 'repair')
        self.btn_edge_flip.setObjectName("secondaryBtn")
        self.btn_edge_flip.setFixedHeight(28)
        self.btn_edge_flip.setEnabled(False)
        self.btn_edge_flip.setToolTip(
            "Edge-flip optimisation — swaps triangle diagonals where\n"
            "doing so improves aspect ratios.  Respects feature edges\n"
            "(won't flip across sharp corners).  Silhouette is preserved.\n"
            "Great after decimation or heavy boolean operations.")
        lay.addWidget(self.btn_edge_flip)

        # Remeshing options
        remesh_box = QGroupBox("Mesh Quality Improvement")
        remesh_box.setObjectName("subGroup")
        rml = QVBoxLayout(remesh_box); rml.setContentsMargins(8,10,8,8); rml.setSpacing(3)

        rm1 = QWidget(); rm1l = QHBoxLayout(rm1); rm1l.setContentsMargins(0,0,0,0); rm1l.setSpacing(4)
        self.btn_isotropic = QPushButton("Isotropic Remesh")
        self.btn_isotropic.setObjectName("nudgeBtn"); self.btn_isotropic.setFixedHeight(24)
        self.btn_isotropic.setEnabled(False)
        self.btn_isotropic.setToolTip("Make all triangles even-sized — best for scans and AI meshes")
        rm1l.addWidget(self.btn_isotropic)
        self.btn_adaptive = QPushButton("Adaptive Remesh")
        self.btn_adaptive.setObjectName("nudgeBtn"); self.btn_adaptive.setFixedHeight(24)
        self.btn_adaptive.setEnabled(False)
        self.btn_adaptive.setToolTip("More detail on curves, less on flats — best for car bodies")
        rm1l.addWidget(self.btn_adaptive)
        rml.addWidget(rm1)

        # Info button for remeshing
        self.btn_remesh_info = QPushButton("ℹ What's the difference?")
        self.btn_remesh_info.setObjectName("nudgeBtn"); self.btn_remesh_info.setFixedHeight(22)
        rml.addWidget(self.btn_remesh_info)
        lay.addWidget(remesh_box)

        # Remove shells button
        self.btn_remove_shells = QPushButton("Remove Floating Shells")
        self.btn_remove_shells.setObjectName("nudgeBtn"); self.btn_remove_shells.setFixedHeight(26)
        self.btn_remove_shells.setEnabled(False)
        self.btn_remove_shells.setToolTip("Remove small disconnected pieces (debris, internal geometry)")
        lay.addWidget(self.btn_remove_shells)

        # Individual repair steps (each undoable separately)
        indiv_box = QGroupBox("Individual Repairs (each can be undone)")
        indiv_box.setObjectName("subGroup")
        ivl = QVBoxLayout(indiv_box); ivl.setContentsMargins(8,10,8,8); ivl.setSpacing(3)
        self._repair_buttons = {}
        repair_steps = [
            ("fix_degenerate", "Remove Degenerate Faces", "Remove zero-area triangles"),
            ("fix_duplicates", "Remove Duplicate Faces", "Remove identical overlapping triangles"),
            ("fix_normals", "Fix Normals & Winding", "Ensure all faces point outward consistently"),
            ("fix_holes", "Fill Holes", "Close gaps in the mesh surface"),
            ("fix_merge", "Merge Close Vertices", "Snap near-duplicate vertices together"),
        ]
        for key, label, tip in repair_steps:
            btn = QPushButton(label)
            btn.setObjectName("nudgeBtn"); btn.setFixedHeight(24)
            btn.setEnabled(False)
            btn.setToolTip(tip)
            ivl.addWidget(btn)
            self._repair_buttons[key] = btn
        lay.addWidget(indiv_box)

        # Region-based repair
        region_box = QGroupBox("Region Select & Repair")
        region_box.setObjectName("subGroup")
        rgl = QVBoxLayout(region_box); rgl.setContentsMargins(8,10,8,8); rgl.setSpacing(3)

        sel_row = QWidget(); srl2 = QHBoxLayout(sel_row); srl2.setContentsMargins(0,0,0,0); srl2.setSpacing(4)
        self.btn_select_mode = QPushButton("Select Faces")
        self._set_btn_icon(self.btn_select_mode, 'brush')
        self.btn_select_mode.setObjectName("secondaryBtn"); self.btn_select_mode.setFixedHeight(26)
        self.btn_select_mode.setCheckable(True)
        self.btn_select_mode.setToolTip(
            "Click/drag to paint-select faces on the model.\n"
            "Shift+click = deselect. Ctrl+click = flood-fill similar normals.\n"
            "Then use repair buttons below to fix only the selected area.")
        srl2.addWidget(self.btn_select_mode)
        srl2.addWidget(QLabel("Radius:"))
        self.spin_brush_radius = QDoubleSpinBox()
        self.spin_brush_radius.setRange(2, 100); self.spin_brush_radius.setValue(25)
        self.spin_brush_radius.setSuffix("mm"); self.spin_brush_radius.setFixedHeight(24)
        srl2.addWidget(self.spin_brush_radius)
        rgl.addWidget(sel_row)

        sel_info = QWidget(); sil = QHBoxLayout(sel_info); sil.setContentsMargins(0,0,0,0); sil.setSpacing(4)
        self.lbl_selection = QLabel("0 faces selected"); self.lbl_selection.setObjectName("dimLabel")
        sil.addWidget(self.lbl_selection)
        self.btn_select_clear = QPushButton("Clear"); self.btn_select_clear.setObjectName("nudgeBtn")
        self.btn_select_clear.setFixedSize(50, 22)
        sil.addWidget(self.btn_select_clear)
        self.btn_select_problems = QPushButton("Auto-Select Problems"); self.btn_select_problems.setObjectName("nudgeBtn")
        self.btn_select_problems.setFixedHeight(22)
        self.btn_select_problems.setToolTip("Auto-select degenerate faces and boundary edges")
        sil.addWidget(self.btn_select_problems)
        rgl.addWidget(sel_info)

        # Region repair buttons
        self.btn_region_fix_normals = QPushButton("Fix Selected Normals")
        self.btn_region_fix_normals.setObjectName("nudgeBtn"); self.btn_region_fix_normals.setFixedHeight(24)
        rgl.addWidget(self.btn_region_fix_normals)
        self.btn_region_smooth = QPushButton("Smooth Selected Region")
        self.btn_region_smooth.setObjectName("nudgeBtn"); self.btn_region_smooth.setFixedHeight(24)
        rgl.addWidget(self.btn_region_smooth)
        self.btn_region_delete = QPushButton("Delete Selected Faces")
        self.btn_region_delete.setObjectName("nudgeBtn"); self.btn_region_delete.setFixedHeight(24)
        self.btn_region_delete.setToolTip("Remove the selected faces entirely (creates holes)")
        rgl.addWidget(self.btn_region_delete)

        lay.addWidget(region_box)

    def _build_smoothing_section(self):
        _, lay = self._section("3b · Mesh Smoothing", key='smooth')

        info = QLabel("Smooths curved surfaces after cutting.\nCut faces are locked — parts still fit together.")
        info.setObjectName("dimLabel"); info.setWordWrap(True); lay.addWidget(info)

        # --- SCOPE: which parts ---
        scope_box = QGroupBox("Apply to")
        scope_box.setObjectName("subGroup")
        sbl = QVBoxLayout(scope_box); sbl.setContentsMargins(8,8,8,8); sbl.setSpacing(4)

        self.radio_smooth_selected = QCheckBox("Selected part only")
        self.radio_smooth_selected.setChecked(False)
        self.radio_smooth_selected.setToolTip(
            "Select a part in the Parts tree first, then smooth just that one.")
        sbl.addWidget(self.radio_smooth_selected)

        self.radio_smooth_all = QCheckBox("All parts")
        self.radio_smooth_all.setChecked(True)
        self.radio_smooth_all.setToolTip("Smooth every leaf part after slicing.")
        sbl.addWidget(self.radio_smooth_all)

        # Make them mutually exclusive manually
        self.radio_smooth_selected.toggled.connect(
            lambda v: self.radio_smooth_all.setChecked(not v))
        self.radio_smooth_all.toggled.connect(
            lambda v: self.radio_smooth_selected.setChecked(not v))
        lay.addWidget(scope_box)

        # --- ALGORITHM ---
        ar = QWidget(); arl = QHBoxLayout(ar); arl.setContentsMargins(0,0,0,0); arl.setSpacing(6)
        arl.addWidget(QLabel("Method:"))
        self.combo_smooth_method = QComboBox()
        self.combo_smooth_method.addItems([
            "Taubin (recommended)", "Laplacian (fastest)", "Humphrey (preserve edges)"])
        self.combo_smooth_method.setFixedHeight(28)
        arl.addWidget(self.combo_smooth_method); lay.addWidget(ar)

        # --- STRENGTH: now with clear labels ---
        pr = QWidget(); prl = QHBoxLayout(pr); prl.setContentsMargins(0,0,0,0); prl.setSpacing(6)
        prl.addWidget(QLabel("Passes:"))
        self.spin_smooth_iters = QSpinBox()
        self.spin_smooth_iters.setRange(1, 50); self.spin_smooth_iters.setValue(3)
        self.spin_smooth_iters.setFixedHeight(26)
        self.spin_smooth_iters.setToolTip(
            "How many smoothing passes to run.\n"
            "Start with 1-3 and see the result.\n"
            "More passes = more smoothing = more shape change.")
        prl.addWidget(self.spin_smooth_iters)
        prl.addWidget(QLabel("Strength:"))
        self.spin_smooth_strength = QDoubleSpinBox()
        self.spin_smooth_strength.setRange(0.05, 0.5)
        self.spin_smooth_strength.setValue(0.1)       # SAFE DEFAULT — was 0.5, now 0.1
        self.spin_smooth_strength.setSingleStep(0.05)
        self.spin_smooth_strength.setDecimals(2)
        self.spin_smooth_strength.setFixedHeight(26)
        self.spin_smooth_strength.setToolTip(
            "How much each pass moves vertices.\n"
            "0.05-0.10 = very gentle (recommended for first try)\n"
            "0.20-0.30 = moderate\n"
            "0.50 = aggressive (may distort shape on dense meshes)")
        prl.addWidget(self.spin_smooth_strength); lay.addWidget(pr)

        # Preset buttons
        preset_row = QWidget(); presetl = QHBoxLayout(preset_row)
        presetl.setContentsMargins(0,0,0,0); presetl.setSpacing(4)
        lbl = QLabel("Presets:"); lbl.setObjectName("dimLabel"); presetl.addWidget(lbl)
        for name, iters, strength in [("Gentle", 3, 0.1), ("Medium", 8, 0.2), ("Strong", 15, 0.3)]:
            btn = QPushButton(name)
            btn.setObjectName("nudgeBtn"); btn.setFixedHeight(24)
            btn.clicked.connect(lambda _, i=iters, s=strength: (
                self.spin_smooth_iters.setValue(i),
                self.spin_smooth_strength.setValue(s)))
            presetl.addWidget(btn)
        presetl.addStretch()
        lay.addWidget(preset_row)

        # Preserve cut faces
        self.chk_preserve_cuts = QCheckBox("Lock cut faces (keeps parts fitting together)")
        self.chk_preserve_cuts.setChecked(True)
        self.chk_preserve_cuts.setToolTip(
            "KEEP THIS ON.\n"
            "Prevents the flat cut faces from being moved by smoothing.\n"
            "Without this, parts will no longer align after smoothing.")
        lay.addWidget(self.chk_preserve_cuts)

        # Apply button
        self.btn_smooth = QPushButton("⬡ Apply Smoothing")
        self.btn_smooth.setObjectName("secondaryBtn"); self.btn_smooth.setFixedHeight(34)
        self.btn_smooth.setEnabled(False)
        lay.addWidget(self.btn_smooth)

        self.lbl_smooth_result = QLabel("")
        self.lbl_smooth_result.setObjectName("dimLabel"); self.lbl_smooth_result.setWordWrap(True)
        lay.addWidget(self.lbl_smooth_result)

    def _build_auto_slice_section(self):
        _, lay = self._section("4 · Auto-Slice Preview", key='slice')
        info = QLabel("Preview all cuts before applying. Drag cut planes in the viewport or use sliders.")
        info.setObjectName("dimLabel"); info.setWordWrap(True); lay.addWidget(info)

        sr = QWidget(); srl = QHBoxLayout(sr); srl.setContentsMargins(0,0,0,0); srl.setSpacing(6)
        srl.addWidget(QLabel("Part size:"))
        self.spin_auto_size = QDoubleSpinBox()
        self.spin_auto_size.setRange(20,500); self.spin_auto_size.setValue(150)
        self.spin_auto_size.setSuffix(" mm"); self.spin_auto_size.setFixedHeight(28)
        srl.addWidget(self.spin_auto_size); lay.addWidget(sr)

        self.btn_preview_cuts = QPushButton("Preview Cuts")
        self._set_btn_icon(self.btn_preview_cuts, 'preview')
        self.btn_preview_cuts.setObjectName("secondaryBtn"); self.btn_preview_cuts.setFixedHeight(32)
        self.btn_preview_cuts.setEnabled(False)
        self.btn_preview_cuts.setToolTip("Show cut planes on model without slicing yet.")
        lay.addWidget(self.btn_preview_cuts)

        # Master offset — shift ALL unlocked cuts together
        mo_row = QWidget(); mol = QHBoxLayout(mo_row); mol.setContentsMargins(0,0,0,0); mol.setSpacing(6)
        mol.addWidget(QLabel("Master offset:"))
        self.spin_master_offset = QDoubleSpinBox()
        self.spin_master_offset.setRange(-500,500); self.spin_master_offset.setValue(0)
        self.spin_master_offset.setSuffix(" mm"); self.spin_master_offset.setSingleStep(1)
        self.spin_master_offset.setFixedHeight(28)
        self.spin_master_offset.setToolTip("Shift ALL unlocked cut planes by this amount")
        mol.addWidget(self.spin_master_offset)
        self.btn_apply_offset = QPushButton("Apply")
        self.btn_apply_offset.setObjectName("nudgeBtn"); self.btn_apply_offset.setFixedHeight(28)
        mol.addWidget(self.btn_apply_offset); lay.addWidget(mo_row)

        # Preview cut list — click to select + gizmo
        self.preview_cut_list = QTableWidget(0, 3)
        self.preview_cut_list.setHorizontalHeaderLabels(["#", "Axis", "Position (mm)"])
        hdr = self.preview_cut_list.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.Stretch)
        self.preview_cut_list.setFixedHeight(110)
        self.preview_cut_list.setObjectName("cutTable")
        self.preview_cut_list.setSelectionBehavior(QTableWidget.SelectRows)
        self.preview_cut_list.setEditTriggers(QTableWidget.NoEditTriggers)
        self.preview_cut_list.setToolTip("Click a cut to select it and show gizmo in viewport.\nDrag the arrow to move, drag rings to rotate.")
        lay.addWidget(self.preview_cut_list)

        # Selected cut fine controls
        fc_row = QWidget(); fcl = QHBoxLayout(fc_row); fcl.setContentsMargins(0,0,0,0); fcl.setSpacing(4)
        self.btn_pn_m10 = QPushButton("−10"); self.btn_pn_m1 = QPushButton("−1")
        self.btn_pn_p1  = QPushButton("+1");  self.btn_pn_p10 = QPushButton("+10")
        for b in [self.btn_pn_m10,self.btn_pn_m1,self.btn_pn_p1,self.btn_pn_p10]:
            b.setObjectName("nudgeBtn"); b.setFixedHeight(26); fcl.addWidget(b)
        lay.addWidget(fc_row)

        # Lock / unlock selected preview cut
        lk_row = QWidget(); lkl = QHBoxLayout(lk_row); lkl.setContentsMargins(0,0,0,0); lkl.setSpacing(6)
        self.btn_lock_preview = QPushButton("🔒 Lock"); self.btn_lock_preview.setObjectName("lockBtn"); self.btn_lock_preview.setFixedHeight(28)
        self.btn_unlock_preview = QPushButton("Unlock"); self.btn_unlock_preview.setObjectName("secondaryBtn"); self.btn_unlock_preview.setFixedHeight(28)
        self.btn_unlock_all_preview = QPushButton("Unlock All"); self.btn_unlock_all_preview.setObjectName("secondaryBtn"); self.btn_unlock_all_preview.setFixedHeight(28)
        lkl.addWidget(self.btn_lock_preview); lkl.addWidget(self.btn_unlock_preview); lkl.addWidget(self.btn_unlock_all_preview)
        lay.addWidget(lk_row)

        self.btn_auto_slice = QPushButton("⚡ Apply All Cuts")
        self.btn_auto_slice.setObjectName("primaryBtn"); self.btn_auto_slice.setFixedHeight(36)
        self.btn_auto_slice.setEnabled(False)
        self.btn_auto_slice.setToolTip("Commit all preview cuts and slice the model.")
        lay.addWidget(self.btn_auto_slice)

        self._preview_planes = []      # CutDefinition list in preview mode
        self._active_preview_idx = -1  # which one has the gizmo

    def _build_seam_section(self):
        _, lay = self._section("4b · Seam Heatmap", key='seam')

        info = QLabel("Analyse where cuts will be easy or hard to hide.\nGreen = good seam location. Red = visible on curved surface.")
        info.setObjectName("dimLabel"); info.setWordWrap(True); lay.addWidget(info)

        self.btn_show_heatmap = QPushButton("Show Seam Heatmap")
        self._set_btn_icon(self.btn_show_heatmap, 'heatmap')
        self.btn_show_heatmap.setObjectName("secondaryBtn"); self.btn_show_heatmap.setFixedHeight(32)
        self.btn_show_heatmap.setEnabled(False)
        self.btn_show_heatmap.setToolTip(
            "Colour the model surface:\n"
            "  Green = flat/crease — cut here, seam hides easily\n"
            "  Yellow = moderate curvature\n"
            "  Red = smooth curve — hard to hide seam here")
        lay.addWidget(self.btn_show_heatmap)

        self.btn_show_creases = QPushButton("〰 Show Natural Seams")
        self.btn_show_creases.setObjectName("secondaryBtn"); self.btn_show_creases.setFixedHeight(32)
        self.btn_show_creases.setEnabled(False)
        self.btn_show_creases.setToolTip(
            "Highlight existing body panel lines and creases.\n"
            "These are ideal cut positions — the seam follows an existing edge.")
        lay.addWidget(self.btn_show_creases)

        self.btn_clear_heatmap = QPushButton("Clear Overlay")
        self.btn_clear_heatmap.setObjectName("nudgeBtn"); self.btn_clear_heatmap.setFixedHeight(26)
        self.btn_clear_heatmap.setEnabled(False)
        lay.addWidget(self.btn_clear_heatmap)

        # Suggestion list
        self.lbl_seam_suggestions = QLabel("")
        self.lbl_seam_suggestions.setObjectName("dimLabel"); self.lbl_seam_suggestions.setWordWrap(True)
        lay.addWidget(self.lbl_seam_suggestions)

        self.seam_suggest_table = QTableWidget(0, 3)
        self.seam_suggest_table.setHorizontalHeaderLabels(["Axis", "Position", "Strength"])
        hdr = self.seam_suggest_table.horizontalHeader()
        for i in range(3): hdr.setSectionResizeMode(i, QHeaderView.Stretch)
        self.seam_suggest_table.setFixedHeight(100)
        self.seam_suggest_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.seam_suggest_table.setToolTip("Click a suggested cut to jump to that position in the Cut section")
        self.seam_suggest_table.setVisible(False)
        lay.addWidget(self.seam_suggest_table)

    def _build_cut_section(self):
        _, lay = self._section("5 · Cut — Advanced Options", key='cut')

        info = QLabel(
            "For basic cuts, use the Quick Cut bar below the viewport.\n"
            "This section is for angled cuts and section (bounded) cuts."
        )
        info.setObjectName("dimLabel"); info.setWordWrap(True); lay.addWidget(info)

        # Mode selector — duplicates bottom bar but needed for rotation/section controls
        mr = QWidget(); mrl = QHBoxLayout(mr); mrl.setContentsMargins(0,0,0,0); mrl.setSpacing(6)
        mrl.addWidget(QLabel("Mode:"))
        self.combo_cut_mode = QComboBox()
        self.combo_cut_mode.addItems(["Full Slice","Free (Angled)","Section Cut","Groove (Zigzag)","Natural (Follow Crease)"])
        self.combo_cut_mode.setFixedHeight(28)
        mrl.addWidget(self.combo_cut_mode)
        lay.addWidget(mr)

        # Axis + position in one compact row (used by rotation/section previews)
        pr = QWidget(); prl = QHBoxLayout(pr); prl.setContentsMargins(0,0,0,0); prl.setSpacing(6)
        prl.addWidget(QLabel("Axis:"))
        self.combo_cut_axis = QComboBox(); self.combo_cut_axis.addItems(["X","Y","Z"])
        self.combo_cut_axis.setFixedSize(45, 26)
        prl.addWidget(self.combo_cut_axis)
        prl.addWidget(QLabel("Pos:"))
        self.spin_cut_pos = QDoubleSpinBox(); self.spin_cut_pos.setRange(-9999,9999)
        self.spin_cut_pos.setSuffix(" mm"); self.spin_cut_pos.setDecimals(1)
        self.spin_cut_pos.setSingleStep(1); self.spin_cut_pos.setFixedHeight(26)
        prl.addWidget(self.spin_cut_pos,1)
        lay.addWidget(pr)

        # Hidden nudge buttons (referenced by signals but not shown — use quick-cut bar instead)
        self.btn_cm10 = QPushButton(); self.btn_cm10.setVisible(False)
        self.btn_cm1 = QPushButton(); self.btn_cm1.setVisible(False)
        self.btn_cp1 = QPushButton(); self.btn_cp1.setVisible(False)
        self.btn_cp10 = QPushButton(); self.btn_cp10.setVisible(False)

        # Rotation (for Free mode)
        self.rot_group = QGroupBox("Tilt angle (degrees)")
        self.rot_group.setObjectName("subGroup")
        rgl = QVBoxLayout(self.rot_group); rgl.setContentsMargins(8,10,8,8); rgl.setSpacing(4)
        def rot_row(lbl, tip, color):
            row = QWidget(); rl = QHBoxLayout(row); rl.setContentsMargins(0,0,0,0); rl.setSpacing(6)
            l = QLabel(lbl); l.setFixedWidth(50); l.setStyleSheet(f"color:{color};font-weight:600;font-size:11px;")
            s = QDoubleSpinBox(); s.setRange(-89,89); s.setSuffix("°"); s.setDecimals(1); s.setSingleStep(5); s.setFixedHeight(26); s.setValue(0)
            s.setToolTip(tip)
            rl.addWidget(l); rl.addWidget(s,1); return row, s
        rr_u, self.spin_rot_x = rot_row("🔴 Tilt U", "Tilt left/right (drag red ring in viewport)", "#e05555")
        rr_v, self.spin_rot_y = rot_row("🟢 Tilt V", "Tilt up/down (drag green ring in viewport)", "#50c870")
        for rr in [rr_u, rr_v]: rgl.addWidget(rr)
        # Keep spin_rot_z as a dummy (not used but referenced)
        self.spin_rot_z = QDoubleSpinBox(); self.spin_rot_z.setVisible(False)
        lay.addWidget(self.rot_group)
        self.rot_group.setVisible(False)

        # Section size (for Section mode)
        self.sec_group = QGroupBox("Section Size")
        self.sec_group.setObjectName("subGroup")
        sgl = QVBoxLayout(self.sec_group); sgl.setContentsMargins(8,10,8,8); sgl.setSpacing(4)
        def sec_row(lbl):
            row = QWidget(); rl = QHBoxLayout(row); rl.setContentsMargins(0,0,0,0); rl.setSpacing(6)
            l = QLabel(lbl); l.setFixedWidth(50)
            s = QDoubleSpinBox(); s.setRange(10,5000); s.setSuffix(" mm"); s.setDecimals(1); s.setSingleStep(10); s.setValue(100); s.setFixedHeight(26)
            rl.addWidget(l); rl.addWidget(s,1); return row, s
        sr_w, self.spin_sec_w = sec_row("Width:")
        sr_h, self.spin_sec_h = sec_row("Height:")
        sgl.addWidget(sr_w); sgl.addWidget(sr_h)
        lay.addWidget(self.sec_group)
        self.sec_group.setVisible(False)

        # Apply cut button
        self.btn_apply_cut = QPushButton("Apply Cut to Selected Part")
        self._set_btn_icon(self.btn_apply_cut, 'cut')
        self.btn_apply_cut.setObjectName("cutBtn"); self.btn_apply_cut.setFixedHeight(38); self.btn_apply_cut.setEnabled(False)
        lay.addWidget(self.btn_apply_cut)

        self.lbl_cut_result = QLabel(""); self.lbl_cut_result.setObjectName("dimLabel"); self.lbl_cut_result.setWordWrap(True)
        lay.addWidget(self.lbl_cut_result)

    def _build_joints_section(self):
        _, lay = self._section("6 · Joint / Dowel Config", key='joints')

        info = QLabel("Configure joints per-cut. Select a cut in the Parts tree,\nthen set the joint type for that face.")
        info.setObjectName("dimLabel"); info.setWordWrap(True)
        lay.addWidget(info)

        # Joint type for selected cut
        tr = QWidget(); trl = QHBoxLayout(tr); trl.setContentsMargins(0,0,0,0); trl.setSpacing(6)
        trl.addWidget(QLabel("Type:"))
        self.combo_joint_type = QComboBox()
        self.combo_joint_type.addItems(["Flat (glue/weld)", "Dowel — Round (steel rod)",
                                        "Dowel — Rect (flat bar)", "Dovetail",
                                        "Magnet Pocket", "Snap-Fit Clip",
                                        "D-Shape (anti-rotation)", "Pyramid (self-centering)",
                                        "Terrace (stepped)", "Square Peg"])
        self.combo_joint_type.setFixedHeight(28)
        self.combo_joint_type.setToolTip(
            "Flat — glue/weld only, no mechanical alignment\n"
            "Round Dowel — steel rod for strong alignment\n"
            "Rect Dowel — flat bar key stock\n"
            "Dovetail — mechanical interlock wedge\n"
            "Magnet — press-fit magnet pockets\n"
            "Snap-Fit — clip that clicks together\n"
            "D-Shape — half-round, prevents rotation\n"
            "Pyramid — tapered, self-centering on assembly\n"
            "Terrace — stepped, large bonding surface\n"
            "Square — square peg, prevents rotation")
        trl.addWidget(self.combo_joint_type)
        lay.addWidget(tr)

        # Round dowel settings
        self.round_group = QGroupBox("Round Dowel (steel rod)")
        self.round_group.setObjectName("subGroup")
        rgl = QVBoxLayout(self.round_group); rgl.setContentsMargins(8,10,8,8); rgl.setSpacing(4)

        r1 = QWidget(); r1l = QHBoxLayout(r1); r1l.setContentsMargins(0,0,0,0); r1l.setSpacing(6)
        r1l.addWidget(QLabel("Rod radius:")); 
        self.spin_rod_radius = QDoubleSpinBox(); self.spin_rod_radius.setRange(1,25)
        self.spin_rod_radius.setValue(5); self.spin_rod_radius.setSuffix(" mm"); self.spin_rod_radius.setFixedHeight(26)
        r1l.addWidget(self.spin_rod_radius); rgl.addWidget(r1)

        r2 = QWidget(); r2l = QHBoxLayout(r2); r2l.setContentsMargins(0,0,0,0); r2l.setSpacing(6)
        r2l.addWidget(QLabel("Depth A:")); 
        self.spin_depth_a = QDoubleSpinBox(); self.spin_depth_a.setRange(5,200)
        self.spin_depth_a.setValue(30); self.spin_depth_a.setSuffix(" mm"); self.spin_depth_a.setFixedHeight(26)
        r2l.addWidget(self.spin_depth_a)
        r2l.addWidget(QLabel("B:"))
        self.spin_depth_b = QDoubleSpinBox(); self.spin_depth_b.setRange(5,200)
        self.spin_depth_b.setValue(30); self.spin_depth_b.setSuffix(" mm"); self.spin_depth_b.setFixedHeight(26)
        r2l.addWidget(self.spin_depth_b); rgl.addWidget(r2)

        r3 = QWidget(); r3l = QHBoxLayout(r3); r3l.setContentsMargins(0,0,0,0); r3l.setSpacing(6)
        r3l.addWidget(QLabel("Count:"))
        self.spin_rod_count = QSpinBox(); self.spin_rod_count.setRange(1,8)
        self.spin_rod_count.setValue(2); self.spin_rod_count.setFixedHeight(26)
        r3l.addWidget(self.spin_rod_count)
        r3l.addWidget(QLabel("Spacing:"))
        self.spin_rod_spacing = QDoubleSpinBox(); self.spin_rod_spacing.setRange(0,500)
        self.spin_rod_spacing.setValue(0); self.spin_rod_spacing.setSuffix(" mm")
        self.spin_rod_spacing.setFixedHeight(26)
        self.spin_rod_spacing.setToolTip("0 = auto-distribute evenly")
        r3l.addWidget(self.spin_rod_spacing); rgl.addWidget(r3)

        r4 = QWidget(); r4l = QHBoxLayout(r4); r4l.setContentsMargins(0,0,0,0); r4l.setSpacing(6)
        r4l.addWidget(QLabel("Tolerance:"))
        self.spin_rod_tol = QDoubleSpinBox(); self.spin_rod_tol.setRange(0.1,2.0)
        self.spin_rod_tol.setValue(0.3); self.spin_rod_tol.setSuffix(" mm"); self.spin_rod_tol.setFixedHeight(26)
        r4l.addWidget(self.spin_rod_tol); r4l.addStretch(); rgl.addWidget(r4)
        lay.addWidget(self.round_group)

        # Rectangular dowel settings
        self.rect_group = QGroupBox("Rectangular Dowel (flat bar / key)")
        self.rect_group.setObjectName("subGroup")
        rcl = QVBoxLayout(self.rect_group); rcl.setContentsMargins(8,10,8,8); rcl.setSpacing(4)

        rc1 = QWidget(); rc1l = QHBoxLayout(rc1); rc1l.setContentsMargins(0,0,0,0); rc1l.setSpacing(4)
        rc1l.addWidget(QLabel("W:"))
        self.spin_rect_w = QDoubleSpinBox(); self.spin_rect_w.setRange(1,50)
        self.spin_rect_w.setValue(3); self.spin_rect_w.setSuffix(" mm"); self.spin_rect_w.setFixedHeight(26)
        rc1l.addWidget(self.spin_rect_w)
        rc1l.addWidget(QLabel("H:"))
        self.spin_rect_h = QDoubleSpinBox(); self.spin_rect_h.setRange(1,100)
        self.spin_rect_h.setValue(12); self.spin_rect_h.setSuffix(" mm"); self.spin_rect_h.setFixedHeight(26)
        rc1l.addWidget(self.spin_rect_h); rcl.addWidget(rc1)

        rc2 = QWidget(); rc2l = QHBoxLayout(rc2); rc2l.setContentsMargins(0,0,0,0); rc2l.setSpacing(6)
        rc2l.addWidget(QLabel("Depth A:"))
        self.spin_rect_depth_a = QDoubleSpinBox(); self.spin_rect_depth_a.setRange(5,300)
        self.spin_rect_depth_a.setValue(60); self.spin_rect_depth_a.setSuffix(" mm"); self.spin_rect_depth_a.setFixedHeight(26)
        rc2l.addWidget(self.spin_rect_depth_a)
        rc2l.addWidget(QLabel("B:"))
        self.spin_rect_depth_b = QDoubleSpinBox(); self.spin_rect_depth_b.setRange(5,300)
        self.spin_rect_depth_b.setValue(60); self.spin_rect_depth_b.setSuffix(" mm"); self.spin_rect_depth_b.setFixedHeight(26)
        rc2l.addWidget(self.spin_rect_depth_b); rcl.addWidget(rc2)

        rc3 = QWidget(); rc3l = QHBoxLayout(rc3); rc3l.setContentsMargins(0,0,0,0); rc3l.setSpacing(6)
        rc3l.addWidget(QLabel("Count:"))
        self.spin_rect_count = QSpinBox(); self.spin_rect_count.setRange(1,6)
        self.spin_rect_count.setValue(1); self.spin_rect_count.setFixedHeight(26)
        rc3l.addWidget(self.spin_rect_count)
        rc3l.addWidget(QLabel("Tol:"))
        self.spin_rect_tol = QDoubleSpinBox(); self.spin_rect_tol.setRange(0.1,2.0)
        self.spin_rect_tol.setValue(0.3); self.spin_rect_tol.setSuffix(" mm"); self.spin_rect_tol.setFixedHeight(26)
        rc3l.addWidget(self.spin_rect_tol); rcl.addWidget(rc3)

        lay.addWidget(self.rect_group)

        # Dovetail settings (simple)
        self.dove_group = QGroupBox("Dovetail")
        self.dove_group.setObjectName("subGroup")
        dvl = QVBoxLayout(self.dove_group); dvl.setContentsMargins(8,10,8,8); dvl.setSpacing(4)
        dv1 = QWidget(); dv1l = QHBoxLayout(dv1); dv1l.setContentsMargins(0,0,0,0); dv1l.setSpacing(6)
        dv1l.addWidget(QLabel("Size:"))
        self.spin_dove_size = QDoubleSpinBox(); self.spin_dove_size.setRange(3,30)
        self.spin_dove_size.setValue(8); self.spin_dove_size.setSuffix(" mm"); self.spin_dove_size.setFixedHeight(26)
        dv1l.addWidget(self.spin_dove_size)
        dv1l.addWidget(QLabel("Tol:"))
        self.spin_dove_tol = QDoubleSpinBox(); self.spin_dove_tol.setRange(0.1,1.0)
        self.spin_dove_tol.setValue(0.3); self.spin_dove_tol.setSuffix(" mm"); self.spin_dove_tol.setFixedHeight(26)
        dv1l.addWidget(self.spin_dove_tol)
        dvl.addWidget(dv1)
        lay.addWidget(self.dove_group)

        # Magnet pocket settings
        self.magnet_group = QGroupBox("Magnet Pocket")
        self.magnet_group.setObjectName("subGroup")
        mgl = QVBoxLayout(self.magnet_group); mgl.setContentsMargins(8,10,8,8); mgl.setSpacing(4)
        mg_info = QLabel("Common magnets: 6×3mm, 8×3mm, 10×2mm (diameter×height)")
        mg_info.setObjectName("dimLabel"); mg_info.setWordWrap(True); mgl.addWidget(mg_info)
        mg1 = QWidget(); mg1l = QHBoxLayout(mg1); mg1l.setContentsMargins(0,0,0,0); mg1l.setSpacing(6)
        mg1l.addWidget(QLabel("Diameter:"))
        self.spin_magnet_dia = QDoubleSpinBox(); self.spin_magnet_dia.setRange(2,30)
        self.spin_magnet_dia.setValue(6); self.spin_magnet_dia.setSuffix(" mm"); self.spin_magnet_dia.setFixedHeight(26)
        mg1l.addWidget(self.spin_magnet_dia)
        mg1l.addWidget(QLabel("Depth:"))
        self.spin_magnet_depth = QDoubleSpinBox(); self.spin_magnet_depth.setRange(1,20)
        self.spin_magnet_depth.setValue(3.2); self.spin_magnet_depth.setSuffix(" mm"); self.spin_magnet_depth.setFixedHeight(26)
        mg1l.addWidget(self.spin_magnet_depth); mgl.addWidget(mg1)
        mg2 = QWidget(); mg2l = QHBoxLayout(mg2); mg2l.setContentsMargins(0,0,0,0); mg2l.setSpacing(6)
        mg2l.addWidget(QLabel("Count:"))
        self.spin_magnet_count = QSpinBox(); self.spin_magnet_count.setRange(1,8)
        self.spin_magnet_count.setValue(2); self.spin_magnet_count.setFixedHeight(26)
        mg2l.addWidget(self.spin_magnet_count)
        mg2l.addWidget(QLabel("Tol:"))
        self.spin_magnet_tol = QDoubleSpinBox(); self.spin_magnet_tol.setRange(0.1,1.0)
        self.spin_magnet_tol.setValue(0.15); self.spin_magnet_tol.setSuffix(" mm"); self.spin_magnet_tol.setFixedHeight(26)
        self.spin_magnet_tol.setToolTip("Clearance around the magnet.\n0.10-0.15mm = tight press-fit\n0.20-0.30mm = drop-in fit")
        mg2l.addWidget(self.spin_magnet_tol); mgl.addWidget(mg2)
        lay.addWidget(self.magnet_group)

        # Snap-fit settings
        self.snap_group = QGroupBox("Snap-Fit Clip")
        self.snap_group.setObjectName("subGroup")
        snl = QVBoxLayout(self.snap_group); snl.setContentsMargins(8,10,8,8); snl.setSpacing(4)
        sn_info = QLabel("Cantilever snap-fit clips on cut faces.\nParts click together — no glue needed.")
        sn_info.setObjectName("dimLabel"); sn_info.setWordWrap(True); snl.addWidget(sn_info)
        sn1 = QWidget(); sn1l = QHBoxLayout(sn1); sn1l.setContentsMargins(0,0,0,0); sn1l.setSpacing(6)
        sn1l.addWidget(QLabel("Clip width:"))
        self.spin_snap_w = QDoubleSpinBox(); self.spin_snap_w.setRange(2,20)
        self.spin_snap_w.setValue(5); self.spin_snap_w.setSuffix(" mm"); self.spin_snap_w.setFixedHeight(26)
        sn1l.addWidget(self.spin_snap_w)
        sn1l.addWidget(QLabel("Height:"))
        self.spin_snap_h = QDoubleSpinBox(); self.spin_snap_h.setRange(2,20)
        self.spin_snap_h.setValue(8); self.spin_snap_h.setSuffix(" mm"); self.spin_snap_h.setFixedHeight(26)
        sn1l.addWidget(self.spin_snap_h); snl.addWidget(sn1)
        sn2 = QWidget(); sn2l = QHBoxLayout(sn2); sn2l.setContentsMargins(0,0,0,0); sn2l.setSpacing(6)
        sn2l.addWidget(QLabel("Count:"))
        self.spin_snap_count = QSpinBox(); self.spin_snap_count.setRange(1,6)
        self.spin_snap_count.setValue(2); self.spin_snap_count.setFixedHeight(26)
        sn2l.addWidget(self.spin_snap_count)
        sn2l.addWidget(QLabel("Tol:"))
        self.spin_snap_tol = QDoubleSpinBox(); self.spin_snap_tol.setRange(0.1,1.0)
        self.spin_snap_tol.setValue(0.3); self.spin_snap_tol.setSuffix(" mm"); self.spin_snap_tol.setFixedHeight(26)
        sn2l.addWidget(self.spin_snap_tol); snl.addWidget(sn2)
        lay.addWidget(self.snap_group)

        # D-Shape connector settings
        self.dshape_group = QGroupBox("D-Shape (anti-rotation)")
        self.dshape_group.setObjectName("subGroup")
        dsl = QVBoxLayout(self.dshape_group); dsl.setContentsMargins(8,10,8,8); dsl.setSpacing(4)
        ds1 = QWidget(); ds1l = QHBoxLayout(ds1); ds1l.setContentsMargins(0,0,0,0); ds1l.setSpacing(6)
        ds1l.addWidget(QLabel("Radius:"))
        self.spin_dshape_radius = QDoubleSpinBox(); self.spin_dshape_radius.setRange(2,25)
        self.spin_dshape_radius.setValue(5); self.spin_dshape_radius.setSuffix(" mm"); self.spin_dshape_radius.setFixedHeight(26)
        ds1l.addWidget(self.spin_dshape_radius)
        ds1l.addWidget(QLabel("Depth:"))
        self.spin_dshape_depth = QDoubleSpinBox(); self.spin_dshape_depth.setRange(5,60)
        self.spin_dshape_depth.setValue(15); self.spin_dshape_depth.setSuffix(" mm"); self.spin_dshape_depth.setFixedHeight(26)
        ds1l.addWidget(self.spin_dshape_depth)
        dsl.addWidget(ds1)
        ds2 = QWidget(); ds2l = QHBoxLayout(ds2); ds2l.setContentsMargins(0,0,0,0); ds2l.setSpacing(6)
        ds2l.addWidget(QLabel("Count:"))
        self.spin_dshape_count = QSpinBox(); self.spin_dshape_count.setRange(1,6)
        self.spin_dshape_count.setValue(2); self.spin_dshape_count.setFixedHeight(26)
        ds2l.addWidget(self.spin_dshape_count)
        ds2l.addWidget(QLabel("Tol:"))
        self.spin_dshape_tol = QDoubleSpinBox(); self.spin_dshape_tol.setRange(0.1,1.0)
        self.spin_dshape_tol.setValue(0.3); self.spin_dshape_tol.setSuffix(" mm"); self.spin_dshape_tol.setFixedHeight(26)
        ds2l.addWidget(self.spin_dshape_tol); dsl.addWidget(ds2)
        info = QLabel("Flat side prevents rotation — parts only fit one way")
        info.setObjectName("dimLabel"); info.setWordWrap(True); dsl.addWidget(info)
        lay.addWidget(self.dshape_group)

        # Pyramid connector settings
        self.pyramid_group = QGroupBox("Pyramid (self-centering)")
        self.pyramid_group.setObjectName("subGroup")
        pyl = QVBoxLayout(self.pyramid_group); pyl.setContentsMargins(8,10,8,8); pyl.setSpacing(4)
        py1 = QWidget(); py1l = QHBoxLayout(py1); py1l.setContentsMargins(0,0,0,0); py1l.setSpacing(6)
        py1l.addWidget(QLabel("Base:"))
        self.spin_pyramid_base = QDoubleSpinBox(); self.spin_pyramid_base.setRange(3,30)
        self.spin_pyramid_base.setValue(8); self.spin_pyramid_base.setSuffix(" mm"); self.spin_pyramid_base.setFixedHeight(26)
        py1l.addWidget(self.spin_pyramid_base)
        py1l.addWidget(QLabel("Depth:"))
        self.spin_pyramid_depth = QDoubleSpinBox(); self.spin_pyramid_depth.setRange(5,40)
        self.spin_pyramid_depth.setValue(12); self.spin_pyramid_depth.setSuffix(" mm"); self.spin_pyramid_depth.setFixedHeight(26)
        py1l.addWidget(self.spin_pyramid_depth)
        pyl.addWidget(py1)
        py2 = QWidget(); py2l = QHBoxLayout(py2); py2l.setContentsMargins(0,0,0,0); py2l.setSpacing(6)
        py2l.addWidget(QLabel("Count:"))
        self.spin_pyramid_count = QSpinBox(); self.spin_pyramid_count.setRange(1,4)
        self.spin_pyramid_count.setValue(2); self.spin_pyramid_count.setFixedHeight(26)
        py2l.addWidget(self.spin_pyramid_count)
        py2l.addWidget(QLabel("Taper:"))
        self.spin_pyramid_taper = QDoubleSpinBox(); self.spin_pyramid_taper.setRange(0.3,0.95)
        self.spin_pyramid_taper.setValue(0.7); self.spin_pyramid_taper.setFixedHeight(26)
        py2l.addWidget(self.spin_pyramid_taper)
        pyl.addWidget(py2)
        info = QLabel("Tapered shape guides parts together during assembly")
        info.setObjectName("dimLabel"); info.setWordWrap(True); pyl.addWidget(info)
        lay.addWidget(self.pyramid_group)

        # Terrace connector settings
        self.terrace_group = QGroupBox("Terrace (stepped)")
        self.terrace_group.setObjectName("subGroup")
        tel = QVBoxLayout(self.terrace_group); tel.setContentsMargins(8,10,8,8); tel.setSpacing(4)
        te1 = QWidget(); te1l = QHBoxLayout(te1); te1l.setContentsMargins(0,0,0,0); te1l.setSpacing(6)
        te1l.addWidget(QLabel("Width:"))
        self.spin_terrace_width = QDoubleSpinBox(); self.spin_terrace_width.setRange(5,30)
        self.spin_terrace_width.setValue(10); self.spin_terrace_width.setSuffix(" mm"); self.spin_terrace_width.setFixedHeight(26)
        te1l.addWidget(self.spin_terrace_width)
        te1l.addWidget(QLabel("Depth:"))
        self.spin_terrace_depth = QDoubleSpinBox(); self.spin_terrace_depth.setRange(5,40)
        self.spin_terrace_depth.setValue(12); self.spin_terrace_depth.setSuffix(" mm"); self.spin_terrace_depth.setFixedHeight(26)
        te1l.addWidget(self.spin_terrace_depth)
        tel.addWidget(te1)
        te2 = QWidget(); te2l = QHBoxLayout(te2); te2l.setContentsMargins(0,0,0,0); te2l.setSpacing(6)
        te2l.addWidget(QLabel("Steps:"))
        self.spin_terrace_steps = QSpinBox(); self.spin_terrace_steps.setRange(2,6)
        self.spin_terrace_steps.setValue(3); self.spin_terrace_steps.setFixedHeight(26)
        te2l.addWidget(self.spin_terrace_steps)
        te2l.addWidget(QLabel("Count:"))
        self.spin_terrace_count = QSpinBox(); self.spin_terrace_count.setRange(1,4)
        self.spin_terrace_count.setValue(1); self.spin_terrace_count.setFixedHeight(26)
        te2l.addWidget(self.spin_terrace_count)
        tel.addWidget(te2)
        info = QLabel("Stair-step profile — large bonding area + easy alignment")
        info.setObjectName("dimLabel"); info.setWordWrap(True); tel.addWidget(info)
        lay.addWidget(self.terrace_group)

        # Square peg settings
        self.square_group = QGroupBox("Square Peg")
        self.square_group.setObjectName("subGroup")
        sql = QVBoxLayout(self.square_group); sql.setContentsMargins(8,10,8,8); sql.setSpacing(4)
        sq1 = QWidget(); sq1l = QHBoxLayout(sq1); sq1l.setContentsMargins(0,0,0,0); sq1l.setSpacing(6)
        sq1l.addWidget(QLabel("Size:"))
        self.spin_square_size = QDoubleSpinBox(); self.spin_square_size.setRange(3,20)
        self.spin_square_size.setValue(6); self.spin_square_size.setSuffix(" mm"); self.spin_square_size.setFixedHeight(26)
        sq1l.addWidget(self.spin_square_size)
        sq1l.addWidget(QLabel("Depth:"))
        self.spin_square_depth = QDoubleSpinBox(); self.spin_square_depth.setRange(5,40)
        self.spin_square_depth.setValue(12); self.spin_square_depth.setSuffix(" mm"); self.spin_square_depth.setFixedHeight(26)
        sq1l.addWidget(self.spin_square_depth)
        sq1l.addWidget(QLabel("Count:"))
        self.spin_square_count = QSpinBox(); self.spin_square_count.setRange(1,6)
        self.spin_square_count.setValue(2); self.spin_square_count.setFixedHeight(26)
        sq1l.addWidget(self.spin_square_count)
        sql.addWidget(sq1)
        info = QLabel("Prevents rotation like D-shape — simpler geometry, easier to print")
        info.setObjectName("dimLabel"); info.setWordWrap(True); sql.addWidget(info)
        lay.addWidget(self.square_group)

        # Manual dowel placement
        manual_box = QGroupBox("Manual Dowel Placement")
        manual_box.setObjectName("subGroup")
        mbl = QVBoxLayout(manual_box); mbl.setContentsMargins(8,10,8,8); mbl.setSpacing(4)
        self.chk_manual_dowels = QCheckBox("Enable manual placement mode")
        self.chk_manual_dowels.setToolTip(
            "When enabled, click on a cut face in the viewport\n"
            "to place a dowel marker at that position.\n"
            "Markers are shown as cyan diamonds.")
        mbl.addWidget(self.chk_manual_dowels)
        mb_row = QWidget(); mb_rl = QHBoxLayout(mb_row); mb_rl.setContentsMargins(0,0,0,0); mb_rl.setSpacing(6)
        self.btn_clear_dowels = QPushButton("Clear All Markers")
        self.btn_clear_dowels.setObjectName("nudgeBtn"); self.btn_clear_dowels.setFixedHeight(26)
        mb_rl.addWidget(self.btn_clear_dowels)
        self.lbl_dowel_count = QLabel("0 markers"); self.lbl_dowel_count.setObjectName("dimLabel")
        mb_rl.addWidget(self.lbl_dowel_count); mb_rl.addStretch()
        mbl.addWidget(mb_row)
        lay.addWidget(manual_box)

        # Apply button
        self.btn_apply_joint = QPushButton("Apply Joint to Selected Part's Cut Face")
        self.btn_apply_joint.setObjectName("secondaryBtn"); self.btn_apply_joint.setFixedHeight(32)
        self.btn_apply_joint.setToolTip("Stores this joint config — applied at export time")
        lay.addWidget(self.btn_apply_joint)

        self.lbl_joint_result = QLabel("")
        self.lbl_joint_result.setObjectName("dimLabel"); self.lbl_joint_result.setWordWrap(True)
        lay.addWidget(self.lbl_joint_result)

        # Tolerance test
        self.btn_tol_test = QPushButton("Export Tolerance Test Print")
        self._set_btn_icon(self.btn_tol_test, 'test')
        self.btn_tol_test.setObjectName("secondaryBtn"); self.btn_tol_test.setFixedHeight(30)
        self.btn_tol_test.setToolTip(
            "Generate a small test STL with the selected joint\n"
            "at 6 different tolerances (0.10mm to 0.40mm).\n"
            "Print it, test the fit, then set the right tolerance.")
        lay.addWidget(self.btn_tol_test)

        # Initial visibility
        self._update_joint_groups(0)

    def _build_hollow_section(self):
        _, lay = self._section("6b · Hollow Shell", key='hollow')
        self.chk_hollow = QCheckBox("Enable hollow shell on export"); self.chk_hollow.setChecked(False)
        lay.addWidget(self.chk_hollow)
        wr = QWidget(); wrl = QHBoxLayout(wr); wrl.setContentsMargins(0,0,0,0); wrl.setSpacing(6)
        wrl.addWidget(QLabel("Wall thickness:"))
        self.spin_wall = QDoubleSpinBox(); self.spin_wall.setRange(1,20); self.spin_wall.setValue(3); self.spin_wall.setSuffix(" mm"); self.spin_wall.setSingleStep(0.5); self.spin_wall.setFixedHeight(28)
        wrl.addWidget(self.spin_wall); lay.addWidget(wr)
        # Joint
        jr = QWidget(); jrl = QHBoxLayout(jr); jrl.setContentsMargins(0,0,0,0); jrl.setSpacing(6)
        jrl.addWidget(QLabel("Joint:"))
        self.combo_joint = QComboBox(); self.combo_joint.addItems(["Flat","Dowel holes","Dovetail slots"]); self.combo_joint.setFixedHeight(26)
        jrl.addWidget(self.combo_joint); lay.addWidget(jr)
        self.lbl_hollow_info = QLabel(""); self.lbl_hollow_info.setObjectName("dimLabel"); self.lbl_hollow_info.setWordWrap(True)
        lay.addWidget(self.lbl_hollow_info)

    def _build_export_section(self):
        _, lay = self._section("7 · Export", key='export')

        # Format + material row
        fr = QWidget(); frl = QHBoxLayout(fr); frl.setContentsMargins(0,0,0,0); frl.setSpacing(6)
        frl.addWidget(QLabel("Format:"))
        self.combo_format = QComboBox(); self.combo_format.addItems(["STL","OBJ","3MF"]); self.combo_format.setFixedHeight(26)
        frl.addWidget(self.combo_format)
        frl.addWidget(QLabel("Material:"))
        self.combo_material = QComboBox(); self.combo_material.addItems(["PETG","PLA","ABS","ASA"]); self.combo_material.setFixedHeight(26)
        frl.addWidget(self.combo_material)
        lay.addWidget(fr)

        # Connector pins
        self.chk_pins = QCheckBox("Add connector pins")
        self.chk_pins.setChecked(True)
        self.chk_pins.setToolTip("Add male/female alignment pins to all cut faces")
        lay.addWidget(self.chk_pins)

        pin_row = QWidget(); prl = QHBoxLayout(pin_row); prl.setContentsMargins(16,0,0,0); prl.setSpacing(6)
        prl.addWidget(QLabel("Radius:"))
        self.spin_pin_radius = QDoubleSpinBox(); self.spin_pin_radius.setRange(1,10); self.spin_pin_radius.setValue(3); self.spin_pin_radius.setSuffix(" mm"); self.spin_pin_radius.setFixedHeight(26)
        prl.addWidget(self.spin_pin_radius)
        prl.addWidget(QLabel("Depth:"))
        self.spin_pin_depth = QDoubleSpinBox(); self.spin_pin_depth.setRange(2,20); self.spin_pin_depth.setValue(6); self.spin_pin_depth.setSuffix(" mm"); self.spin_pin_depth.setFixedHeight(26)
        prl.addWidget(self.spin_pin_depth)
        lay.addWidget(pin_row)

        pin_row2 = QWidget(); pr2l = QHBoxLayout(pin_row2); pr2l.setContentsMargins(16,0,0,0); pr2l.setSpacing(6)
        pr2l.addWidget(QLabel("Pins per face:"))
        self.combo_pin_count = QComboBox(); self.combo_pin_count.addItems(["2","4","Grid"]); self.combo_pin_count.setFixedHeight(26)
        pr2l.addWidget(self.combo_pin_count)
        pr2l.addWidget(QLabel("Tol:"))
        self.spin_pin_tol = QDoubleSpinBox(); self.spin_pin_tol.setRange(0.1,1.0); self.spin_pin_tol.setValue(0.25); self.spin_pin_tol.setSuffix(" mm"); self.spin_pin_tol.setFixedHeight(26)
        pr2l.addWidget(self.spin_pin_tol); pr2l.addStretch()
        lay.addWidget(pin_row2)

        # Part numbering
        self.chk_part_numbers = QCheckBox("Emboss part numbers on back face")
        self.chk_part_numbers.setChecked(False)
        self.chk_part_numbers.setToolTip("Adds raised part number text to the interior face of each part")
        lay.addWidget(self.chk_part_numbers)

        # Cut face labels (A1/A2 matching)
        self.chk_face_labels = QCheckBox("Emboss matching joint labels (A1/A2)")
        self.chk_face_labels.setChecked(False)
        self.chk_face_labels.setToolTip(
            "Emboss matching labels on both sides of each cut face.\n"
            "Helps identify which pieces connect during assembly.\n"
            "e.g. A1↔A2, B1↔B2")
        lay.addWidget(self.chk_face_labels)
        self.btn_preview_labels = QPushButton("Preview Labels")
        self.btn_preview_labels.setObjectName("nudgeBtn"); self.btn_preview_labels.setFixedHeight(26)
        self.btn_preview_labels.setEnabled(False)
        self.btn_preview_labels.setToolTip("Show which faces get which labels without embossing")
        lay.addWidget(self.btn_preview_labels)

        # Part orientation (per-part rotation for optimal printing)
        orient_box = QGroupBox("Part Orientation")
        orient_box.setObjectName("subGroup")
        obl = QVBoxLayout(orient_box); obl.setContentsMargins(8,10,8,8); obl.setSpacing(4)
        orient_info = QLabel("Rotate selected part for optimal print orientation.")
        orient_info.setObjectName("dimLabel"); orient_info.setWordWrap(True); obl.addWidget(orient_info)
        ob_row = QWidget(); ob_rl = QHBoxLayout(ob_row); ob_rl.setContentsMargins(0,0,0,0); ob_rl.setSpacing(4)
        self.btn_orient_flat = QPushButton("Flat Down")
        self.btn_orient_flat.setObjectName("nudgeBtn"); self.btn_orient_flat.setFixedHeight(26)
        self.btn_orient_flat.setToolTip("Rotate so largest flat face is on the bottom")
        ob_rl.addWidget(self.btn_orient_flat)
        self.btn_orient_x90 = QPushButton("X +90°")
        self.btn_orient_x90.setObjectName("nudgeBtn"); self.btn_orient_x90.setFixedHeight(26)
        ob_rl.addWidget(self.btn_orient_x90)
        self.btn_orient_y90 = QPushButton("Y +90°")
        self.btn_orient_y90.setObjectName("nudgeBtn"); self.btn_orient_y90.setFixedHeight(26)
        ob_rl.addWidget(self.btn_orient_y90)
        self.btn_orient_z90 = QPushButton("Z +90°")
        self.btn_orient_z90.setObjectName("nudgeBtn"); self.btn_orient_z90.setFixedHeight(26)
        ob_rl.addWidget(self.btn_orient_z90)
        obl.addWidget(ob_row)

        # Overhang analysis
        self.btn_overhang = QPushButton("Analyse Overhangs")
        self._set_btn_icon(self.btn_overhang, 'resize')
        self.btn_overhang.setObjectName("nudgeBtn"); self.btn_overhang.setFixedHeight(28)
        self.btn_overhang.setToolTip("Check each part for overhangs that need support material")
        obl.addWidget(self.btn_overhang)
        self.lbl_overhang = QLabel("")
        self.lbl_overhang.setObjectName("dimLabel"); self.lbl_overhang.setWordWrap(True)
        obl.addWidget(self.lbl_overhang)

        lay.addWidget(orient_box)

        # Bond strength analysis
        self.btn_bond_analysis = QPushButton("🔗 Analyse Bond Surfaces")
        self.btn_bond_analysis.setObjectName("secondaryBtn"); self.btn_bond_analysis.setFixedHeight(30)
        self.btn_bond_analysis.setEnabled(False)
        self.btn_bond_analysis.setToolTip(
            "Check if cut faces provide enough bonding area.\n"
            "Flags weak joints that need extra reinforcement.")
        lay.addWidget(self.btn_bond_analysis)
        self.lbl_bond = QLabel(""); self.lbl_bond.setObjectName("dimLabel"); self.lbl_bond.setWordWrap(True)
        lay.addWidget(self.lbl_bond)

        # Filament cost
        cost_row = QWidget(); crl = QHBoxLayout(cost_row); crl.setContentsMargins(0,0,0,0); crl.setSpacing(6)
        crl.addWidget(QLabel("Filament cost:"))
        self.spin_filament_cost = QDoubleSpinBox()
        self.spin_filament_cost.setRange(5, 200); self.spin_filament_cost.setValue(25)
        self.spin_filament_cost.setPrefix("$"); self.spin_filament_cost.setSuffix("/kg")
        self.spin_filament_cost.setFixedHeight(26)
        crl.addWidget(self.spin_filament_cost)
        lay.addWidget(cost_row)

        # Dry-fit view button
        # Auto-orient all parts
        self.btn_orient_all = QPushButton("Auto-Orient All Parts (Flat Down)")
        self._set_btn_icon(self.btn_orient_all, 'rotate')
        self.btn_orient_all.setObjectName("secondaryBtn"); self.btn_orient_all.setFixedHeight(30)
        self.btn_orient_all.setEnabled(False)
        self.btn_orient_all.setToolTip(
            "Rotate every part so its largest flat face is on the bottom.\n"
            "Minimises supports for all parts at once.")
        lay.addWidget(self.btn_orient_all)

        # Measurement tool
        self.btn_measure = QPushButton("Measure Distance")
        self._set_btn_icon(self.btn_measure, 'measure')
        self.btn_measure.setObjectName("secondaryBtn"); self.btn_measure.setFixedHeight(28)
        self.btn_measure.setCheckable(True)
        self.btn_measure.setToolTip(
            "Click two points on the model to measure the distance between them.\n"
            "Click button again to exit measure mode.")
        lay.addWidget(self.btn_measure)
        self.lbl_measure = QLabel(""); self.lbl_measure.setObjectName("dimLabel")
        lay.addWidget(self.lbl_measure)

        self.btn_dryfit = QPushButton("Preview Assembly (Dry-Fit)")
        self._set_btn_icon(self.btn_dryfit, 'preview')
        self.btn_dryfit.setObjectName("secondaryBtn"); self.btn_dryfit.setFixedHeight(30)
        self.btn_dryfit.setEnabled(False)
        self.btn_dryfit.setToolTip("Show all parts in their assembled positions in the viewport")
        lay.addWidget(self.btn_dryfit)

        self.btn_export = QPushButton("Export All Parts…")
        self.btn_export.setObjectName("primaryBtn"); self.btn_export.setFixedHeight(36); self.btn_export.setEnabled(False)
        lay.addWidget(self.btn_export)

        self.btn_bambu_export = QPushButton("Export for Bambu Studio (.3mf)")
        self._set_btn_icon(self.btn_bambu_export, 'box')
        self.btn_bambu_export.setObjectName("secondaryBtn"); self.btn_bambu_export.setFixedHeight(34)
        self.btn_bambu_export.setEnabled(False)
        self.btn_bambu_export.setToolTip(
            "Export all parts as a single .3mf file.\n"
            "Parts are auto-oriented and packed onto print plates.\n"
            "Opens directly in Bambu Studio ready to print.")
        lay.addWidget(self.btn_bambu_export)

        self.btn_pdf_export = QPushButton("Assembly Guide PDF")
        self._set_btn_icon(self.btn_pdf_export, 'pdf')
        self.btn_pdf_export.setObjectName("secondaryBtn"); self.btn_pdf_export.setFixedHeight(34)
        self.btn_pdf_export.setEnabled(False)
        self.btn_pdf_export.setToolTip(
            "Generate a PDF assembly guide with:\n"
            "• Parts list with dimensions\n"
            "• Print plate assignments\n"
            "• Assembly order\n"
            "• Print settings & finishing tips")
        lay.addWidget(self.btn_pdf_export)

        # Export presets
        preset_row = QWidget(); prl2 = QHBoxLayout(preset_row); prl2.setContentsMargins(0,0,0,0); prl2.setSpacing(4)
        prl2.addWidget(QLabel("Preset:"))
        self.combo_export_preset = QComboBox()
        self.combo_export_preset.addItems(["(default)"])
        self.combo_export_preset.setFixedHeight(26)
        prl2.addWidget(self.combo_export_preset)
        self.btn_save_preset = QPushButton("Save")
        self.btn_save_preset.setObjectName("nudgeBtn"); self.btn_save_preset.setFixedHeight(26)
        prl2.addWidget(self.btn_save_preset)
        self.btn_load_preset = QPushButton("Load")
        self.btn_load_preset.setObjectName("nudgeBtn"); self.btn_load_preset.setFixedHeight(26)
        prl2.addWidget(self.btn_load_preset)
        lay.addWidget(preset_row)

        self.lbl_export_result = QLabel(""); self.lbl_export_result.setObjectName("dimLabel"); self.lbl_export_result.setWordWrap(True)
        lay.addWidget(self.lbl_export_result)

    def _build_parts_tab(self):
        w = QWidget(); wl = QVBoxLayout(w); wl.setContentsMargins(4,4,4,4)
        self.flat_parts_table = QTableWidget(0,5)
        self.flat_parts_table.setHorizontalHeaderLabels(["Part","X mm","Y mm","Z mm","Triangles"])
        self.flat_parts_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.flat_parts_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.flat_parts_table.setObjectName("partsTable")
        wl.addWidget(self.flat_parts_table)
        self.bottom_tabs.addTab(w,"Parts List")

    def _build_estimate_tab(self):
        w = QWidget(); wl = QVBoxLayout(w); wl.setContentsMargins(4,4,4,4)
        self.estimate_table = QTableWidget(0, 5)
        self.estimate_table.setHorizontalHeaderLabels(["Part", "Time", "Weight (g)", "Filament (g)", "Note"])
        self.estimate_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.estimate_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.estimate_table.setObjectName("partsTable")
        wl.addWidget(self.estimate_table)
        # Bottom row: totals + action buttons
        bot = QWidget(); bl = QHBoxLayout(bot); bl.setContentsMargins(0,2,0,0); bl.setSpacing(6)
        self.lbl_est_total = QLabel(""); self.lbl_est_total.setObjectName("dimLabel")
        bl.addWidget(self.lbl_est_total, 1)
        self.btn_printability = QPushButton("Printability Check")
        self.btn_printability.setObjectName("nudgeBtn"); self.btn_printability.setFixedHeight(24)
        self.btn_printability.setEnabled(False)
        bl.addWidget(self.btn_printability)
        self.btn_wall_check = QPushButton("Wall Thickness")
        self.btn_wall_check.setObjectName("nudgeBtn"); self.btn_wall_check.setFixedHeight(24)
        self.btn_wall_check.setEnabled(False)
        bl.addWidget(self.btn_wall_check)
        wl.addWidget(bot)
        self.bottom_tabs.addTab(w, "Print Estimate")

    def _build_history_tab(self):
        w = QWidget(); wl = QVBoxLayout(w); wl.setContentsMargins(4,4,4,4)
        self.history_list = QTableWidget(0, 3)
        self.history_list.setHorizontalHeaderLabels(["#", "Action", "Detail"])
        self.history_list.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.history_list.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.history_list.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.history_list.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.history_list.setSelectionBehavior(QTableWidget.SelectRows)
        self.history_list.setObjectName("partsTable")
        wl.addWidget(self.history_list)
        # Undo-to-here button
        bot = QWidget(); bl = QHBoxLayout(bot); bl.setContentsMargins(0,2,0,0); bl.setSpacing(6)
        self.btn_undo_to = QPushButton("↩ Undo to Selected")
        self.btn_undo_to.setObjectName("nudgeBtn"); self.btn_undo_to.setFixedHeight(24)
        self.btn_undo_to.setToolTip("Undo all actions down to the selected row")
        bl.addWidget(self.btn_undo_to); bl.addStretch()
        self.lbl_history_count = QLabel("0 actions"); self.lbl_history_count.setObjectName("dimLabel")
        bl.addWidget(self.lbl_history_count)
        wl.addWidget(bot)
        self.bottom_tabs.addTab(w, "History")

    def _build_help_tab(self):
        w = QWidget(); wl = QVBoxLayout(w); wl.setContentsMargins(8,8,8,8)
        info = QLabel(
            "<b>Cutting workflow:</b><br>"
            "1. Import and resize your model<br>"
            "2. Use <b>Auto-Slice</b> to generate a starting grid, OR<br>"
            "3. Select a part in the tree → choose cut mode → position the cut → Apply<br>"
            "4. Each cut splits the selected part into two — both appear in the tree<br>"
            "5. Click any part to select it and make further cuts on just that piece<br>"
            "6. Use <b>↩ Undo</b> to merge two pieces back<br><br>"
            "<b>Cut modes:</b><br>"
            "• <b>Full Slice</b> — cuts all the way through the selected part<br>"
            "• <b>Free (Angled)</b> — full cut at any rotation angle<br>"
            "• <b>Section Cut</b> — cuts only within a defined W×H rectangle<br>"
            "• <b>Groove (Zigzag)</b> — interlocking teeth for mechanical interlock<br><br>"
            "<b>Shortcuts:</b> Ctrl+Z=undo, Ctrl+S=save, Ctrl+O=open, Space=toggle vis, Del=remove cut, R=reset view"
        )
        info.setWordWrap(True); info.setObjectName("infoLabel")
        wl.addWidget(info); wl.addStretch()
        self.bottom_tabs.addTab(w,"Help")

    # ═══════════════════════════════════════════════════════
    # THEME
    # ═══════════════════════════════════════════════════════

    def _apply_theme(self):
        self.setStyleSheet("""
        QMainWindow,QWidget{background:#0f1117;color:#e0e4ed;
            font-family:'Segoe UI','SF Pro Text',Arial,sans-serif;font-size:12px;}
        #leftPanel{background:#111420;border-right:1px solid #1e2130;}
        #titleBar{background:#0d0f18;border-bottom:1px solid #1e2130;}
        #appTitle{font-size:15px;font-weight:700;letter-spacing:3px;color:#4fc3f7;}
        #sidebarScroll,#sidebarContent{background:transparent;border:none;}
        QScrollBar:vertical{background:#13151e;width:5px;border-radius:2px;}
        QScrollBar::handle:vertical{background:#2a2d3d;border-radius:2px;}
        #sectionHeader{background:#1c2438;border:1px solid #2a3550;border-radius:5px;
            color:#7088a8;font-size:11px;font-weight:700;letter-spacing:1px;
            text-align:left;padding-left:12px;
            border-left:3px solid #3060a0;}
        #sectionHeader:checked{background:#1e2a40;color:#a0c0e8;
            border-color:#3a5080;border-left:3px solid #4080d0;}
        #sectionHeader:hover{background:#1e2840;color:#90a8c8;}
        #sectionHeader[stageState="locked"]{background:#14171f;color:#3a4258;
            border-color:#1a1e28;border-left:3px solid #1a2030;}
        #sectionHeader[stageState="active"]{background:#1a3a2a;color:#90e0b0;
            border-color:#2a6048;border-left:3px solid #40c080;}
        #sectionHeader[stageState="active"]:checked{background:#1e4a34;color:#b0f0c8;}
        #sectionHeader[stageState="done"]{color:#6a8878;
            border-left:3px solid #40a068;}
        #sectionHeader:disabled{background:#14171f;color:#3a4258;
            border-color:#1a1e28;border-left:3px solid #1a2030;}
        #sectionContent{background:#161c28;border:1px solid #222e40;border-top:none;
            border-radius:0 0 7px 7px;border-left:3px solid #2a4060;}
        #viewBar{background:#0d1119;border-bottom:1px solid #1a2232;}
        #viewBarLabel{color:#6a7d9a;font-size:10px;font-weight:700;
            letter-spacing:1.5px;text-transform:uppercase;padding-right:6px;}
        #viewBarHint{color:#6af0b8;font-size:10px;font-style:italic;
            letter-spacing:0.3px;}
        #viewBarMode{color:#7a8ca8;font-size:10px;font-weight:700;
            font-family:"Consolas","Courier New",monospace;
            letter-spacing:1.5px;padding:0 10px;
            border-left:1px solid #1e2838;border-right:1px solid #1e2838;}
        #viewBarMode[active="true"]{color:#ffb040;}
        #viewPresetBtn{background:#141a26;color:#8aa0c0;
            border:1px solid #1e2838;border-radius:3px;
            font-family:"Consolas","Courier New",monospace;
            font-size:10px;font-weight:700;letter-spacing:1px;padding:0 8px;}
        #viewPresetBtn:hover{background:#1a2234;color:#d0e0f5;border-color:#2a3852;}
        #viewPresetBtn:checked{background:#1e3550;color:#8ad0ff;
            border:1px solid #3a6090;}
        #viewPresetBtn:pressed{background:#0e1420;}
        #contextBar{background:#131824;border-bottom:1px solid #1e2838;}
        #contextLabel{color:#c0d0e8;font-size:11px;font-weight:600;}
        #ctxBtn{background:#1e2940;color:#a8c0e0;border:1px solid #2a3858;
            border-radius:4px;font-size:11px;font-weight:600;padding:0 10px;}
        #ctxBtn:hover{background:#263452;color:#c8dcf8;border-color:#3a4870;}
        #ctxBtn:pressed{background:#1a2236;}
        #ctxPrimaryBtn{background:qlineargradient(x1:0,y1:0,x2:0,y2:1,
            stop:0 #3a7040,stop:1 #2a5830);
            color:#e0ffe8;border:1px solid #408850;border-radius:4px;
            font-size:11px;font-weight:700;padding:0 12px;}
        #ctxPrimaryBtn:hover{background:qlineargradient(x1:0,y1:0,x2:0,y2:1,
            stop:0 #469050,stop:1 #306a38);}
        #ctxPrimaryBtn:pressed{background:#2a5830;}
        #workflowBar{background:#0f1420;border:1px solid #1e2838;border-radius:6px;}
        #workflowStageLabel{color:#a0b8d0;font-size:11px;font-weight:600;}
        #workflowDot{background:#1a2030;color:#506080;border:1px solid #252e40;
            border-radius:3px;font-size:10px;font-weight:600;padding:0 4px;}
        #workflowDot[stage="locked"]{background:#14171f;color:#3a4258;border-color:#1a1e28;}
        #workflowDot[stage="available"]{background:#1e2840;color:#7090b8;border-color:#2a3a58;}
        #workflowDot[stage="active"]{background:#1a4030;color:#90e0b0;border-color:#2a6048;font-weight:700;}
        #workflowDot[stage="done"]{background:#1a3028;color:#6a9080;border-color:#2a4838;}
        #workflowDot:hover:!disabled{background:#253458;color:#a0c0e0;}
        #sectionBox{background:#161921;border:1px solid #1e2130;border-radius:7px;
            color:#5a6280;font-size:10px;font-weight:600;letter-spacing:1px;margin-top:3px;}
        #sectionBox::title{subcontrol-origin:margin;left:10px;padding:0 5px;}
        #subGroup{background:#131820;border:1px solid #1e2838;border-radius:5px;
            color:#6080a0;font-size:10px;font-weight:600;margin-top:4px;}
        #subGroup::title{subcontrol-origin:margin;left:8px;padding:0 4px;}
        #primaryBtn{background:qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #1e7fd4,stop:1 #1565c0);
            color:white;border:none;border-radius:6px;font-weight:600;}
        #primaryBtn:hover{background:#2991e8;}
        #primaryBtn:pressed{background:#1256a5;}
        #primaryBtn:disabled{background:#232535;color:#4a4f65;}
        #cutBtn{background:qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #8b2020,stop:1 #6b1010);
            color:#ffcccc;border:none;border-radius:6px;font-weight:700;font-size:13px;}
        #cutBtn:hover{background:#a03030;}
        #cutBtn:pressed{background:#6b1010;}
        #cutBtn:disabled{background:#232535;color:#4a4f65;}
        #colourBtn{background:qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #2a4a1a,stop:1 #1a3010);
            color:#80e040;border:1px solid #3a6020;border-radius:5px;font-weight:600;}
        #colourBtn:hover{background:#3a6020;color:#a0f060;}
        #colourBtn:disabled{background:#1a1d2a;color:#3a4a2a;border-color:#252838;}
        #secondaryBtn{background:#1e2130;color:#8eb4e3;border:1px solid #2a3050;border-radius:5px;font-weight:500;}
        #secondaryBtn:hover{background:#252840;}
        #secondaryBtn:disabled{color:#3a3f55;}
        #nudgeBtn{background:#161a28;color:#6a8ab0;border:1px solid #222840;border-radius:4px;font-size:11px;font-weight:600;}
        #nudgeBtn:hover{background:#1e2438;color:#90b8e0;}
        #lockBtn{background:#1a2818;color:#c0a020;border:1px solid #3a4020;border-radius:5px;font-weight:600;}
        #lockBtn:hover{background:#2a3820;color:#e0c040;}
        #toolbarBtn{background:#1a1d2a;color:#7a85a0;border:1px solid #252838;border-radius:5px;padding:2px 8px;}
        #toolbarBtn:hover{background:#20243a;color:#a0b0cc;}
        #cutBar{background:#10121a;border-top:1px solid #1a1d2a;border-bottom:1px solid #1a1d2a;}
        #toolbar{background:#0d0f18;border-bottom:1px solid #1a1d2a;}
        #statusLabel{color:#5a6280;font-size:11px;}
        #treeHeader{background:#283848;border-bottom:1px solid #3a5060;}
        #treePanel{background:#283848;border-right:1px solid #3a5060;}
        #treeHeaderLabel{color:#90a8c0;font-size:10px;font-weight:600;letter-spacing:1px;}
        #visBtn{background:#161a28;color:#5a7090;border:1px solid #1e2438;border-radius:3px;
            font-size:10px;font-weight:600;padding:0px;}
        #visBtn:hover{background:#1e2848;color:#90b8e0;}
        #visSoloBtn{background:#1a1428;color:#c070e0;border:1px solid #2a1848;border-radius:3px;
            font-size:10px;font-weight:600;padding:0px;}
        #visSoloBtn:hover{background:#2a1848;color:#e090ff;}
        #partsTreeWidget{background:#283848;color:#e8ecf4;border:none;font-size:11px;}
        #partsTreeWidget::item{padding:4px 4px;border-bottom:1px solid #344858;}
        #partsTreeWidget::item:selected{background:#3a5888;color:#ffffff;}
        #partsTreeWidget::item:hover{background:#304868;}
        QDoubleSpinBox,QSpinBox,QComboBox{background:#1a1d2a;color:#c8d4e8;
            border:1px solid #252838;border-radius:4px;padding:2px 6px;min-height:22px;}
        QDoubleSpinBox:focus,QComboBox:focus{border-color:#3060b0;}
        QComboBox::drop-down{border:none;width:16px;}
        QComboBox QAbstractItemView{background:#1a1d2a;color:#c8d4e8;
            selection-background-color:#253060;border:1px solid #2a3050;}
        QCheckBox{color:#8892aa;spacing:5px;}
        QCheckBox::indicator{width:13px;height:13px;border:1px solid #2a3050;border-radius:3px;background:#1a1d2a;}
        QCheckBox::indicator:checked{background:#1e7fd4;border-color:#1e7fd4;}
        #dimLabel{color:#7088a0;font-size:11px;}
        #pctLabel{font-size:11px;font-weight:600;}
        #infoLabel{color:#6a7590;font-size:11px;line-height:1.5;}
        QTableWidget{background:#0d0f18;color:#8892aa;gridline-color:#1a1d2a;
            border:1px solid #1a1d2a;border-radius:4px;font-size:11px;}
        QTableWidget::item:selected{background:#1a2d50;color:#c8d8f0;}
        QHeaderView::section{background:#131620;color:#4a5270;border:none;
            border-bottom:1px solid #1e2130;padding:3px;font-size:10px;font-weight:600;}
        QTabWidget::pane{background:#0d0f18;border:1px solid #1a1d2a;}
        QTabBar::tab{background:#13151e;color:#4a5270;border:1px solid #1a1d2a;
            border-bottom:none;padding:3px 10px;font-size:11px;}
        QTabBar::tab:selected{background:#0d0f18;color:#8ab0d8;}
        QStatusBar{background:#0a0c14;color:#3a4060;font-size:10px;}
        QProgressBar{background:#1a1d2a;border:1px solid #252838;border-radius:3px;height:6px;}
        QProgressBar::chunk{background:#1e7fd4;border-radius:3px;}
        """)

    # ═══════════════════════════════════════════════════════
    # SIGNALS
    # ═══════════════════════════════════════════════════════

    def _connect_signals(self):
        # Keyboard shortcuts
        QShortcut(QKeySequence("Ctrl+Z"), self, self._undo)
        QShortcut(QKeySequence("Ctrl+S"), self, self._save_project)
        QShortcut(QKeySequence("Ctrl+O"), self, self._import_model)
        QShortcut(QKeySequence("Ctrl+E"), self, self._export_parts)
        QShortcut(QKeySequence("Space"), self, self._toggle_selected_visibility)
        QShortcut(QKeySequence("Delete"), self, self._delete_selected_preview_cut)
        # ESC — universal "stop doing whatever weird mode I'm in" key.
        # Clears rotate-object, selection-faces, manual-dowel, measure and
        # any in-flight drag. Mirrors the viewport's ESC handler so it
        # fires regardless of which widget has focus.
        QShortcut(QKeySequence("Escape"), self, self._reset_all_interaction_modes)
        # Note: R-key for momentary rotate-object mode is handled directly
        # by the viewport's VTK interactor event filter, not a QShortcut —
        # this lets it track press AND release to toggle the mode cleanly.

        # Explode slider
        self.slider_explode.valueChanged.connect(self._explode_changed)
        # Snap angle config
        self.combo_snap.currentIndexChanged.connect(self._snap_config_changed)

        self.combo_printer.currentTextChanged.connect(self._update_printer_info)
        self.btn_show_heatmap.clicked.connect(self._show_seam_heatmap)
        self.btn_show_creases.clicked.connect(self._show_natural_seams)
        self.btn_clear_heatmap.clicked.connect(self._clear_heatmap)
        self.seam_suggest_table.currentItemChanged.connect(
            lambda cur, _: self._apply_seam_suggestion(self.seam_suggest_table.currentRow()))
        self.btn_import.clicked.connect(self._import_model)
        self.btn_ai_generate.clicked.connect(self._ai_generate)
        self.btn_text_to_3d.clicked.connect(self._text_to_3d)
        self.btn_save.clicked.connect(self._save_project)
        self.btn_load.clicked.connect(self._load_project)
        self.btn_undo.clicked.connect(self._undo)
        self.btn_apply_resize.clicked.connect(self._apply_resize)
        self.btn_repair.clicked.connect(self._run_full_repair)
        self.btn_diagnose.clicked.connect(self._run_diagnose)
        self.btn_decimate.clicked.connect(self._run_decimate)
        self.btn_subdivide.clicked.connect(self._run_subdivide)
        self.btn_pymeshfix.clicked.connect(lambda: self._run_individual_repair('fix_with_pymeshfix'))
        self.btn_print_ready.clicked.connect(self._run_print_ready_repair)
        self.btn_thin_heatmap.toggled.connect(self._toggle_thin_heatmap)
        self.btn_edge_flip.clicked.connect(self._run_edge_flip)
        self.btn_decimate_pro.clicked.connect(self._run_decimate_pro)
        self.btn_isotropic.clicked.connect(lambda: self._run_remesh('isotropic'))
        self.btn_adaptive.clicked.connect(lambda: self._run_remesh('adaptive'))
        self.btn_remesh_info.clicked.connect(self._show_remesh_info)
        self.btn_workflow.clicked.connect(self._show_workflow_guide)
        self.btn_remove_shells.clicked.connect(self._run_remove_shells)
        # Individual repair steps
        self._repair_buttons['fix_degenerate'].clicked.connect(
            lambda: self._run_individual_repair('fix_degenerate_faces'))
        self._repair_buttons['fix_duplicates'].clicked.connect(
            lambda: self._run_individual_repair('fix_duplicate_faces'))
        self._repair_buttons['fix_normals'].clicked.connect(
            lambda: self._run_individual_repair('fix_normals'))
        self._repair_buttons['fix_holes'].clicked.connect(
            lambda: self._run_individual_repair('fix_holes'))
        self._repair_buttons['fix_merge'].clicked.connect(
            lambda: self._run_individual_repair('fix_merge_vertices'))
        # Region selection + repair
        self.btn_select_mode.toggled.connect(self._toggle_selection_mode)
        self.btn_select_clear.clicked.connect(self._clear_selection)
        self.btn_select_problems.clicked.connect(self._auto_select_problems)
        self.btn_region_fix_normals.clicked.connect(self._region_fix_normals)
        self.btn_region_smooth.clicked.connect(self._region_smooth)
        self.btn_region_delete.clicked.connect(self._region_delete)
        self.spin_brush_radius.valueChanged.connect(
            lambda v: setattr(self.viewport, 'selection_brush_radius', v))
        self.viewport.faces_selected.connect(self._on_faces_selected)
        # Quick-cut toolbar
        self.btn_qc_cut.clicked.connect(self._quick_cut)
        self.btn_qc_m10.clicked.connect(lambda: self.spin_qc_pos.setValue(self.spin_qc_pos.value() - 10))
        self.btn_qc_m1.clicked.connect(lambda: self.spin_qc_pos.setValue(self.spin_qc_pos.value() - 1))
        self.btn_qc_p1.clicked.connect(lambda: self.spin_qc_pos.setValue(self.spin_qc_pos.value() + 1))
        self.btn_qc_p10.clicked.connect(lambda: self.spin_qc_pos.setValue(self.spin_qc_pos.value() + 10))
        self.spin_qc_pos.valueChanged.connect(self._quick_cut_preview)
        self.btn_smooth.clicked.connect(self._run_smoothing)
        # Auto-slice preview
        self.btn_preview_cuts.clicked.connect(self._preview_cuts)
        self.btn_apply_offset.clicked.connect(self._apply_master_offset)
        self.btn_pn_m10.clicked.connect(lambda: self._nudge_preview(-10))
        self.btn_pn_m1.clicked.connect(lambda:  self._nudge_preview(-1))
        self.btn_pn_p1.clicked.connect(lambda:  self._nudge_preview(1))
        self.btn_pn_p10.clicked.connect(lambda: self._nudge_preview(10))
        self.btn_lock_preview.clicked.connect(self._lock_preview_cut)
        self.btn_unlock_preview.clicked.connect(self._unlock_preview_cut)
        self.btn_unlock_all_preview.clicked.connect(self._unlock_all_preview)
        self.preview_cut_list.currentItemChanged.connect(
            lambda cur, prev: self._preview_cut_selected(self.preview_cut_list.currentRow()))
        self.btn_auto_slice.clicked.connect(self._auto_slice)
        # Gizmo signals from viewport
        self.viewport.cut_moved.connect(self._on_gizmo_moved)
        self.viewport.cut_rotated.connect(self._on_gizmo_rotated)
        # Cut section
        self.btn_apply_cut.clicked.connect(self._apply_cut)
        self.btn_export.clicked.connect(self._export_parts)
        self.btn_bambu_export.clicked.connect(self._export_bambu)
        self.btn_pdf_export.clicked.connect(self._export_pdf)
        self.btn_dryfit.clicked.connect(self._toggle_dryfit)
        self.btn_colour_split.clicked.connect(self._colour_split)
        self.btn_reset_view.clicked.connect(lambda: self.viewport.keyPressEvent(
            type('E',(),{'key': lambda s: Qt.Key_R})()))
        self.chk_wireframe.toggled.connect(lambda v: setattr(self.viewport,'show_wireframe',v) or self.viewport.update())
        self.chk_sel_wire.toggled.connect(lambda v: setattr(self.viewport,'show_selection_wireframe',v) or self.viewport.update())
        self.chk_preview.toggled.connect(lambda v: setattr(self.viewport,'show_cut_preview',v) or self.viewport.update())
        self.combo_cut_mode.currentIndexChanged.connect(self._cut_mode_changed)
        self.combo_cut_axis.currentIndexChanged.connect(lambda _: self._update_cut_preview())
        self.spin_cut_pos.valueChanged.connect(lambda _: self._update_cut_preview())
        self.spin_rot_x.valueChanged.connect(lambda _: self._update_cut_preview())
        self.spin_rot_y.valueChanged.connect(lambda _: self._update_cut_preview())
        self.spin_rot_z.valueChanged.connect(lambda _: self._update_cut_preview())
        self.spin_sec_w.valueChanged.connect(lambda _: self._update_cut_preview())
        self.spin_sec_h.valueChanged.connect(lambda _: self._update_cut_preview())
        self.btn_cm10.clicked.connect(lambda: self._nudge_cut(-10))
        self.btn_cm1.clicked.connect(lambda:  self._nudge_cut(-1))
        self.btn_cp1.clicked.connect(lambda:  self._nudge_cut(1))
        self.btn_cp10.clicked.connect(lambda: self._nudge_cut(10))
        self.btn_show_all.clicked.connect(self._show_all_parts)
        self.btn_hide_all.clicked.connect(self._hide_all_parts)
        self.btn_solo.clicked.connect(self._solo_selected_part)
        self.parts_tree_widget.currentItemChanged.connect(self._tree_selection_changed)
        self.parts_tree_widget.itemClicked.connect(self._tree_item_clicked)
        self.combo_joint_type.currentIndexChanged.connect(self._update_joint_groups)
        self.btn_apply_joint.clicked.connect(self._apply_joint_config)
        # Live joint preview when spinners change
        for spin in [self.spin_rod_radius, self.spin_depth_a, self.spin_depth_b,
                     self.spin_rod_count, self.spin_rod_tol,
                     self.spin_rect_w, self.spin_rect_h,
                     self.spin_rect_depth_a, self.spin_rect_depth_b,
                     self.spin_rect_count, self.spin_rect_tol,
                     self.spin_dove_size]:
            spin.valueChanged.connect(lambda _: self._update_joint_preview())
        self.btn_clear_dowels.clicked.connect(self._clear_manual_dowels)
        self.chk_manual_dowels.toggled.connect(self._toggle_manual_dowel_mode)
        self.viewport.dowel_placed.connect(self._on_dowel_placed)
        # Export presets
        self.btn_save_preset.clicked.connect(self._save_export_preset)
        self.btn_load_preset.clicked.connect(self._load_export_preset)
        # Part orientation
        self.btn_overhang.clicked.connect(self._run_overhang_analysis)
        self.btn_bond_analysis.clicked.connect(self._run_bond_analysis)
        self.btn_printability.clicked.connect(self._run_printability_check)
        self.btn_wall_check.clicked.connect(self._run_wall_check)
        self.btn_undo_to.clicked.connect(self._undo_to_selected)
        self.btn_orient_all.clicked.connect(self._orient_all_parts)
        self.btn_measure.toggled.connect(self._toggle_measure_mode)
        self.viewport.measure_clicked.connect(self._on_measure_click)
        self.viewport.part_left_clicked.connect(self._viewport_select_part)
        self.viewport.part_right_clicked.connect(self._on_viewport_part_picked)
        # Context menu + double-click + hover on tree
        self.parts_tree_widget.customContextMenuRequested.connect(self._tree_context_menu)
        self.parts_tree_widget.itemDoubleClicked.connect(self._tree_double_clicked)
        # Hover to preview — temporarily shows a hidden part when hovering its name
        self.parts_tree_widget.itemEntered.connect(self._tree_item_hovered)
        # Build volume toggle
        self.chk_build_vol.toggled.connect(self._toggle_build_volume)
        # Quick-cut mode colour
        self.combo_qc_mode.currentIndexChanged.connect(self._update_qc_bar_colour)

        # ── Cut-control two-way sync ───────────────────────
        # The Quick Cut bar (below the viewport) and the Advanced Options
        # sidebar section BOTH have axis/position/mode controls. Before this
        # sync they diverged silently — the orange preview followed the
        # sidebar while the Cut button on the quick bar used its own values,
        # so e.g. setting Groove/Y in the sidebar and clicking Cut gave a
        # Full/X cut. Now they mirror each other in both directions.
        self._cut_sync_guard = False
        self.combo_qc_axis.currentIndexChanged.connect(
            lambda _i: self._sync_cut_controls('qc_axis'))
        self.combo_cut_axis.currentIndexChanged.connect(
            lambda _i: self._sync_cut_controls('cut_axis'))
        self.spin_qc_pos.valueChanged.connect(
            lambda _v: self._sync_cut_controls('qc_pos'))
        self.spin_cut_pos.valueChanged.connect(
            lambda _v: self._sync_cut_controls('cut_pos'))
        self.combo_qc_mode.currentIndexChanged.connect(
            lambda _i: self._sync_cut_controls('qc_mode'))
        self.combo_cut_mode.currentIndexChanged.connect(
            lambda _i: self._sync_cut_controls('cut_mode'))
        self.btn_preview_labels.clicked.connect(self._preview_face_labels)
        self.btn_tol_test.clicked.connect(self._export_tolerance_test)
        self.btn_orient_flat.clicked.connect(self._orient_flat_down)
        self.btn_orient_x90.clicked.connect(lambda: self._orient_rotate(0, 90))
        self.btn_orient_y90.clicked.connect(lambda: self._orient_rotate(1, 90))
        self.btn_orient_z90.clicked.connect(lambda: self._orient_rotate(2, 90))
        self.chk_hollow.toggled.connect(self._update_hollow_info)
        self.spin_wall.valueChanged.connect(self._update_hollow_info)
        self.chk_uniform.toggled.connect(lambda v: self.combo_lock_axis.setEnabled(v))
        self.spin_x.valueChanged.connect(self._x_changed)

    # ═══════════════════════════════════════════════════════
    # IMPORT
    # ═══════════════════════════════════════════════════════

    # ═══════════════════════════════════════════════════════
    # AI GENERATE FROM PHOTO
    # ═══════════════════════════════════════════════════════

    def _ai_generate(self):
        """Open photo(s) and generate a 3D mesh using AI."""
        # Check backends first
        from core.ai_generate import check_backends
        status = check_backends()

        if not status['gpu_available']:
            QMessageBox.warning(self, "No GPU",
                "AI generation requires an NVIDIA GPU.\n"
                "This machine doesn't have one.\n\n"
                "Run this on your home PC with the RTX 4070 Ti Super instead.")
            return

        if not status['sam3d'] and not status['triposr']:
            reply = QMessageBox.question(self, "AI Backends Not Installed",
                "No AI backends found.\n\n"
                "Run setup_ai_backends.bat to install TripoSR and/or SAM 3D Objects.\n\n"
                "Would you like to open the setup folder?",
                QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                import subprocess
                subprocess.Popen(['explorer', os.path.dirname(os.path.dirname(
                    os.path.abspath(__file__)))])
            return

        # Let user pick 1-4 photos
        paths, _ = QFileDialog.getOpenFileNames(self, "Select Photo(s) — up to 4 for better accuracy", "",
            "Images (*.jpg *.jpeg *.png *.bmp *.webp);;All Files (*)")
        if not paths: return
        if len(paths) > 4:
            paths = paths[:4]
            self.status_bar.showMessage("Using first 4 images only.")

        # Pick backend
        backend_map = {0: 'auto', 1: 'trellis2', 2: 'sam3d', 3: 'partcrafter', 4: 'triposr'}
        backend = backend_map.get(self.combo_ai_backend.currentIndex(), 'auto')

        # Output directory
        import tempfile
        output_dir = tempfile.mkdtemp(prefix='3dprintslicer_ai_')

        self.lbl_ai_status.setText(f"Generating 3D from {len(paths)} photo(s)…")
        self._busy(True, f"AI generating 3D mesh from {len(paths)} photo(s)…")
        self.btn_ai_generate.setEnabled(False)

        self._ai_worker = AIGenerateWorker(paths, output_dir, backend)
        self._ai_worker.finished.connect(self._ai_generate_done)
        self._ai_worker.error.connect(self._ai_generate_error)
        self._ai_worker.start()

    def _ai_generate_done(self, obj_path, msg):
        """AI generation completed — load the mesh."""
        self._busy(False)
        self.btn_ai_generate.setEnabled(True)
        self.lbl_ai_status.setText(f"✓ {msg}")

        # Load the generated OBJ into 3D Print Slicer
        self._busy(True, "Loading generated mesh…")
        self._load_w = LoadWorker(self.mesh_handler, obj_path)
        self._load_w.finished.connect(self._ai_import_done)
        self._load_w.error.connect(
            lambda e: (self._busy(False), self.btn_ai_generate.setEnabled(True),
                       QMessageBox.critical(self, "Load Failed", e)))
        self._load_w.start()

    def _ai_import_done(self, ok, path):
        """AI-generated mesh loaded — run auto-repair since AI meshes need it."""
        self._import_done(ok, path)
        if ok:
            self.lbl_ai_status.setText(
                f"✓ Generated and loaded. Running auto-repair…")
            # AI meshes always need repair
            self._run_full_repair(silent=True)
            self.lbl_ai_status.setText(
                f"✓ AI model ready — resize to target dimensions, then slice.")

    def _ai_generate_error(self, msg):
        self._busy(False)
        self.btn_ai_generate.setEnabled(True)
        self.lbl_ai_status.setText(f"⚠ {msg}")
        QMessageBox.warning(self, "AI Generation Failed", msg)

    def _text_to_3d(self):
        """Generate a 3D model from a text description using OpenSCAD."""
        from PyQt5.QtWidgets import QInputDialog
        text, ok = QInputDialog.getText(self, "Text to 3D",
            "Describe the shape you want:\n\n"
            "Examples:\n"
            "  box 100x80x60\n"
            "  cylinder 50mm diameter 100mm tall\n"
            "  sphere 40mm\n"
            "  box 200x150x100 with rounded corners\n\n"
            "Shape description:")
        if not ok or not text.strip(): return

        import tempfile
        output_dir = tempfile.mkdtemp(prefix='3dprintslicer_text_')
        self._busy(True, "Generating 3D from text…")
        try:
            from core.ai_generate import generate_from_text_openscad
            stl_path, msg = generate_from_text_openscad(text.strip(), output_dir)
            self._busy(False)
            if stl_path:
                self.lbl_ai_status.setText(f"✓ {msg}")
                # Load the generated STL
                self._busy(True, "Loading generated model…")
                self._load_w = LoadWorker(self.mesh_handler, stl_path)
                self._load_w.finished.connect(self._import_done)
                self._load_w.error.connect(
                    lambda e: (self._busy(False),
                               QMessageBox.critical(self, "Load Failed", e)))
                self._load_w.start()
            else:
                self.lbl_ai_status.setText(f"⚠ {msg}")
                QMessageBox.warning(self, "Text to 3D Failed", msg)
        except Exception as e:
            self._busy(False)
            QMessageBox.warning(self, "Error", str(e))

    def _import_model(self):
        path, _ = QFileDialog.getOpenFileName(self,"Import 3D Model","",
            "3D Files (*.stl *.obj *.3mf);;STL (*.stl);;OBJ (*.obj);;3MF (*.3mf)")
        if not path: return
        self._busy(True, "Loading model…")
        self.btn_import.setEnabled(False)
        self._load_w = LoadWorker(self.mesh_handler, path)
        self._load_w.finished.connect(self._import_done)
        self._load_w.error.connect(
            lambda e: (self._busy(False), self.btn_import.setEnabled(True),
                       QMessageBox.critical(self, "Import Failed", e)))
        self._load_w.start()

    def _import_done(self, ok, path):
        self._busy(False)
        self.btn_import.setEnabled(True)
        if not ok:
            QMessageBox.critical(self,"Import Failed",f"Could not load:\n{path}"); return

        fname = os.path.basename(path)
        dims = self.mesh_handler.get_dimensions_mm()
        orig = self.mesh_handler.original_dims()
        tris = self.mesh_handler.get_triangle_count()
        wt = self.mesh_handler.is_watertight()

        self.lbl_file.setText(fname)
        self.lbl_mesh_info.setText(f"{dims[0]:.0f}×{dims[1]:.0f}×{dims[2]:.0f} mm  |  {tris:,} tris  |  {'✓ Watertight' if wt else '⚠ Not watertight'}")
        self.lbl_orig.setText(f"Original: {orig[0]:.1f}×{orig[1]:.1f}×{orig[2]:.1f} mm")
        self.lbl_repair.setText("✓ Watertight" if wt else "⚠ Mesh has issues — run Auto-Repair.")

        self._lock_signals = True
        self.spin_x.setValue(round(dims[0],1)); self.spin_y.setValue(round(dims[1],1)); self.spin_z.setValue(round(dims[2],1))
        self._lock_signals = False
        self._update_pct_labels()

        # Load into part tree
        self.part_tree.load_mesh(self.mesh_handler.mesh, os.path.splitext(fname)[0])
        self._refresh_tree()

        # Show in viewport
        verts, faces = self.mesh_handler.get_vertex_array()
        self.viewport.set_mesh(verts, faces, self.mesh_handler.get_bounds())

        # Set cut position to centre of model
        bounds = self.mesh_handler.get_bounds()
        centre = (bounds[0] + bounds[1]) / 2
        self.spin_cut_pos.setValue(round(float(centre[0]), 1))

        self.btn_repair.setEnabled(True)
        self.btn_diagnose.setEnabled(True)
        self.btn_decimate.setEnabled(True)
        self.btn_subdivide.setEnabled(True)
        self.btn_remove_shells.setEnabled(True)
        self.btn_pymeshfix.setEnabled(True)
        self.btn_print_ready.setEnabled(True)
        self.btn_thin_heatmap.setEnabled(True)
        self.btn_edge_flip.setEnabled(True)
        self.btn_decimate_pro.setEnabled(True)
        self.btn_isotropic.setEnabled(True)
        self.btn_adaptive.setEnabled(True)
        for btn in self._repair_buttons.values():
            btn.setEnabled(True)
        self.btn_smooth.setEnabled(True)

        # Auto-repair on import if enabled
        if self.chk_auto_repair.isChecked():
            self._run_full_repair(silent=True)
        self.btn_show_heatmap.setEnabled(True)
        self.btn_show_creases.setEnabled(True)
        self.btn_clear_heatmap.setEnabled(True)
        self.btn_auto_slice.setEnabled(True)
        self.btn_preview_cuts.setEnabled(True)
        self.btn_apply_cut.setEnabled(True)
        self.btn_qc_cut.setEnabled(True)

        # Set quick-cut position to model centre — block signals so this
        # doesn't spawn a cut-plane preview before the user has even asked
        # for one. The preview should only appear when the user explicitly
        # interacts with cut controls.
        bounds = self.mesh_handler.get_bounds()
        if bounds is not None:
            centre = (bounds[0] + bounds[1]) / 2
            self.spin_qc_pos.blockSignals(True)
            self.spin_qc_pos.setValue(round(float(centre[0]), 1))
            self.spin_qc_pos.blockSignals(False)

        # Check for colour data
        raw = getattr(self.mesh_handler, 'raw_loaded', None)
        has_colour = (has_multi_geometry(raw) if raw is not None
                      else has_colour_data(self.mesh_handler.mesh))
        self.btn_colour_split.setEnabled(has_colour)
        if has_colour:
            self.lbl_colour_info.setText("✓ Colour / material data detected — ready to split")
        else:
            self.lbl_colour_info.setText("No colour data found in this file")

        self.lbl_status.setText(f"{fname}")
        self.status_bar.showMessage(f"Loaded: {dims[0]:.0f}×{dims[1]:.0f}×{dims[2]:.0f} mm")
        # Intentionally NOT calling _update_cut_preview() here — the user
        # hasn't asked to cut anything yet. The preview will appear the
        # moment they change a cut control.

        # Advance workflow to "prepare" stage
        self._set_workflow_stage('prepare')

        # Auto-expand relevant sidebar sections
        if hasattr(self, '_sections'):
            self._sections.get('resize', None) and self._sections['resize'].expand()
            self._sections.get('slice', None) and self._sections['slice'].expand()

    # ═══════════════════════════════════════════════════════
    # RESIZE
    # ═══════════════════════════════════════════════════════

    def _apply_resize(self):
        if self.mesh_handler.mesh is None: return
        # Save undo snapshot before resize
        if self.part_tree.root is not None:
            self.part_tree.push_mesh_snapshot(
                [self.part_tree.root], "resize")
            self.btn_undo.setEnabled(True)
            self._update_undo_tooltip()
        self.mesh_handler.target_x = self.spin_x.value()
        self.mesh_handler.target_y = self.spin_y.value()
        self.mesh_handler.target_z = self.spin_z.value()
        if self.chk_uniform.isChecked():
            axis = ['x','y','z'][self.combo_lock_axis.currentIndex()]
            self.mesh_handler.apply_uniform_resize(lock_axis=axis)
            self._lock_signals = True
            self.spin_x.setValue(round(self.mesh_handler.target_x,1))
            self.spin_y.setValue(round(self.mesh_handler.target_y,1))
            self.spin_z.setValue(round(self.mesh_handler.target_z,1))
            self._lock_signals = False
        else:
            self.mesh_handler.apply_resize()
        self._update_pct_labels()
        self.part_tree.load_mesh(self.mesh_handler.mesh,
            self.part_tree.root.label if self.part_tree.root else "Body")
        self._refresh_tree()
        verts, faces = self.mesh_handler.get_vertex_array()
        self.viewport.set_mesh(verts, faces, self.mesh_handler.get_bounds())
        self._update_cut_preview()
        self.status_bar.showMessage(f"Resized to {self.mesh_handler.get_dimensions_mm()[0]:.0f}×{self.mesh_handler.get_dimensions_mm()[1]:.0f}×{self.mesh_handler.get_dimensions_mm()[2]:.0f} mm")

    def _update_pct_labels(self):
        px, py, pz = self.mesh_handler.pct_change()
        for pct, lbl in [(px,self.lbl_pct_x),(py,self.lbl_pct_y),(pz,self.lbl_pct_z)]:
            if abs(pct) < 0.05: lbl.setText("—"); lbl.setStyleSheet("color:#4a5270;")
            elif pct > 0: lbl.setText(f"+{pct:.1f}%"); lbl.setStyleSheet("color:#50c870;font-weight:600;")
            else: lbl.setText(f"{pct:.1f}%"); lbl.setStyleSheet("color:#e05555;font-weight:600;")

    def _x_changed(self, val):
        if self._lock_signals or not self.chk_uniform.isChecked() or self.mesh_handler.mesh is None: return
        orig = self.mesh_handler.original_dims()
        if orig[0] == 0: return
        factor = val / orig[0]
        self._lock_signals = True
        self.spin_y.setValue(round(orig[1]*factor,1)); self.spin_z.setValue(round(orig[2]*factor,1))
        self._lock_signals = False

    # ═══════════════════════════════════════════════════════
    # REPAIR
    # ═══════════════════════════════════════════════════════

    def _update_printer_info(self):
        """Update the cut size spinner and info label based on selected printer profile."""
        from core.printer_profiles import get_profile
        name = self.combo_printer.currentText() if hasattr(self, 'combo_printer') else 'Bambu P2S'
        profile = get_profile(name)
        is_custom = (name == "Custom")
        if hasattr(self, 'custom_size_row'):
            self.custom_size_row.setVisible(is_custom)
        if hasattr(self, 'lbl_printer_info'):
            self.lbl_printer_info.setText(
                f"{profile.build_x:.0f}×{profile.build_y:.0f}×{profile.build_z:.0f}mm  "
                f"→ rec. cut size: {profile.recommended_cut_size:.0f}mm")
        # Auto-update cut size spinner
        if hasattr(self, 'spin_auto_size') and not is_custom:
            self.spin_auto_size.setValue(profile.recommended_cut_size)

    def _get_printer_profile(self):
        """Return current printer profile."""
        from core.printer_profiles import get_profile
        name = self.combo_printer.currentText() if hasattr(self, 'combo_printer') else 'Bambu P2S'
        return get_profile(name)

    def _show_seam_heatmap(self):
        """Compute and display seam quality heatmap on the model."""
        if self.mesh_handler.mesh is None: return
        self._busy(True, "Computing seam heatmap…")
        self._seam_w = SeamWorker(self.mesh_handler.mesh)
        self._seam_w.finished.connect(self._seam_heatmap_done)
        self._seam_w.error.connect(
            lambda e: (self._busy(False),
                       QMessageBox.critical(self, "Heatmap Error", e)))
        self._seam_w.start()

    def _seam_heatmap_done(self, scores_rgba, seam_lines, suggestions):
        self._busy(False)
        self._last_seam_suggestions = suggestions
        # Feed crease snap data to viewport for snap-to-crease cutting
        self.viewport.crease_snap_points = suggestions if suggestions else []
        # Show heatmap on the mesh (need raw vertex array)
        verts, faces = self.mesh_handler.get_vertex_array()
        self.viewport.set_mesh(verts, faces, self.mesh_handler.get_bounds())
        self.viewport.set_heatmap(scores_rgba)
        self.viewport.update()

        # Populate suggestions table
        self.seam_suggest_table.setRowCount(0)
        self.seam_suggest_table.setVisible(bool(suggestions))
        if suggestions:
            self.lbl_seam_suggestions.setText(
                f"Found {len(suggestions)} natural cut positions — click to apply:")
            for s in suggestions[:10]:
                row = self.seam_suggest_table.rowCount()
                self.seam_suggest_table.insertRow(row)
                ax_item = QTableWidgetItem(s['axis'].upper())
                ax_item.setForeground(QColor(
                    {'x':'#e05555','y':'#50c870','z':'#5588e8'}.get(s['axis'],'#8892aa')))
                pos_item = QTableWidgetItem(f"{s['position']:.0f} mm")
                str_item = QTableWidgetItem(f"{min(s['score']/100,1)*100:.0f}%")
                for item in [ax_item, pos_item, str_item]:
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.seam_suggest_table.setItem(row, 0, ax_item)
                self.seam_suggest_table.setItem(row, 1, pos_item)
                self.seam_suggest_table.setItem(row, 2, str_item)

        self.status_bar.showMessage(
            "Seam heatmap: green=good cut, red=bad cut. "
            f"{len(suggestions)} natural seam positions found.")

    def _show_natural_seams(self):
        """Highlight crease lines on the model."""
        if self.mesh_handler.mesh is None: return
        self._busy(True, "Finding natural seams…")
        self._seam_w2 = SeamWorker(self.mesh_handler.mesh)
        self._seam_w2.finished.connect(
            lambda rgba, seams, sug: self._natural_seams_done(seams))
        self._seam_w2.error.connect(
            lambda e: (self._busy(False),
                       QMessageBox.critical(self, "Seam Error", e)))
        self._seam_w2.start()

    def _natural_seams_done(self, seam_lines):
        self._busy(False)
        self.viewport.set_crease_lines(seam_lines)
        self.viewport.update()
        self.status_bar.showMessage(
            f"Found {len(seam_lines)} crease edges — cyan lines show natural panel seams.")

    def _clear_heatmap(self):
        self.viewport.clear_heatmap()
        self.lbl_seam_suggestions.setText("")
        self.seam_suggest_table.setVisible(False)
        self.seam_suggest_table.setRowCount(0)
        # Restore normal mesh view
        if self.part_tree.get_all_leaves():
            self._refresh_viewport()
        else:
            verts, faces = self.mesh_handler.get_vertex_array()
            self.viewport.set_mesh(verts, faces, self.mesh_handler.get_bounds())
        self.status_bar.showMessage("Heatmap cleared.")

    def _apply_seam_suggestion(self, row):
        """Click a suggestion → set the cut position spinner to that location."""
        if not hasattr(self, '_last_seam_suggestions'): return
        suggestions = self._last_seam_suggestions
        if row < 0 or row >= len(suggestions): return
        s = suggestions[row]
        axis = s['axis']
        pos  = s['position']
        # Set the cut section controls
        axis_map = {'x':0,'y':1,'z':2}
        if hasattr(self, 'combo_cut_axis'):
            self.combo_cut_axis.setCurrentIndex(axis_map.get(axis, 0))
        if hasattr(self, 'spin_cut_pos'):
            self.spin_cut_pos.setValue(pos)
        self.status_bar.showMessage(
            f"Cut position set: {axis.upper()} = {pos:.0f}mm (from seam suggestion)")

    def _run_full_repair(self, silent=False):
        """Run the comprehensive auto-repair pipeline."""
        if self.mesh_handler.mesh is None: return
        if self.part_tree.root is not None and not silent:
            self.part_tree.push_mesh_snapshot([self.part_tree.root], "repair")
            self.btn_undo.setEnabled(True); self._update_undo_tooltip()
        if not silent:
            self._busy(True, "Running full auto-repair…")
        try:
            from core.auto_repair import full_repair
            mesh, report = full_repair(self.mesh_handler.mesh)
            self.mesh_handler.mesh = mesh
            self.mesh_handler.original_mesh = mesh.copy()
            if not silent:
                self._busy(False)
            self._update_health_display(report)
            self.part_tree.load_mesh(mesh, self.part_tree.root.label if self.part_tree.root else "Body")
            self._refresh_tree()
            verts, faces = self.mesh_handler.get_vertex_array()
            self.viewport.set_mesh(verts, faces, self.mesh_handler.get_bounds())
            if not silent:
                self.status_bar.showMessage(
                    f"Repair: {report.total_fixes} fixes, health {report.health_score}/100")
            else:
                self.status_bar.showMessage(
                    f"Auto-repaired on import: {report.total_fixes} fixes, "
                    f"health {report.health_score}/100")
        except Exception as e:
            if not silent:
                self._busy(False)
            self.lbl_repair.setText(f"Repair error: {e}")

    def _run_print_ready_repair(self):
        """Full auto-repair PLUS aggressive passes (self-intersection cleanup,
        non-manifold splitting, sliver removal) and a thin-wall scan. This
        is the 'prep a file for the printer' one-click button."""
        if self.mesh_handler.mesh is None: return
        if self.part_tree.root is not None:
            self.part_tree.push_mesh_snapshot([self.part_tree.root], "print-ready repair")
            self.btn_undo.setEnabled(True); self._update_undo_tooltip()
        self._busy(True, "Running print-ready repair — this may take a moment…")
        try:
            from core.auto_repair import full_repair
            mesh, report = full_repair(
                self.mesh_handler.mesh,
                aggressive=True, print_ready=True,
                check_self_intersections=False,  # manifold pass covers this
                min_wall_mm=0.8,
            )
            self.mesh_handler.mesh = mesh
            self.mesh_handler.original_mesh = mesh.copy()
            self._busy(False)
            self._update_health_display(report)
            self.part_tree.load_mesh(
                mesh, self.part_tree.root.label if self.part_tree.root else "Body")
            self._refresh_tree()
            verts, faces = self.mesh_handler.get_vertex_array()
            self.viewport.set_mesh(verts, faces, self.mesh_handler.get_bounds())

            # Surface any warnings (thin walls, leftover non-manifold) in a
            # proper dialog so the user knows what still needs attention.
            unfixed = [i for i in report.issues
                       if not i.fixed and i.severity in ('error', 'warning')]
            if unfixed:
                lines = [f"• {i.description}" for i in unfixed]
                QMessageBox.warning(
                    self, "Print-Ready Repair — Manual attention needed",
                    f"Fixed {report.total_fixes} issues, health {report.health_score}/100.\n\n"
                    "The following still need attention:\n\n" + "\n".join(lines))
            self.status_bar.showMessage(
                f"Print-ready repair: {report.total_fixes} fixes applied, "
                f"health {report.health_score}/100, "
                f"{'watertight' if report.is_watertight else 'NOT watertight'}.")
        except Exception as e:
            self._busy(False)
            QMessageBox.critical(self, "Print-Ready Repair Failed", str(e))

    def _toggle_thin_heatmap(self, on):
        """Paint the mesh with a green/yellow/red thickness heatmap, or
        clear it. Runs in a thread — the ray queries take a couple of
        seconds on dense models."""
        mesh, _name = self._get_target_mesh()
        if mesh is None:
            self.btn_thin_heatmap.blockSignals(True)
            self.btn_thin_heatmap.setChecked(False)
            self.btn_thin_heatmap.blockSignals(False)
            return
        if not on:
            self.viewport.clear_heatmap()
            self.status_bar.showMessage("Thin-wall heatmap cleared.")
            return

        self._busy(True, "Computing wall thickness — please wait…")

        def do_work():
            from core.mesh_quality import thin_wall_heatmap
            return thin_wall_heatmap(mesh, min_thickness=0.8)

        def on_done(rgba):
            self._busy(False)
            try:
                self.viewport.set_heatmap(rgba)
                self.status_bar.showMessage(
                    "Thin-wall heatmap on — green=safe, yellow=caution, "
                    "red=will fail (< 0.8 mm). Un-toggle to hide.")
            except Exception as e:
                QMessageBox.warning(self, "Heatmap Failed", str(e))

        self._run_threaded(do_work, (), "Computing wall thickness…", on_done)

    def _run_edge_flip(self):
        """Run edge-flip optimisation on the selected mesh."""
        mesh, target_name = self._get_target_mesh()
        if mesh is None: return
        parts, _ = self._get_target_parts()
        if parts:
            self.part_tree.push_mesh_snapshot(parts, "edge-flip")
            self.btn_undo.setEnabled(True); self._update_undo_tooltip()

        def do_work():
            from core.mesh_quality import optimise_edge_flips
            return optimise_edge_flips(mesh, max_iterations=3, flatness_deg=10.0)

        def on_done(result):
            new_mesh, msg = result
            sel = self.part_tree.selected_part
            if sel and sel.is_leaf:
                sel.mesh = new_mesh
            else:
                self.mesh_handler.mesh = new_mesh
                self.part_tree.load_mesh(
                    new_mesh,
                    self.part_tree.root.label if self.part_tree.root else "Body")
            self._refresh_tree()
            verts, faces = self.mesh_handler.get_vertex_array()
            self.viewport.set_mesh(verts, faces, self.mesh_handler.get_bounds())
            self.status_bar.showMessage(f"{target_name}: {msg}")

        self._run_threaded(do_work, (),
                           f"Optimising edges on {target_name}…", on_done)

    def _run_decimate_pro(self):
        """Feature-preserving decimation via pyvista.decimate_pro."""
        mesh, target_name = self._get_target_mesh()
        if mesh is None: return
        parts, _ = self._get_target_parts()
        if parts:
            self.part_tree.push_mesh_snapshot(parts, "decimate-pro")
            self.btn_undo.setEnabled(True); self._update_undo_tooltip()
        target_ratio = self.spin_decimate.value()  # reuse same spinner

        def do_work():
            from core.mesh_quality import decimate_pro
            return decimate_pro(mesh,
                                 target_ratio=target_ratio,
                                 feature_angle=30.0,
                                 preserve_topology=True,
                                 preserve_boundary=True)

        def on_done(result):
            new_mesh, msg = result
            sel = self.part_tree.selected_part
            if sel and sel.is_leaf:
                sel.mesh = new_mesh
            else:
                self.mesh_handler.mesh = new_mesh
                self.part_tree.load_mesh(
                    new_mesh,
                    self.part_tree.root.label if self.part_tree.root else "Body")
            self._refresh_tree()
            verts, faces = self.mesh_handler.get_vertex_array()
            self.viewport.set_mesh(verts, faces, self.mesh_handler.get_bounds())
            self.status_bar.showMessage(f"{target_name}: {msg}")

        self._run_threaded(do_work, (),
                           f"Feature-preserving decimation on {target_name}…",
                           on_done)

    def _run_individual_repair(self, func_name):
        """Run a single repair step in a thread with its own undo snapshot."""
        mesh, target_name = self._get_target_mesh()
        if mesh is None: return

        parts, _ = self._get_target_parts()
        if parts:
            self.part_tree.push_mesh_snapshot(parts, func_name.replace('_', ' '))
            self.btn_undo.setEnabled(True); self._update_undo_tooltip()

        def do_repair():
            import core.auto_repair as ar
            func = getattr(ar, func_name)
            return func(mesh)

        def on_done(result):
            new_mesh, description = result
            sel = self.part_tree.selected_part
            if sel and sel.is_leaf:
                sel.mesh = new_mesh
                self._refresh_viewport()
            else:
                self.mesh_handler.mesh = new_mesh
                self.mesh_handler.original_mesh = new_mesh.copy()
                self.part_tree.load_mesh(new_mesh,
                    self.part_tree.root.label if self.part_tree.root else "Body")
                self._refresh_tree()
                verts, faces = self.mesh_handler.get_vertex_array()
                self.viewport.set_mesh(verts, faces, self.mesh_handler.get_bounds())
            self.lbl_repair.setText(description)
            self.status_bar.showMessage(description)
            self._refresh_history()

        self._run_threaded(do_repair, (),
            f"Running {func_name.replace('_', ' ')} on {target_name}…", on_done)

    # ═══════════════════════════════════════════════════════
    # REGION SELECTION + REPAIR
    # ═══════════════════════════════════════════════════════

    def _toggle_selection_mode(self, active):
        self.viewport.selection_mode = active
        if active:
            self.viewport.setCursor(Qt.CrossCursor)
            self.lbl_status.setText("🖌 SELECT MODE — click to paint, Shift=deselect, Ctrl=flood-fill")
            self.lbl_status.setStyleSheet("color:#f0a040;font-weight:700;")
        else:
            self.viewport.setCursor(Qt.ArrowCursor)
            self.lbl_status.setStyleSheet("")
            self.lbl_status.setText(self._get_status_text())
        self._update_mode_indicator()

    def _clear_selection(self):
        self.viewport.selected_faces = set()
        self.viewport.update()
        self.lbl_selection.setText("0 faces selected")

    def _on_faces_selected(self, faces):
        n = len(faces) if faces else 0
        self.lbl_selection.setText(f"{n} faces selected")

    def _auto_select_problems(self):
        if self.mesh_handler.mesh is None: return
        from core.region_repair import select_problem_faces
        problems = select_problem_faces(self.mesh_handler.mesh)
        self.viewport.selected_faces = problems
        self.viewport.update()
        self.lbl_selection.setText(f"{len(problems)} problem faces selected")
        self.status_bar.showMessage(f"Auto-selected {len(problems)} problem faces")

    def _region_fix_normals(self):
        sel = self.viewport.selected_faces
        if not sel:
            self.status_bar.showMessage("No faces selected."); return
        if self.mesh_handler.mesh is None: return
        if self.part_tree.root:
            self.part_tree.push_mesh_snapshot([self.part_tree.root], "fix normals (region)")
            self.btn_undo.setEnabled(True); self._update_undo_tooltip()
        from core.region_repair import repair_selected_normals
        mesh, desc = repair_selected_normals(self.mesh_handler.mesh, sel)
        self.mesh_handler.mesh = mesh
        self._reload_mesh_after_repair(desc)

    def _region_smooth(self):
        sel = self.viewport.selected_faces
        if not sel:
            self.status_bar.showMessage("No faces selected."); return
        if self.mesh_handler.mesh is None: return
        if self.part_tree.root:
            self.part_tree.push_mesh_snapshot([self.part_tree.root], "smooth (region)")
            self.btn_undo.setEnabled(True); self._update_undo_tooltip()
        from core.region_repair import smooth_selected_region
        mesh, desc = smooth_selected_region(self.mesh_handler.mesh, sel, iterations=3, strength=0.3)
        self.mesh_handler.mesh = mesh
        self._reload_mesh_after_repair(desc)

    def _region_delete(self):
        sel = self.viewport.selected_faces
        if not sel:
            self.status_bar.showMessage("No faces selected."); return
        if self.mesh_handler.mesh is None: return
        if self.part_tree.root:
            self.part_tree.push_mesh_snapshot([self.part_tree.root], "delete faces (region)")
            self.btn_undo.setEnabled(True); self._update_undo_tooltip()
        from core.region_repair import delete_selected_faces
        mesh, desc = delete_selected_faces(self.mesh_handler.mesh, sel)
        self.mesh_handler.mesh = mesh
        self.viewport.selected_faces = set()  # clear selection after delete
        self._reload_mesh_after_repair(desc)

    def _reload_mesh_after_repair(self, desc):
        """Refresh everything after a region repair operation."""
        mesh = self.mesh_handler.mesh
        self.mesh_handler.original_mesh = mesh.copy()
        self.part_tree.load_mesh(mesh, self.part_tree.root.label if self.part_tree.root else "Body")
        self._refresh_tree()
        verts, faces = self.mesh_handler.get_vertex_array()
        self.viewport.set_mesh(verts, faces, self.mesh_handler.get_bounds())
        self.lbl_repair.setText(desc)
        self.status_bar.showMessage(desc)
        self._refresh_history()

    def _run_diagnose(self):
        """Quick diagnosis without fixing anything."""
        if self.mesh_handler.mesh is None: return
        from core.auto_repair import quick_diagnose
        result = quick_diagnose(self.mesh_handler.mesh)
        issues = result.get('issues', [])
        if issues:
            msg = f"Found {len(issues)} issues:\n" + "\n".join(f"  • {i}" for i in issues)
        else:
            msg = "✓ No issues found — mesh is clean."
        msg += (f"\n\n{result['triangles']:,} tris  |  {result['vertices']:,} verts  |  "
                f"{result['body_count']} body/ies  |  "
                f"{'Watertight ✓' if result['is_watertight'] else 'Not watertight ⚠'}")
        QMessageBox.information(self, "Mesh Diagnosis", msg)

    def _run_decimate(self):
        """Reduce polygon count."""
        if self.mesh_handler.mesh is None: return
        if self.part_tree.root:
            self.part_tree.push_mesh_snapshot([self.part_tree.root], "decimate")
            self.btn_undo.setEnabled(True); self._update_undo_tooltip()
        self._busy(True, "Reducing polygons…")
        try:
            from core.auto_repair import simplify_mesh
            ratio = self.spin_decimate.value()
            mesh, stats = simplify_mesh(self.mesh_handler.mesh, target_ratio=ratio)
            self.mesh_handler.mesh = mesh
            self._busy(False)
            self.part_tree.load_mesh(mesh, self.part_tree.root.label if self.part_tree.root else "Body")
            self._refresh_tree()
            verts, faces = self.mesh_handler.get_vertex_array()
            self.viewport.set_mesh(verts, faces, self.mesh_handler.get_bounds())
            self.lbl_repair.setText(
                f"Simplified: {stats['before']:,} → {stats['after']:,} tris "
                f"({stats['reduction_pct']}% reduction, {stats['method']})")
            self.status_bar.showMessage(f"Decimated: {stats['reduction_pct']}% reduction")
        except Exception as e:
            self._busy(False)
            QMessageBox.warning(self, "Decimation Error", str(e))

    def _run_remove_shells(self):
        """Remove small disconnected shells."""
        if self.mesh_handler.mesh is None: return
        if self.part_tree.root:
            self.part_tree.push_mesh_snapshot([self.part_tree.root], "remove shells")
            self.btn_undo.setEnabled(True); self._update_undo_tooltip()
        try:
            mesh = self.mesh_handler.mesh
            components = mesh.split(only_watertight=False)
            if len(components) <= 1:
                self.status_bar.showMessage("Only 1 shell — nothing to remove.")
                return
            components.sort(key=lambda c: len(c.faces), reverse=True)
            total = sum(len(c.faces) for c in components)
            # Keep components with >1% of total faces
            keep = [c for c in components if len(c.faces) >= max(10, total * 0.01)]
            removed = len(components) - len(keep)
            if removed > 0 and keep:
                import trimesh
                mesh = trimesh.util.concatenate(keep)
                self.mesh_handler.mesh = mesh
                self.part_tree.load_mesh(mesh, self.part_tree.root.label if self.part_tree.root else "Body")
                self._refresh_tree()
                verts, faces = self.mesh_handler.get_vertex_array()
                self.viewport.set_mesh(verts, faces, self.mesh_handler.get_bounds())
                self.status_bar.showMessage(
                    f"Removed {removed} shells ({len(components)} → {len(keep)} bodies)")
            else:
                self.status_bar.showMessage("No shells small enough to remove.")
        except Exception as e:
            QMessageBox.warning(self, "Shell Removal Error", str(e))

    def _update_health_display(self, report=None):
        """Update the mesh health dashboard."""
        if report is None:
            self.lbl_repair.setText("")
            self.lbl_health_score.setText("")
            return
        # Health bar colour
        score = report.health_score
        if score >= 80:
            col = "#50c870"
            icon = "✓"
        elif score >= 50:
            col = "#f0a020"
            icon = "~"
        else:
            col = "#e05050"
            icon = "⚠"
        self.lbl_health_score.setText(
            f"<span style='color:{col};font-weight:700;font-size:13px;'>"
            f"{icon} Health: {score}/100</span>")
        self.lbl_health_score.setTextFormat(Qt.RichText)
        # Summary text
        self.lbl_repair.setText(report.summary()[:300])

    def _repair_done(self, mesh, report):
        """Legacy repair callback — still used by RepairWorker."""
        self.mesh_handler.mesh = mesh; self.mesh_handler.original_mesh = mesh.copy()
        self._busy(False); self.btn_repair.setEnabled(True)
        self.lbl_repair.setText(report.summary()[:120])
        self.part_tree.load_mesh(mesh, self.part_tree.root.label if self.part_tree.root else "Body")
        self._refresh_tree()
        verts, faces = self.mesh_handler.get_vertex_array()
        self.viewport.set_mesh(verts, faces, self.mesh_handler.get_bounds())
        self.status_bar.showMessage(f"Repair done — {'watertight ✓' if report.is_watertight_after else 'still has issues ⚠'}")

    def _run_remesh(self, mode):
        """Run isotropic or adaptive remeshing on the selected part or whole model."""
        mesh, target_name = self._get_target_mesh()
        if mesh is None:
            QMessageBox.warning(self, "No Model", "Import a model first.")
            return

        mode_name = "Isotropic" if mode == "isotropic" else "Adaptive"
        before_count = len(mesh.faces)

        reply = QMessageBox.question(self, f"{mode_name} Remesh",
            f"Run {mode_name} remesh on {target_name}?\n\n"
            f"Current: {before_count:,} triangles\n\n"
            f"{'Makes all triangles even-sized' if mode == 'isotropic' else 'More detail on curves, less on flats'}\n\n"
            f"This can be undone with Ctrl+Z.",
            QMessageBox.Yes | QMessageBox.No)
        if reply != QMessageBox.Yes:
            return

        # Save undo
        parts, _ = self._get_target_parts()
        if parts:
            self.part_tree.push_mesh_snapshot(parts, f"{mode_name} remesh {target_name}")
            self.btn_undo.setEnabled(True); self._update_undo_tooltip()

        def do_remesh():
            from core.remesh import isotropic_remesh, adaptive_remesh
            if mode == 'isotropic':
                return isotropic_remesh(mesh)
            else:
                return adaptive_remesh(mesh)

        def on_done(result):
            new_mesh, stats = result
            # Apply to the correct target
            sel = self.part_tree.selected_part
            if sel and sel.is_leaf:
                sel.mesh = new_mesh
                self._refresh_viewport()
            else:
                self.mesh_handler.mesh = new_mesh
                self.mesh_handler.original_mesh = new_mesh.copy()
                self.part_tree.load_mesh(new_mesh,
                    self.part_tree.root.label if self.part_tree.root else "Body")
                self._refresh_tree()
                verts, faces = self.mesh_handler.get_vertex_array()
                self.viewport.set_mesh(verts, faces, self.mesh_handler.get_bounds())

            result_msg = stats.get('description', f"{before_count:,} → {len(new_mesh.faces):,} triangles")
            self.lbl_repair.setText(f"✓ {result_msg}")
            self.status_bar.showMessage(result_msg)
            self._refresh_history()
            QMessageBox.information(self, f"{mode_name} Remesh Complete",
                f"{result_msg}\n\nUndo with Ctrl+Z if needed.")

        self._run_threaded(do_remesh, (), f"Running {mode_name} remesh on {target_name}…", on_done)

    def _show_workflow_guide(self):
        """Show the step-by-step workflow guide in a proper resizable,
        scrollable dialog instead of a cramped QMessageBox."""
        from core.workflow import MAIN_WORKFLOW
        from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QScrollArea,
                                     QLabel, QPushButton, QHBoxLayout)

        dlg = QDialog(self)
        dlg.setWindowTitle("Workflow Guide")
        dlg.setMinimumSize(760, 620)
        dlg.resize(820, 680)

        root = QVBoxLayout(dlg)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Header
        hdr = QLabel("  3D Print Slicer — Workflow Guide")
        hdr.setStyleSheet(
            "background:#0f1420;color:#6ac0ff;"
            "font-size:14px;font-weight:700;letter-spacing:1px;"
            "padding:14px 18px;border-bottom:1px solid #1e2838;")
        root.addWidget(hdr)

        # Scrollable content area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            "QScrollArea{background:#141a24;border:none;}"
            "QScrollBar:vertical{background:#141a24;width:12px;}"
            "QScrollBar::handle:vertical{background:#2a3850;border-radius:4px;min-height:30px;}"
            "QScrollBar::handle:vertical:hover{background:#3a4a6a;}")
        inner = QWidget()
        inner_lay = QVBoxLayout(inner)
        inner_lay.setContentsMargins(22, 18, 22, 18)
        inner_lay.setSpacing(14)

        for i, step in enumerate(MAIN_WORKFLOW, start=1):
            card = QFrame()
            card.setStyleSheet(
                "background:#1a2030;border:1px solid #243048;"
                "border-radius:6px;padding:12px 14px;")
            cl = QVBoxLayout(card); cl.setContentsMargins(0, 0, 0, 0); cl.setSpacing(6)
            title = QLabel(f"<span style='color:#ffa040;font-weight:700;'>"
                           f"STEP {i}</span>  <span style='color:#d8e4f5;"
                           f"font-size:13px;font-weight:700;'>{step.name}</span>")
            title.setTextFormat(Qt.RichText)
            cl.addWidget(title)
            body = QLabel(step.description)
            body.setWordWrap(True)
            body.setStyleSheet("color:#a8b8d0;font-size:11px;line-height:1.5;")
            cl.addWidget(body)
            inner_lay.addWidget(card)
        inner_lay.addStretch()
        scroll.setWidget(inner)
        root.addWidget(scroll, 1)

        # Footer with a close button
        foot = QHBoxLayout(); foot.setContentsMargins(14, 10, 14, 12); foot.addStretch()
        btn_close = QPushButton("Close")
        btn_close.setFixedSize(90, 28)
        btn_close.setStyleSheet(
            "QPushButton{background:#263452;color:#d0e0f5;"
            "border:1px solid #3a4a6a;border-radius:4px;font-weight:600;}"
            "QPushButton:hover{background:#304068;}")
        btn_close.clicked.connect(dlg.accept)
        foot.addWidget(btn_close)
        foot_w = QWidget(); foot_w.setLayout(foot)
        foot_w.setStyleSheet("background:#0f1420;border-top:1px solid #1e2838;")
        root.addWidget(foot_w)

        dlg.exec_()

    def _show_feature_info(self, feature_key):
        """Show info dialog for a specific feature."""
        from core.workflow import FEATURE_INFO
        info = FEATURE_INFO.get(feature_key, {})
        title = info.get('title', feature_key)
        text = info.get('text', 'No information available.')
        QMessageBox.information(self, title, text)

    def _show_remesh_info(self):
        """Show info dialog explaining remeshing options."""
        from core.remesh import REMESH_INFO
        msg = ""
        for key in ['isotropic', 'adaptive', 'subdivide']:
            info = REMESH_INFO[key]
            msg += f"━━━ {info['name']} ━━━\n{info['description']}\n\n"
        QMessageBox.information(self, "Mesh Quality Options", msg)

    def _run_subdivide(self):
        """Subdivide selected part or whole mesh to increase polygon count."""
        mesh, target_name = self._get_target_mesh()
        if mesh is None: return
        iters = self.spin_subdivide.value()
        old_count = len(mesh.faces)

        parts, _ = self._get_target_parts()
        if parts:
            self.part_tree.push_mesh_snapshot(parts, f"subdivide {target_name}")
            self.btn_undo.setEnabled(True); self._update_undo_tooltip()

        def do_subdivide():
            from trimesh.remesh import subdivide
            v, f = mesh.vertices, mesh.faces
            for _ in range(iters):
                v, f = subdivide(v, f)
            import trimesh
            return trimesh.Trimesh(vertices=v, faces=f)

        def on_done(new_mesh):
            sel = self.part_tree.selected_part
            if sel and sel.is_leaf:
                sel.mesh = new_mesh
                self._refresh_viewport()
            else:
                self.mesh_handler.mesh = new_mesh
                self.mesh_handler.original_mesh = new_mesh.copy()
                self.part_tree.load_mesh(new_mesh,
                    self.part_tree.root.label if self.part_tree.root else "Body")
                self._refresh_tree()
                verts, faces = self.mesh_handler.get_vertex_array()
                self.viewport.set_mesh(verts, faces, self.mesh_handler.get_bounds())
            new_count = len(new_mesh.faces)
            msg = f"Subdivided {target_name}: {old_count:,} → {new_count:,} triangles"
            self.lbl_repair.setText(msg)
            self.status_bar.showMessage(msg)

        self._run_threaded(do_subdivide, (),
            f"Subdividing {target_name} ({iters} passes, {old_count:,} tris)…", on_done)

    def _run_smoothing(self):
        """Smooth parts — selected part or all parts depending on scope."""
        method_map = {0: 'taubin', 1: 'laplacian', 2: 'humphrey'}
        method     = method_map.get(self.combo_smooth_method.currentIndex(), 'taubin')
        iterations = self.spin_smooth_iters.value()
        strength   = self.spin_smooth_strength.value()
        preserve   = self.chk_preserve_cuts.isChecked()
        smooth_all = self.radio_smooth_all.isChecked()

        if smooth_all:
            parts = self.part_tree.get_all_leaves()
            if not parts:
                QMessageBox.warning(self, "No Parts",
                    "No parts to smooth.\nSlice the model first, or import a model."); return
        else:
            sel = self.part_tree.selected_part
            if sel is None:
                QMessageBox.warning(self, "No Part Selected",
                    "Select a part in the Parts tree first.\n"
                    "Then make sure 'Selected part only' is ticked."); return
            if not sel.is_leaf:
                QMessageBox.warning(self, "Not a Leaf Part",
                    f"'{sel.label}' has children — select one of its sub-parts instead."); return
            parts = [sel]

        n = len(parts)
        method_map_str = {0:'Taubin', 1:'Laplacian', 2:'Humphrey'}
        method_name = method_map_str.get(self.combo_smooth_method.currentIndex(), method)
        scope_str = "all parts" if smooth_all else f"'{parts[0].label}'"

        # Save undo snapshot BEFORE smoothing
        self.part_tree.push_mesh_snapshot(parts, f"smooth {scope_str}")
        self.btn_undo.setEnabled(True)
        self._update_undo_tooltip()

        self._busy(True,
            f"Smoothing {scope_str} — {method_name}, {iterations} passes, strength {strength:.2f}…")
        self.btn_smooth.setEnabled(False)

        self._sm_w = SmoothWorker(parts, method, iterations, strength, preserve)
        self._sm_w.progress.connect(lambda d,t: self._update_progress(d, t, "Smoothing"))
        self._sm_w.finished.connect(self._smooth_done)
        self._sm_w.error.connect(
            lambda e: (self._busy(False),
                       self.btn_smooth.setEnabled(True),
                       QMessageBox.critical(self, "Smooth Error", e)))
        self._sm_w.start()

    def _smooth_done(self, results):
        """Apply smoothed meshes back to the part tree."""
        self._busy(False); self.btn_smooth.setEnabled(True)

        id_map = {p.id: p for p in self.part_tree.get_all_leaves()}
        total_locked = total_smoothed = 0
        vol_changes = []

        for part_id, smoothed_mesh, stats in results:
            if part_id in id_map:
                id_map[part_id].mesh = smoothed_mesh
                total_locked   += stats.get('locked_verts', 0)
                total_smoothed += stats.get('smoothed_verts', 0)
                vc = stats.get('volume_change_pct', 0)
                if vc != 0: vol_changes.append(vc)

        avg_vol = sum(vol_changes)/len(vol_changes) if vol_changes else 0.0
        method  = results[0][2].get('method', '') if results else ''
        iters   = results[0][2].get('iterations', 0) if results else 0

        if total_smoothed == 0 and total_locked > 0:
            note = "\n⚠ All vertices were on cut faces — nothing moved.\nTry a model with curved surfaces."
        elif abs(avg_vol) > 5:
            note = f"\n⚠ Large volume change ({avg_vol:+.1f}%) — reduce Strength or Passes."
        else:
            note = ""

        msg = (f"✓ {len(results)} part{'s' if len(results)>1 else ''} smoothed"
               f" ({method}, {iters} passes)\n"
               f"  {total_smoothed:,} verts smoothed, {total_locked:,} locked (cut faces)\n"
               f"  Avg volume change: {avg_vol:+.1f}%{note}")

        self.lbl_smooth_result.setText(msg)
        self.status_bar.showMessage(
            f"Smoothing done — {len(results)} part{'s' if len(results)>1 else ''}, "
            f"vol change {avg_vol:+.1f}%")
        self._refresh_tree()
        self._refresh_viewport()

    # ═══════════════════════════════════════════════════════
    # AUTO SLICE
    # ═══════════════════════════════════════════════════════

    # ═══════════════════════════════════════════════════════
    # AUTO-SLICE PREVIEW
    # ═══════════════════════════════════════════════════════

    def _preview_cuts(self):
        """Generate cut planes and show them in viewport WITHOUT slicing."""
        if self.part_tree.root is None:
            QMessageBox.warning(self, "No Model", "Import a model first."); return
        from core.slicer import Slicer
        from core.cut_definition import CutDefinition
        mesh = self.part_tree.root.mesh
        bounds = mesh.bounds
        mins, maxs = bounds[0], bounds[1]
        cut_size = self.spin_auto_size.value()

        planes = []
        for ax_idx, axis in enumerate(['x','y','z']):
            lo, hi = float(mins[ax_idx]), float(maxs[ax_idx])
            span = hi - lo
            n_cuts = int(np.floor(span / cut_size))
            for i in range(1, n_cuts+1):
                pos_val = lo + i * cut_size
                if pos_val >= hi: break
                cut = CutDefinition('full')
                cut.axis = axis
                pt = np.zeros(3); pt[ax_idx] = pos_val
                cut.position = pt.copy()
                cut.pinned = False
                planes.append(cut)

        self._preview_planes = planes
        self._active_preview_idx = 0 if planes else -1
        self.btn_auto_slice.setEnabled(len(planes) > 0)
        self._refresh_preview_list()
        bounds = self.mesh_handler.get_bounds()
        self.viewport.mesh_bounds = bounds
        self.viewport.set_preview_cuts(planes, self._active_preview_idx)
        self.status_bar.showMessage(
            f"Preview: {len(planes)} cut planes — drag in viewport or use nudge buttons. Hit ⚡ Apply to commit.")

    def _refresh_preview_list(self):
        planes = self._preview_planes
        self.preview_cut_list.blockSignals(True)
        self.preview_cut_list.setRowCount(len(planes))
        axis_colors = {'x':'#e05555','y':'#50c870','z':'#5588e8'}
        for i, p in enumerate(planes):
            n = QTableWidgetItem(str(i+1))
            n.setFlags(n.flags() & ~Qt.ItemIsEditable)
            n.setForeground(QColor("#3a4a6a"))
            self.preview_cut_list.setItem(i, 0, n)
            a = QTableWidgetItem(p.axis.upper())
            a.setFlags(a.flags() & ~Qt.ItemIsEditable)
            a.setForeground(QColor(axis_colors.get(p.axis,'#8892aa')))
            self.preview_cut_list.setItem(i, 1, a)
            pos_val = p.effective_position
            pos = QTableWidgetItem(f"{pos_val:.1f}")
            pos.setFlags(pos.flags() & ~Qt.ItemIsEditable)
            locked = getattr(p,'pinned',False)
            pos.setForeground(QColor("#f0c040" if locked else "#8ab0d8"))
            self.preview_cut_list.setItem(i, 2, pos)
            if i == self._active_preview_idx:
                for col in range(3):
                    item = self.preview_cut_list.item(i, col)
                    if item: item.setBackground(QColor("#1a2d50"))
        self.preview_cut_list.blockSignals(False)

    def _preview_cut_selected(self, row):
        if row < 0 or row >= len(self._preview_planes): return
        self._active_preview_idx = row
        self._refresh_preview_list()
        self.viewport.set_preview_cuts(self._preview_planes, row)

    def _nudge_preview(self, delta):
        idx = self._active_preview_idx
        if idx < 0 or idx >= len(self._preview_planes): return
        p = self._preview_planes[idx]
        if getattr(p, 'pinned', False):
            self.status_bar.showMessage("Cut is locked — unlock it first.")
            return
        ax_idx = {'x':0,'y':1,'z':2}[p.axis]
        p.position[ax_idx] += delta
        self._refresh_preview_list()
        self.viewport.set_preview_cuts(self._preview_planes, idx)

    def _apply_master_offset(self):
        offset = self.spin_master_offset.value()
        if offset == 0: return
        for p in self._preview_planes:
            if not getattr(p,'pinned',False):
                ax_idx = {'x':0,'y':1,'z':2}[p.axis]
                p.position[ax_idx] += offset
        self.spin_master_offset.setValue(0)
        self._refresh_preview_list()
        self.viewport.set_preview_cuts(self._preview_planes, self._active_preview_idx)

    def _lock_preview_cut(self):
        idx = self._active_preview_idx
        if idx < 0 or idx >= len(self._preview_planes): return
        self._preview_planes[idx].pinned = True
        self._refresh_preview_list()
        self.status_bar.showMessage(f"Cut {idx+1} locked.")

    def _unlock_preview_cut(self):
        idx = self._active_preview_idx
        if idx < 0 or idx >= len(self._preview_planes): return
        self._preview_planes[idx].pinned = False
        self._refresh_preview_list()

    def _unlock_all_preview(self):
        for p in self._preview_planes: p.pinned = False
        self._refresh_preview_list()

    def _on_gizmo_moved(self, new_pos):
        """Viewport gizmo dragged — update the active preview cut position."""
        idx = self._active_preview_idx
        if idx < 0 or idx >= len(self._preview_planes):
            # No preview planes — update the sidebar cut preview instead
            self.spin_cut_pos.setValue(new_pos)
            return
        p = self._preview_planes[idx]
        if getattr(p, 'pinned', False):
            self.status_bar.showMessage("Cut is locked — unlock it first.")
            return
        ax_idx = {'x':0,'y':1,'z':2}[p.axis]
        p.position[ax_idx] = new_pos
        self._refresh_preview_list()
        self.viewport.set_preview_cuts(self._preview_planes, idx)

    def _on_gizmo_rotated(self, ru, rv, rn):
        """Viewport ring dragged — update active cut rotation."""
        idx = self._active_preview_idx
        if idx < 0 or idx >= len(self._preview_planes):
            # No preview planes — update sidebar spinners directly
            self.spin_rot_x.blockSignals(True)
            self.spin_rot_y.blockSignals(True)
            self.spin_rot_x.setValue(ru)
            self.spin_rot_y.setValue(rv)
            self.spin_rot_x.blockSignals(False)
            self.spin_rot_y.blockSignals(False)
            self._update_cut_preview()
            return
        p = self._preview_planes[idx]
        if getattr(p, 'pinned', False):
            self.status_bar.showMessage("Cut is locked — unlock it first.")
            return
        if p.mode == 'full':
            p.mode = 'free'
        p.rot_u = ru
        p.rot_v = rv
        p.rot_n = rn

        # Sync sidebar spinners so user can see the angle
        self.spin_rot_x.blockSignals(True)
        self.spin_rot_y.blockSignals(True)
        self.spin_rot_x.setValue(ru)
        self.spin_rot_y.setValue(rv)
        self.spin_rot_x.blockSignals(False)
        self.spin_rot_y.blockSignals(False)

        # Switch combo to Free mode visually
        if self.combo_cut_mode.currentIndex() == 0:
            self.combo_cut_mode.blockSignals(True)
            self.combo_cut_mode.setCurrentIndex(1)
            self.combo_cut_mode.blockSignals(False)
            self.rot_group.setVisible(True)

        self._refresh_preview_list()
        self.viewport.set_preview_cuts(self._preview_planes, idx)

    def _auto_slice(self):
        """Apply all preview cuts if they exist, otherwise generate fresh ones."""
        if self.part_tree.root is None: return

        # Save undo snapshot before slicing
        self.part_tree.push_mesh_snapshot(
            [self.part_tree.root], "auto-slice")
        self.btn_undo.setEnabled(True)
        self._update_undo_tooltip()

        # If we have preview planes, apply them directly to the part tree
        if self._preview_planes:
            self._busy(True, f"Applying {len(self._preview_planes)} preview cuts…")
            self.btn_auto_slice.setEnabled(False)
            try:
                # Reset to root
                self.part_tree.root.children = []
                from core.part_tree import Part
                Part._color_counter = 1

                # Apply each preview plane as a cut, iterating until all leaves fit
                for cut_def in self._preview_planes:
                    # Cut every current leaf that crosses this plane
                    for leaf in list(self.part_tree.root.all_leaves()):
                        bounds = leaf.mesh.bounds
                        ax_idx = {'x':0, 'y':1, 'z':2}[cut_def.axis]
                        lo = float(bounds[0][ax_idx])
                        hi = float(bounds[1][ax_idx])
                        pos = cut_def.effective_position
                        if lo < pos < hi:
                            leaf.split(cut_def)

                # Renumber leaves
                base = self.part_tree.root.label
                for i, leaf in enumerate(self.part_tree.root.all_leaves()):
                    leaf.label = f"{base}-{i+1:03d}"

                self._preview_planes = []
                self._active_preview_idx = -1
                self.viewport.set_preview_cuts([], -1)
                self._auto_slice_done()
            except Exception as e:
                self._busy(False)
                QMessageBox.critical(self, "Slice Error", str(e))
            return

        # No preview planes — use parallel auto-slicer from scratch
        self._busy(True, "Slicing… (parallel, 4 threads)")
        self.btn_auto_slice.setEnabled(False)
        self._aw = AutoSliceWorker(self.part_tree, self.spin_auto_size.value())
        self._aw.finished.connect(self._auto_slice_done)
        self._aw.progress.connect(lambda done, total: self._update_progress(done, total, "Slicing"))
        self._aw.error.connect(lambda e: (self._busy(False), QMessageBox.critical(self,"Slice Error",e)))
        self._aw.start()

    def _update_progress(self, done, total, label=""):
        if total > 0:
            pct = int(done * 100 / total)
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(pct)
            self.status_bar.showMessage(f"{label}: {done}/{total} ({pct}%)")
            self.setWindowTitle(f"3D Print Slicer — {label} {pct}%")

    def _auto_slice_done(self):
        self._busy(False); self.btn_auto_slice.setEnabled(True)
        self._refresh_tree()
        self._refresh_viewport()
        self._refresh_parts_tables()
        self.btn_export.setEnabled(True)
        self.btn_bambu_export.setEnabled(True)
        self.btn_pdf_export.setEnabled(True)
        self.btn_dryfit.setEnabled(True)
        self.btn_preview_labels.setEnabled(True)
        self.btn_bond_analysis.setEnabled(True)
        self.btn_orient_all.setEnabled(True)
        leaves = self.part_tree.get_all_leaves()

        # Smart spatial renumbering
        from core.grid_numbering import renumber_parts_spatially
        base = self.part_tree.root.label if self.part_tree.root else "Body"
        new_labels = renumber_parts_spatially(leaves, base)
        for part, label in zip(leaves, new_labels):
            part.label = label

        # Auto build volume validation
        profile = self._get_printer_profile()
        oversized = []
        for p in leaves:
            e = p.mesh.extents
            if e[0] > profile.usable_x or e[1] > profile.usable_y or e[2] > profile.usable_z:
                oversized.append(p.label)
        if oversized:
            warn_str = ", ".join(oversized[:5])
            extra = f" +{len(oversized)-5} more" if len(oversized) > 5 else ""
            self.status_bar.showMessage(
                f"Sliced into {len(leaves)} parts. ⚠ {len(oversized)} exceed build volume: {warn_str}{extra}")
        else:
            self.status_bar.showMessage(f"Sliced into {len(leaves)} parts. ✓ All fit build volume.")
        self.lbl_status.setText(f"{len(leaves)} parts")

        # Advance workflow to "connect" stage
        self._set_workflow_stage('connect')

        # Auto-expand export section
        if hasattr(self, '_sections'):
            self._sections.get('export', None) and self._sections['export'].expand()
            self._sections.get('joints', None) and self._sections['joints'].expand()

    # ═══════════════════════════════════════════════════════
    # CUT CONTROLS
    # ═══════════════════════════════════════════════════════

    def _cut_mode_changed(self, idx):
        mode_map = {0:'full', 1:'free', 2:'section', 3:'groove', 4:'natural'}
        mode = mode_map.get(idx, 'full')
        self.rot_group.setVisible(mode in ('free', 'section'))
        self.sec_group.setVisible(mode == 'section')
        self._update_cut_preview()

    def _nudge_cut(self, delta):
        self.spin_cut_pos.setValue(self.spin_cut_pos.value() + delta)

    def _build_cut_def(self) -> CutDefinition:
        mode_map = {0:'full', 1:'free', 2:'section', 3:'groove', 4:'natural'}
        mode = mode_map.get(self.combo_cut_mode.currentIndex(), 'full')
        rot_u = self.spin_rot_x.value()
        rot_v = self.spin_rot_y.value()

        # KEY FIX: if any rotation is set, must use 'free' mode
        # In 'full' mode get_normal() ignores rot_u/rot_v entirely
        if mode == 'full' and (abs(rot_u) > 0.01 or abs(rot_v) > 0.01):
            mode = 'free'
            # Also update the combo so the user can see it switched
            self.combo_cut_mode.blockSignals(True)
            self.combo_cut_mode.setCurrentIndex(1)
            self.combo_cut_mode.blockSignals(False)
            self.rot_group.setVisible(True)

        cut = CutDefinition(mode)
        cut.axis = self.combo_cut_axis.currentText().lower()
        ax_idx = {'x':0,'y':1,'z':2}[cut.axis]
        pos = np.zeros(3); pos[ax_idx] = self.spin_cut_pos.value()
        cut.position = pos
        cut.rot_u = rot_u
        cut.rot_v = rot_v
        cut.section_w = self.spin_sec_w.value()
        cut.section_h = self.spin_sec_h.value()
        return cut

    def _update_cut_preview(self):
        if self.part_tree.root is None: return
        try:
            cut = self._build_cut_def()
            self.viewport.set_preview_cut(cut)
            self.viewport.mesh_bounds = self.mesh_handler.get_bounds()
        except Exception:
            pass

    # Mapping between the Quick-Cut mode combo (short labels) and the
    # Advanced-Options mode combo (long labels). Same underlying modes.
    _QC_TO_CUT_MODE = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}  # Full, Angled, Section, Groove, Natural
    _CUT_TO_QC_MODE = {v: k for k, v in _QC_TO_CUT_MODE.items()}

    def _sync_cut_controls(self, source):
        """Mirror axis/position/mode between the Quick Cut bar and the
        Advanced Options sidebar so both always agree with the orange
        preview. `source` identifies which control fired so we can copy in
        the right direction without an infinite signal loop."""
        if getattr(self, '_cut_sync_guard', False):
            return
        self._cut_sync_guard = True
        try:
            if source == 'qc_axis':
                self.combo_cut_axis.setCurrentIndex(self.combo_qc_axis.currentIndex())
            elif source == 'cut_axis':
                self.combo_qc_axis.setCurrentIndex(self.combo_cut_axis.currentIndex())
            elif source == 'qc_pos':
                self.spin_cut_pos.setValue(self.spin_qc_pos.value())
            elif source == 'cut_pos':
                self.spin_qc_pos.setValue(self.spin_cut_pos.value())
            elif source == 'qc_mode':
                self.combo_cut_mode.setCurrentIndex(
                    self._QC_TO_CUT_MODE.get(self.combo_qc_mode.currentIndex(), 0))
            elif source == 'cut_mode':
                self.combo_qc_mode.setCurrentIndex(
                    self._CUT_TO_QC_MODE.get(self.combo_cut_mode.currentIndex(), 0))
        finally:
            self._cut_sync_guard = False
        # After sync, refresh the preview from the unified state.
        self._update_cut_preview()

    def _quick_cut(self):
        """Quick-cut from the toolbar — uses selected part, axis/position from quick bar."""
        selected = self.part_tree.selected_part
        if selected is None or not selected.is_leaf:
            self.status_bar.showMessage("Select a leaf part in the tree first.")
            return
        mode_map = {0:'full', 1:'free', 2:'section', 3:'groove', 4:'natural'}
        mode = mode_map.get(self.combo_qc_mode.currentIndex(), 'full')

        # Natural cut uses a different code path
        if mode == 'natural':
            from core.natural_cut import natural_cut as do_natural_cut
            axis = self.combo_qc_axis.currentText().lower()
            pos_val = self.spin_qc_pos.value()
            side_a, side_b, info = do_natural_cut(selected.mesh, axis, pos_val)
            if side_a is None and side_b is None:
                self.status_bar.showMessage(f"Natural cut failed: {info.get('description', '')}")
                return
            # Manually split the part
            from core.part_tree import Part
            children = []
            for i, m in enumerate([side_a, side_b]):
                if m is not None and len(m.faces) > 0:
                    suffix = 'A' if i == 0 else 'B'
                    child = Part(m, f"{selected.label}-{suffix}", parent=selected)
                    children.append(child)
            if not children:
                self.status_bar.showMessage("Natural cut missed."); return
            selected.children = children
            self.part_tree._undo_stack.append({'action': 'split', 'part': selected})
            self.part_tree.selected_part = children[0]
            count = len(children)
            self.status_bar.showMessage(f"Natural cut: {info.get('description', '')} → {count} pieces")
            child_a = children[0] if children else None
            child_b = children[1] if len(children) > 1 else None
        else:
            # Route through _build_cut_def() so the Quick Cut button produces
            # the SAME cut that the orange preview shows — including groove
            # teeth count/depth/width, rotation, section W×H, etc. Before
            # this, the Quick bar only knew axis+position and always cut
            # with default CutDefinition params, so picking "Groove" in the
            # sidebar followed by hitting Cut on the bar gave a flat cut.
            cut = self._build_cut_def()
            child_a, child_b = self.part_tree.apply_cut(selected, cut)
            # Detect silent fallback (groove → full) so the user knows why.
            if mode == 'groove' and child_a is not None and child_b is not None:
                # If the cut result has perfectly flat contact bounds, groove
                # implementation fell back. Easy heuristic: if the child
                # bounds touch on a single plane with 0 variance, it's flat.
                try:
                    ba = np.asarray(child_a.mesh.bounds)
                    bb = np.asarray(child_b.mesh.bounds)
                    ax_idx = {'x':0, 'y':1, 'z':2}[cut.axis]
                    same = abs(ba[1][ax_idx] - bb[0][ax_idx]) < 0.1 or \
                           abs(ba[0][ax_idx] - bb[1][ax_idx]) < 0.1
                    if same:
                        self.status_bar.showMessage(
                            "Note: groove cut fell back to flat — "
                            "try a larger depth or different axis.")
                except Exception:
                    pass
        if child_a is None and child_b is None:
            self.status_bar.showMessage("Cut missed — adjust position.")
            return
        count = sum(1 for c in [child_a, child_b] if c is not None)
        self.status_bar.showMessage(f"Quick cut: split into {count} pieces")
        # Same post-cut cleanup as _apply_cut — stale heatmap + selection.
        try: self.viewport.clear_heatmap()
        except Exception: pass
        new_leaf = child_a if child_a is not None else child_b
        if new_leaf is not None:
            self.part_tree.select(new_leaf)
        self.btn_undo.setEnabled(self.part_tree.can_undo())
        self._update_undo_tooltip()
        self._refresh_tree()
        self._refresh_viewport()
        self._refresh_parts_tables()
        self.btn_export.setEnabled(True)
        self.btn_qc_cut.setEnabled(True)
        # Advance workflow to 'connect' if we now have multiple parts
        if len(self.part_tree.get_all_leaves()) > 1 and self._current_stage in ('import', 'prepare', 'slice'):
            self._set_workflow_stage('connect')

    def _quick_cut_preview(self):
        """Update the preview from the unified cut state. Since the Quick
        Cut bar is now synced two-way with the Advanced Options sidebar,
        we just delegate — guaranteed to match what the Cut button will do."""
        self._update_cut_preview()

    def _apply_cut(self):
        selected = self.part_tree.selected_part
        if selected is None or not selected.is_leaf:
            self.lbl_cut_result.setText("⚠ Select a leaf part in the tree first.")
            return

        cut = self._build_cut_def()
        child_a, child_b = self.part_tree.apply_cut(selected, cut)

        if child_a is None and child_b is None:
            self.lbl_cut_result.setText("⚠ Cut missed the selected part — adjust position.")
            return

        count = sum(1 for c in [child_a, child_b] if c is not None)
        self.lbl_cut_result.setText(f"✓ Split into {count} pieces — {cut.describe()}")

        # Post-cut cleanup — the heatmap vertex data was computed against
        # the pre-cut mesh and will show as stale red smears on the old
        # geometry location if left on. Same for wireframe overlays that
        # were set on the now-deleted parent actor.
        try: self.viewport.clear_heatmap()
        except Exception: pass
        # Move selection to the first new leaf so the user isn't stranded
        # on a non-leaf parent (which makes subsequent tools no-op).
        new_leaf = child_a if child_a is not None else child_b
        if new_leaf is not None:
            self.part_tree.select(new_leaf)

        self.btn_undo.setEnabled(self.part_tree.can_undo())
        self._update_undo_tooltip()
        self._refresh_tree()
        self._refresh_viewport()
        self._refresh_parts_tables()
        self._refresh_history()
        self.btn_export.setEnabled(True)

    def _undo(self):
        description = self.part_tree.undo()
        if description:
            self.btn_undo.setEnabled(self.part_tree.can_undo())
            self._update_undo_tooltip()
            self._refresh_tree()
            self._refresh_viewport()
            self._refresh_parts_tables()
            self._refresh_history()
            self.lbl_cut_result.setText(f"↩ {description}")
            self.status_bar.showMessage(f"↩ {description}")
            if not self.part_tree.get_all_leaves():
                self.btn_export.setEnabled(False)
            # Re-evaluate workflow stage after undo
            self._recompute_workflow_stage()
        else:
            self.status_bar.showMessage("Nothing to undo.")

    def _recompute_workflow_stage(self):
        """Figure out what stage we should be on based on current state."""
        if self.mesh_handler.mesh is None:
            self._set_workflow_stage('import')
        elif self.part_tree.root and len(self.part_tree.get_all_leaves()) > 1:
            # Has sliced parts — connect stage (or export if user is exporting)
            if self._current_stage not in ('connect', 'export'):
                self._set_workflow_stage('connect')
            else:
                self._update_sidebar_visibility()
        else:
            # Has mesh but no slice yet
            if self._current_stage == 'import':
                self._set_workflow_stage('prepare')
            else:
                self._update_sidebar_visibility()

    def _update_undo_tooltip(self):
        """Keep undo button tooltip current."""
        if self.part_tree.can_undo():
            self.btn_undo.setEnabled(True)
            self.btn_undo.setToolTip(f"↩ {self.part_tree.undo_description()}")
        else:
            self.btn_undo.setEnabled(False)
            self.btn_undo.setToolTip("Nothing to undo")

    # ═══════════════════════════════════════════════════════
    # TREE WIDGET
    # ═══════════════════════════════════════════════════════

    def _refresh_tree(self):
        self.parts_tree_widget.blockSignals(True)
        self.parts_tree_widget.clear()
        self.parts_tree_widget.setColumnCount(3)
        if self.part_tree.root is None:
            self.parts_tree_widget.blockSignals(False); return

        def add_item(part, parent_item):
            dims = part.get_dimensions()
            # Use short display name to avoid truncation
            # Show just the suffix (e.g., "001" from "ModelName-001") + compact dims
            full_label = part.label
            if '-' in full_label:
                short = full_label.rsplit('-', 1)[-1]
            else:
                # Truncate long names
                short = full_label[:20] + "…" if len(full_label) > 20 else full_label
            dims_str = f"{dims[0]:.0f}×{dims[1]:.0f}×{dims[2]:.0f}"
            if part.is_leaf:
                name = f"{short}  {dims_str}"
            else:
                name = f"▸ {short}"

            if parent_item is None:
                item = QTreeWidgetItem(self.parts_tree_widget)
            else:
                item = QTreeWidgetItem(parent_item)
            item.setText(0, name)
            item.setData(0, Qt.UserRole, part.id)

            if part.is_leaf:
                # Column 1: wireframe toggle (◇ = solid, ◆ = wireframe)
                is_wf = getattr(part, '_wireframe', False)
                item.setText(1, "◆" if is_wf else "◇")
                item.setData(1, Qt.UserRole, "wireframe")
                item.setForeground(1, QColor("#60d0ff") if is_wf else QColor("#3a5070"))
                item.setToolTip(1, "Click to toggle wireframe/solid")

                # Column 2: eye icon for visibility
                item.setText(2, "👁" if part.visible else "○")
                item.setData(2, Qt.UserRole, "eye")
                item.setForeground(2, QColor("#80c0ff") if part.visible else QColor("#303848"))
                item.setToolTip(2, "Click to toggle visibility")

                # Label color — bright, high contrast
                c = part.color
                if part.visible:
                    # Brighten the part color for readability
                    r = min(255, int(c[0]*255) + 60)
                    g = min(255, int(c[1]*255) + 60)
                    b = min(255, int(c[2]*255) + 60)
                    col = QColor(r, g, b)
                else:
                    col = QColor("#505868")
                item.setForeground(0, col)
                font = item.font(0); font.setBold(True); item.setFont(0, font)
                item.setToolTip(0, f"{part.label}\n{dims[0]:.1f} × {dims[1]:.1f} × {dims[2]:.1f} mm\n{len(part.mesh.faces):,} triangles\nClick to select")
            else:
                item.setForeground(0, QColor("#8090b0"))

            # Highlight selected — use brighter, more visible background
            is_sel = self.part_tree.selected_part and part.id == self.part_tree.selected_part.id
            if is_sel:
                for col_idx in range(3):
                    item.setBackground(col_idx, QColor("#1e3a6a"))

            for child in part.children:
                add_item(child, item)
            return item

        add_item(self.part_tree.root, None)
        self.parts_tree_widget.expandAll()

        if self.part_tree.selected_part:
            self._select_tree_item_by_id(self.part_tree.selected_part.id)

        self.parts_tree_widget.blockSignals(False)

    def _select_tree_item_by_id(self, part_id):
        def find(item):
            if item.data(0, Qt.UserRole) == part_id:
                self.parts_tree_widget.setCurrentItem(item)
                return True
            for i in range(item.childCount()):
                if find(item.child(i)):
                    return True
            return False
        root = self.parts_tree_widget.invisibleRootItem()
        for i in range(root.childCount()):
            find(root.child(i))

    def _tree_item_clicked(self, item, column):
        """Column 0 = select, Column 1 = wireframe toggle, Column 2 = visibility."""
        part_id = item.data(0, Qt.UserRole)
        part = self.part_tree.find_by_id(part_id)
        if part is None:
            return
        if column == 1 and part.is_leaf:
            # Wireframe toggle
            part._wireframe = not getattr(part, '_wireframe', False)
            self._refresh_tree()
            self._refresh_viewport()
            mode = "wireframe" if part._wireframe else "solid"
            self.status_bar.showMessage(f"{part.label}: {mode}")
        elif column == 2 and part.is_leaf:
            # Visibility toggle
            part.visible = not part.visible
            self._refresh_tree()
            self._refresh_viewport()
            self.status_bar.showMessage(
                f"{'Showing' if part.visible else 'Hidden'}: {part.label}")
        # Column 0 selection is handled by _tree_selection_changed

    def _show_all_parts(self):
        self.part_tree.show_all()
        self._refresh_tree()
        self._refresh_viewport()
        self.status_bar.showMessage("All parts visible.")

    def _hide_all_parts(self):
        for p in self.part_tree.get_all_leaves():
            p.visible = False
        self._refresh_tree()
        self._refresh_viewport()
        self.status_bar.showMessage("All parts hidden.")

    def _solo_selected_part(self):
        sel = self.part_tree.selected_part
        if sel is None:
            self.status_bar.showMessage("Select a part first."); return
        # If selected part is not a leaf, solo all its leaves
        target_ids = {p.id for p in (sel.all_leaves() if not sel.is_leaf else [sel])}
        for p in self.part_tree.get_all_leaves():
            p.visible = (p.id in target_ids)
        self._refresh_tree()
        self._refresh_viewport()
        self.status_bar.showMessage(f"Solo: {sel.label}")

    def _tree_selection_changed(self, current, previous):
        if current is None: return
        part_id = current.data(0, Qt.UserRole)
        part = self.part_tree.find_by_id(part_id)
        if part is None: return
        self.part_tree.select(part)
        self.viewport.set_selected_part(part_id)

        dims = part.get_dimensions()
        state = "leaf — ready to cut or export" if part.is_leaf else f"has {len(part.children)} children"
        # Build cut history for this part
        cut_history = ""
        p = part
        history_lines = []
        while p.parent is not None:
            cut = p.cut_used
            if cut:
                history_lines.append(cut.describe())
            p = p.parent
        if history_lines:
            cut_history = "\nCut history: " + " → ".join(reversed(history_lines))
        # Joint config
        joint_info = ""
        jt = getattr(part, '_joint_type', None)
        if jt and jt != 'flat':
            joint_info = f"\nJoint: {jt}"
        self.lbl_part_info.setText(
            f"{part.label}\n{dims[0]:.1f}×{dims[1]:.1f}×{dims[2]:.1f} mm\n"
            f"{len(part.mesh.faces):,} triangles\n{state}{cut_history}{joint_info}")

        # Update cut position to centre of selected part's bounds
        if part.is_leaf:
            bounds = part.get_bounds()
            centre = (bounds[0]+bounds[1])/2
            ax_idx = {'x':0,'y':1,'z':2}.get(self.combo_cut_axis.currentText().lower(), 0)
            self.spin_cut_pos.blockSignals(True)
            self.spin_cut_pos.setValue(round(float(centre[ax_idx]),1))
            self.spin_cut_pos.blockSignals(False)
            self._update_cut_preview()

        self.btn_apply_cut.setEnabled(part.is_leaf)
        self.lbl_cut_result.setText("" if part.is_leaf else "⚠ Select a leaf part to cut.")
        # Refresh joint preview for newly selected part
        self._update_joint_preview()
        # Update context bar
        self._update_context_bar()

    # ═══════════════════════════════════════════════════════
    # VIEWPORT REFRESH
    # ═══════════════════════════════════════════════════════

    def _refresh_viewport(self):
        leaves = self.part_tree.get_all_leaves()
        all_parts = self.part_tree.get_all_parts()
        bounds = self.mesh_handler.get_bounds()
        self.viewport.set_parts(leaves, all_parts, bounds)
        self.viewport.mesh_bounds = bounds
        if self.part_tree.selected_part:
            self.viewport.set_selected_part(self.part_tree.selected_part.id)
        self._update_cut_preview()

    # ═══════════════════════════════════════════════════════
    # ADJACENT-PARTS VIEW  +  RESET ORIENTATION
    # ═══════════════════════════════════════════════════════

    def _find_adjacent_parts(self, part, tolerance_mm=1.5):
        """Return the parts whose bounding box touches `part`'s bounding box
        within `tolerance_mm`. Uses AABB proximity because any two parts
        that share a cut face will have touching AABBs (they were a single
        mesh split on a plane). Cheap and robust."""
        if part is None or not hasattr(part, 'mesh'):
            return []
        try:
            pb = part.mesh.bounds
        except Exception:
            return []
        p_min = pb[0] - tolerance_mm
        p_max = pb[1] + tolerance_mm

        neighbours = []
        for other in self.part_tree.get_all_leaves():
            if other is part:
                continue
            try:
                ob = other.mesh.bounds
            except Exception:
                continue
            # AABB overlap test in 3D
            if (ob[1][0] >= p_min[0] and ob[0][0] <= p_max[0] and
                ob[1][1] >= p_min[1] and ob[0][1] <= p_max[1] and
                ob[1][2] >= p_min[2] and ob[0][2] <= p_max[2]):
                neighbours.append(other)
        return neighbours

    def _ctx_toggle_adjacent(self):
        """Toggle 'show only selected + adjacent' visibility mode."""
        sel = self.part_tree.selected_part
        if sel is None or not sel.is_leaf:
            self.status_bar.showMessage("Select a single part first.")
            return

        on = not getattr(self, '_adjacent_mode', False)
        self._adjacent_mode = on
        all_leaves = self.part_tree.get_all_leaves()

        if on:
            # Remember what was visible so we can restore it cleanly.
            self._pre_adjacent_visibility = {p.id: p.visible for p in all_leaves}
            neighbours = self._find_adjacent_parts(sel)
            keep_ids = {sel.id} | {n.id for n in neighbours}
            for p in all_leaves:
                p.visible = (p.id in keep_ids)
            if neighbours:
                names = ", ".join(n.label for n in neighbours[:4])
                more = f" (+{len(neighbours)-4} more)" if len(neighbours) > 4 else ""
                self.status_bar.showMessage(
                    f"Showing {sel.label} + {len(neighbours)} adjacent: {names}{more}. "
                    "Use Explode slider to separate them.")
            else:
                self.status_bar.showMessage(f"{sel.label} has no detected neighbours.")
        else:
            # Restore previous visibility
            prev = getattr(self, '_pre_adjacent_visibility', {})
            for p in all_leaves:
                if p.id in prev:
                    p.visible = prev[p.id]
                else:
                    p.visible = True
            self._pre_adjacent_visibility = {}
            self.status_bar.showMessage("Showing all parts again.")

        self._refresh_tree()
        self._refresh_viewport()
        self._update_context_bar()

    def _reset_orientation(self):
        """Undo manual rotations so parts snap back to their assembly
        position. The rotate-object tool records each committed transform
        onto part._user_rotation; this applies the inverse and clears it."""
        sel = self.part_tree.selected_part
        # Scope: selected leaf if any, otherwise every part.
        if sel is not None and sel.is_leaf:
            targets = [sel]
            scope_msg = f"Reset orientation: {sel.label}"
        else:
            targets = self.part_tree.get_all_leaves()
            scope_msg = f"Reset orientation on all {len(targets)} parts"

        reset_count = 0
        for p in targets:
            M = getattr(p, '_user_rotation', None)
            if M is None:
                continue
            try:
                inv = np.linalg.inv(np.asarray(M, dtype=float))
                p.mesh.apply_transform(inv)
                p._user_rotation = None
                reset_count += 1
            except Exception as e:
                print(f"Reset-orient failed on {p.label}: {e}")

        if reset_count == 0:
            self.status_bar.showMessage("No manual rotations to reset.")
            return
        self._refresh_viewport()
        self.status_bar.showMessage(f"{scope_msg} — {reset_count} part(s) returned to assembly position.")

    # ═══════════════════════════════════════════════════════
    # VIEW PRESETS + ROTATE-OBJECT MODE
    # ═══════════════════════════════════════════════════════

    def _set_view_preset(self, key):
        """Snap the camera to a fixed orthographic preset. Preserves the
        current focal point (orbit centre) so the model stays framed."""
        if not hasattr(self, 'viewport'):
            return
        plotter = self.viewport.plotter
        # Compute a focal point: use mesh bounds centre if available.
        bounds = self.viewport.mesh_bounds
        if bounds is not None:
            fx = (bounds[0][0] + bounds[1][0]) * 0.5
            fy = (bounds[0][1] + bounds[1][1]) * 0.5
            fz = (bounds[0][2] + bounds[1][2]) * 0.5
            extent = float(np.max(bounds[1] - bounds[0]))
        else:
            fx, fy, fz = 0.0, 0.0, 0.0
            extent = 200.0
        dist = max(extent * 2.2, 150.0)
        fp = (fx, fy, fz)
        # (position, viewup) for each preset
        presets = {
            'front':  ((fx, fy - dist, fz), (0, 0, 1)),
            'back':   ((fx, fy + dist, fz), (0, 0, 1)),
            'left':   ((fx - dist, fy, fz), (0, 0, 1)),
            'right':  ((fx + dist, fy, fz), (0, 0, 1)),
            'top':    ((fx, fy, fz + dist), (0, 1, 0)),
            'bottom': ((fx, fy, fz - dist), (0, 1, 0)),
            'iso':    ((fx + dist*0.7, fy - dist*0.7, fz + dist*0.6), (0, 0, 1)),
        }
        pos, up = presets.get(key, presets['iso'])
        try:
            plotter.camera.position = pos
            plotter.camera.focal_point = fp
            plotter.camera.up = up
            plotter.camera.parallel_projection = (key != 'iso')
            plotter.render()
        except Exception:
            pass
        # Update toggle state
        for k, btn in self._view_buttons.items():
            btn.setChecked(k == key)
        self.lbl_view_hint.setText(f"Locked: {key.title()} view — hold R and drag to rotate object")

    def _reset_all_interaction_modes(self):
        """The master 'get me back to normal' action, wired to Esc and the
        Normal View button. Idempotent — safe to spam-click."""
        vp = self.viewport
        # Clear every sticky viewport mode + drag state.
        try:
            vp._reset_all_modes()
        except Exception:
            pass

        # Clear heatmap overlays
        try:
            vp.clear_heatmap()
        except Exception:
            pass

        # Uncheck UI toggles without firing their handlers (would recurse)
        for chk in ['chk_wireframe', 'chk_sel_wire',
                    'chk_manual_dowels', 'btn_select_mode',
                    'btn_rotate_obj', 'btn_thin_heatmap']:
            w = getattr(self, chk, None)
            if w is None: continue
            try:
                w.blockSignals(True)
                if hasattr(w, 'setChecked'):
                    w.setChecked(False)
                w.blockSignals(False)
            except Exception:
                pass

        # Reset viewport flags
        vp.show_wireframe = False
        vp.show_selection_wireframe = False
        vp.selection_mode = False
        vp.manual_dowel_mode = False
        vp._measure_active = False

        # Force every part back to solid shading
        for p in self.part_tree.get_all_leaves() if self.part_tree.root else []:
            if getattr(p, '_wireframe', False):
                p._wireframe = False

        # Kill hover state + cursor
        self._hover_highlight_id = None
        self._hover_was_hidden = False
        try: vp.setCursor(Qt.ArrowCursor)
        except Exception: pass

        # Redraw everything fresh
        self._refresh_viewport()
        self._refresh_tree()
        self._update_mode_indicator()
        self.status_bar.showMessage("All interaction modes cleared — solid view restored.")

    def _update_mode_indicator(self):
        """Label in the view bar that reflects the current LMB-drag behaviour
        so the user never has to guess what's active."""
        if not hasattr(self, 'lbl_mode'):
            return
        vp = self.viewport
        mode_label = "Normal"
        active = False
        if getattr(vp, 'manual_dowel_mode', False):
            mode_label = "Dowel Placement"; active = True
        elif getattr(vp, 'selection_mode', False):
            mode_label = "Select Faces"; active = True
        elif getattr(vp, '_measure_active', False):
            mode_label = "Measure"; active = True
        elif getattr(vp, '_rotate_obj_mode', False):
            mode_label = "Rotate Object"; active = True
        self.lbl_mode.setText(f"MODE: {mode_label}")
        self.lbl_mode.setProperty('active', 'true' if active else 'false')
        self.lbl_mode.style().unpolish(self.lbl_mode)
        self.lbl_mode.style().polish(self.lbl_mode)

    def _toggle_rotate_object_mode(self, checked):
        """Switch LMB-drag behaviour: True = rotate the selected part around
        its own centroid. False = normal camera orbit."""
        self.viewport.set_rotate_object_mode(bool(checked))
        if checked:
            self.lbl_view_hint.setText("Rotate-Object mode — LMB drag rotates the selected part")
            self.status_bar.showMessage(
                "Rotate-Object mode: drag on viewport to rotate the selected part. "
                "Press Esc to exit.")
        else:
            self.lbl_view_hint.setText("")
            self.status_bar.showMessage("Camera orbit mode")
        self._update_mode_indicator()

    # ═══════════════════════════════════════════════════════
    # TABLES
    # ═══════════════════════════════════════════════════════

    def _refresh_parts_tables(self):
        leaves = self.part_tree.get_all_leaves()

        self.flat_parts_table.setRowCount(len(leaves))
        for i, part in enumerate(leaves):
            e = part.get_dimensions()
            for j, val in enumerate([part.label, f"{e[0]:.1f}", f"{e[1]:.1f}", f"{e[2]:.1f}", f"{len(part.mesh.faces):,}"]):
                self.flat_parts_table.setItem(i, j, QTableWidgetItem(val))

        # Estimates
        from core.estimator import estimate_all, total_summary, _orientation_hint
        from core.slicer import SlicedPart as SP
        # Wrap parts for estimator compatibility
        class FakePart:
            def __init__(self, p): self.mesh = p.mesh; self.label = p.label; self.grid_index=(0,0,0); self.joint_type='flat'
        fake = [FakePart(p) for p in leaves]
        ests = estimate_all(fake, wall_thickness_mm=self.spin_wall.value(), material=self.combo_material.currentText())
        summ = total_summary(ests)
        self.estimate_table.setRowCount(len(ests))
        for i, e in enumerate(ests):
            vals = [e['label'], e['time_str'],
                    f"{e.get('weight_g', e['filament_g'])}g",
                    f"{e['filament_g']}g",
                    e['orientation_hint']]
            for j, val in enumerate(vals):
                item = QTableWidgetItem(val)
                if j == 4 and val: item.setForeground(QColor("#f0a020"))
                self.estimate_table.setItem(i, j, item)
        # Cost and weight
        cost_per_kg = self.spin_filament_cost.value()
        total_cost = summ['total_filament_g'] / 1000.0 * cost_per_kg
        total_weight = summ.get('total_weight_g', summ['total_filament_g'])
        self.lbl_est_total.setText(
            f"Total: {summ['total_time_str']}  ·  {total_weight}g ({total_weight/1000:.2f}kg)  ·  "
            f"{summ['total_filament_spools']:.2f} spools  ·  ${total_cost:.2f}  ·  "
            f"{summ['part_count']} parts")
        self.btn_printability.setEnabled(len(leaves) > 0)
        self.btn_wall_check.setEnabled(len(leaves) > 0)

        # Enable bond analysis button
        self.btn_bond_analysis.setEnabled(len(leaves) > 0)

    # ═══════════════════════════════════════════════════════
    # HOLLOW / EXPORT
    # ═══════════════════════════════════════════════════════

    def _update_joint_groups(self, idx):
        """Show/hide joint config panels based on selected type."""
        self.round_group.setVisible(idx == 1)
        self.rect_group.setVisible(idx == 2)
        self.dove_group.setVisible(idx == 3)
        self.magnet_group.setVisible(idx == 4)
        self.snap_group.setVisible(idx == 5)
        self.dshape_group.setVisible(idx == 6)
        self.pyramid_group.setVisible(idx == 7)
        self.terrace_group.setVisible(idx == 8)
        self.square_group.setVisible(idx == 9)
        self._update_joint_preview()

    def _update_joint_preview(self):
        """Generate and show joint preview geometry in the viewport."""
        if not hasattr(self, 'viewport'):
            return
        idx = self.combo_joint_type.currentIndex()
        if idx == 0:
            self.viewport.set_joint_preview([]); return

        sel = self.part_tree.selected_part
        if sel is None or not sel.is_leaf:
            self.viewport.set_joint_preview([]); return

        try:
            import trimesh as _tm
            mesh = sel.mesh

            # Find all large flat faces (cut faces) using facets
            # A cut face is a large axis-aligned planar region
            cut_faces = self._find_cut_faces(mesh)

            if not cut_faces:
                self.viewport.set_joint_preview([]); return

            preview_meshes = []

            for normal, origin in cut_faces:
                # Normal points OUTWARD from the face. To show connectors
                # going INTO this part, we offset inward (opposite to normal).
                n = np.array(normal, dtype=float)
                n /= max(np.linalg.norm(n), 1e-9)

                if idx == 1:  # round dowel
                    from core.dowel_joints import DowelConfig, _distribute_dowel_positions
                    cfg = DowelConfig(
                        shape='round', count=self.spin_rod_count.value(),
                        depth_a=self.spin_depth_a.value(),
                        depth_b=self.spin_depth_b.value(),
                        radius=self.spin_rod_radius.value(),
                        tolerance=self.spin_rod_tol.value()
                    )
                    positions = _distribute_dowel_positions(mesh, normal, origin, cfg)
                    depth = cfg.depth_a  # only show THIS part's side
                    for pos in positions:
                        cyl = _tm.creation.cylinder(
                            radius=cfg.radius, height=depth, sections=16)
                        rot = _tm.geometry.align_vectors([0,0,1], (-n).tolist())
                        cyl.apply_transform(rot)
                        # Position: start at face, go inward
                        cyl.apply_translation(pos - n * (depth / 2))
                        preview_meshes.append(cyl)

                elif idx == 2:  # rect slot
                    from core.dowel_joints import DowelConfig, _distribute_dowel_positions
                    cfg = DowelConfig(
                        shape='rect', count=self.spin_rect_count.value(),
                        depth_a=self.spin_rect_depth_a.value(),
                        depth_b=self.spin_rect_depth_b.value(),
                        rect_width=self.spin_rect_w.value(),
                        rect_height=self.spin_rect_h.value(),
                        tolerance=self.spin_rect_tol.value()
                    )
                    positions = _distribute_dowel_positions(mesh, normal, origin, cfg)
                    depth = cfg.depth_a
                    for pos in positions:
                        slot = _tm.creation.box([depth, cfg.rect_width, cfg.rect_height])
                        rot = _tm.geometry.align_vectors([1,0,0], (-n).tolist())
                        slot.apply_transform(rot)
                        slot.apply_translation(pos - n * (depth / 2))
                        preview_meshes.append(slot)

                elif idx == 3:  # dovetail
                    size = self.spin_dove_size.value()
                    depth = size * 2.0
                    box = _tm.creation.box([depth, size*2, size*1.2])
                    rot = _tm.geometry.align_vectors([1,0,0], (-n).tolist())
                    box.apply_transform(rot)
                    box.apply_translation(origin - n * (depth / 2))
                    preview_meshes.append(box)

                elif idx == 4:  # magnet pocket
                    from core.dowel_joints import DowelConfig, _distribute_dowel_positions
                    r = self.spin_magnet_dia.value() / 2.0
                    d = self.spin_magnet_depth.value()
                    cfg = DowelConfig(
                        shape='round', count=self.spin_magnet_count.value(),
                        depth_a=d, depth_b=d, radius=r,
                        tolerance=self.spin_magnet_tol.value())
                    positions = _distribute_dowel_positions(mesh, normal, origin, cfg)
                    for pos in positions:
                        cyl = _tm.creation.cylinder(radius=r, height=d, sections=16)
                        rot = _tm.geometry.align_vectors([0,0,1], (-n).tolist())
                        cyl.apply_transform(rot)
                        cyl.apply_translation(pos - n * (d / 2))
                        preview_meshes.append(cyl)

                elif idx == 5:  # snap-fit
                    from core.dowel_joints import DowelConfig, _distribute_dowel_positions
                    w = self.spin_snap_w.value()
                    h = self.spin_snap_h.value()
                    cfg = DowelConfig(
                        shape='rect', count=self.spin_snap_count.value(),
                        rect_width=w, rect_height=h,
                        tolerance=self.spin_snap_tol.value())
                    positions = _distribute_dowel_positions(mesh, normal, origin, cfg)
                    depth = w * 2
                    for pos in positions:
                        clip = _tm.creation.box([depth, w, h])
                        rot = _tm.geometry.align_vectors([1,0,0], (-n).tolist())
                        clip.apply_transform(rot)
                        clip.apply_translation(pos - n * (depth / 2))
                        preview_meshes.append(clip)

                elif idx in (6, 7, 8, 9):  # D-shape, Pyramid, Terrace, Square
                    from core.connector_shapes import make_d_shape, make_pyramid, make_terrace, make_square_peg
                    from core.dowel_joints import DowelConfig, _distribute_dowel_positions
                    count_map = {
                        6: self.spin_dshape_count.value(),
                        7: self.spin_pyramid_count.value(),
                        8: self.spin_terrace_count.value(),
                        9: self.spin_square_count.value(),
                    }
                    count = count_map.get(idx, 2)
                    cfg = DowelConfig(shape='round', count=count, radius=5)
                    positions = _distribute_dowel_positions(mesh, normal, origin, cfg)
                    for pos in positions:
                        try:
                            if idx == 6:  # D-shape
                                shape = make_d_shape(self.spin_dshape_radius.value(),
                                    self.spin_dshape_depth.value())
                            elif idx == 7:  # Pyramid
                                shape = make_pyramid(self.spin_pyramid_base.value(),
                                    self.spin_pyramid_depth.value(),
                                    self.spin_pyramid_taper.value())
                            elif idx == 8:  # Terrace
                                shape = make_terrace(self.spin_terrace_width.value(),
                                    self.spin_terrace_depth.value(),
                                    self.spin_terrace_steps.value())
                            elif idx == 9:  # Square
                                shape = make_square_peg(self.spin_square_size.value(),
                                    self.spin_square_depth.value())
                            else:
                                continue
                            if shape is not None:
                                # Align connector going INTO the part (opposite to face normal)
                                rot = _tm.geometry.align_vectors([0,0,1], (-n).tolist())
                                shape.apply_transform(rot)
                                # Get the depth for positioning
                                shape_depth = float(shape.extents[2]) if len(shape.extents) > 2 else 12.0
                                shape.apply_translation(pos - n * (shape_depth / 2))
                                preview_meshes.append(shape)
                        except Exception as e:
                            print(f"Connector preview error: {e}")

            self.viewport.set_joint_preview(preview_meshes)

        except Exception as e:
            print(f"Joint preview error: {e}")
            self.viewport.set_joint_preview([])

    def _find_cut_faces(self, mesh):
        """
        Find the flat cut faces on a mesh — the large planar regions
        created by slicing. Returns list of (normal, centre_point) tuples.
        """
        result = []
        try:
            if not hasattr(mesh, 'facets') or len(mesh.facets) == 0:
                return result

            # Sort facets by area, largest first
            order = np.argsort(mesh.facets_area)[::-1]

            # Take the top flat faces (area > 5% of total mesh area)
            total_area = mesh.area
            min_area = total_area * 0.03

            seen_faces = []  # (normal, centre) — dedupe by position, not just axis
            for i in order:
                if mesh.facets_area[i] < min_area:
                    break
                if i >= len(mesh.facets_normal):
                    continue
                normal = mesh.facets_normal[i]
                # Only include axis-aligned faces (cut faces are always flat/axis-aligned)
                abs_n = np.abs(normal)
                if np.max(abs_n) < 0.95:
                    continue  # skip non-axis-aligned faces
                # Get face centre
                facet = mesh.facets[i]
                verts = mesh.vertices[mesh.faces[facet].flatten()]
                centre = verts.mean(axis=0)
                # Avoid duplicates: skip if we already have a face at similar position
                is_dup = any(np.linalg.norm(centre - sc) < 5.0 for _, sc in seen_faces)
                if is_dup:
                    continue
                seen_faces.append((normal.copy(), centre.copy()))
                result.append((normal.copy(), centre.copy()))

            return result
        except Exception as e:
            print(f"Cut face detection error: {e}")
            return result

    def _face_axes_from_normal(self, normal):
        """Return two axes perpendicular to normal."""
        n = np.array(normal, dtype=float)
        h = np.array([0,0,1.0]) if abs(n[2])<0.9 else np.array([0,1.0,0])
        u = np.cross(n, h); u /= np.linalg.norm(u)
        v = np.cross(n, u); v /= np.linalg.norm(v)
        return u, v

    def _get_dowel_config(self):
        """Build a DowelConfig from current UI values."""
        from core.dowel_joints import DowelConfig
        idx = self.combo_joint_type.currentIndex()
        if idx == 1:  # round
            return DowelConfig(
                shape='round',
                count=self.spin_rod_count.value(),
                spacing=self.spin_rod_spacing.value(),
                depth_a=self.spin_depth_a.value(),
                depth_b=self.spin_depth_b.value(),
                tolerance=self.spin_rod_tol.value(),
                radius=self.spin_rod_radius.value(),
            )
        else:  # rect
            return DowelConfig(
                shape='rect',
                count=self.spin_rect_count.value(),
                depth_a=self.spin_rect_depth_a.value(),
                depth_b=self.spin_rect_depth_b.value(),
                tolerance=self.spin_rect_tol.value(),
                rect_width=self.spin_rect_w.value(),
                rect_height=self.spin_rect_h.value(),
            )

    def _apply_joint_config(self):
        """Store joint config on the selected part for use at export time."""
        sel = self.part_tree.selected_part
        if sel is None:
            self.lbl_joint_result.setText("⚠ Select a part first.")
            return
        idx = self.combo_joint_type.currentIndex()
        type_names = ['flat', 'round_dowel', 'rect_dowel', 'dovetail',
                      'magnet', 'snap_fit', 'd_shape', 'pyramid', 'terrace', 'square']
        joint_type = type_names[idx] if idx < len(type_names) else 'flat'

        # Store on the part
        if not hasattr(sel, '_joint_configs'):
            sel._joint_configs = {}

        if joint_type in ('round_dowel', 'rect_dowel'):
            sel._joint_configs['dowel'] = self._get_dowel_config()
        elif joint_type == 'magnet':
            from core.dowel_joints import DowelConfig
            sel._joint_configs['magnet'] = DowelConfig(
                shape='round',
                count=self.spin_magnet_count.value(),
                radius=self.spin_magnet_dia.value() / 2.0,
                depth_a=self.spin_magnet_depth.value(),
                depth_b=self.spin_magnet_depth.value(),
                tolerance=self.spin_magnet_tol.value(),
            )
        elif joint_type == 'snap_fit':
            from core.dowel_joints import DowelConfig
            sel._joint_configs['snap'] = DowelConfig(
                shape='rect',
                count=self.spin_snap_count.value(),
                rect_width=self.spin_snap_w.value(),
                rect_height=self.spin_snap_h.value(),
                tolerance=self.spin_snap_tol.value(),
            )
        elif joint_type == 'd_shape':
            sel._joint_configs['d_shape'] = {
                'radius': self.spin_dshape_radius.value(),
                'depth': self.spin_dshape_depth.value(),
                'count': self.spin_dshape_count.value(),
                'tolerance': self.spin_dshape_tol.value(),
            }
        elif joint_type == 'pyramid':
            sel._joint_configs['pyramid'] = {
                'base_size': self.spin_pyramid_base.value(),
                'depth': self.spin_pyramid_depth.value(),
                'count': self.spin_pyramid_count.value(),
                'taper': self.spin_pyramid_taper.value(),
            }
        elif joint_type == 'terrace':
            sel._joint_configs['terrace'] = {
                'width': self.spin_terrace_width.value(),
                'depth': self.spin_terrace_depth.value(),
                'steps': self.spin_terrace_steps.value(),
                'count': self.spin_terrace_count.value(),
            }
        elif joint_type == 'square':
            sel._joint_configs['square'] = {
                'size': self.spin_square_size.value(),
                'depth': self.spin_square_depth.value(),
                'count': self.spin_square_count.value(),
            }
        sel._joint_type = joint_type

        type_labels = ['Flat', 'Round dowel', 'Rect dowel', 'Dovetail',
                       'Magnet pocket', 'Snap-fit clip', 'D-Shape', 'Pyramid',
                       'Terrace', 'Square peg']
        label = type_labels[idx] if idx < len(type_labels) else 'Unknown'
        self.lbl_joint_result.setText(
            f"✓ {label} joint set on {sel.label}")
        self.status_bar.showMessage(
            f"Joint config saved on {sel.label} — applied at export")

    def _update_hollow_info(self):
        if self.mesh_handler.mesh is None or not self.chk_hollow.isChecked():
            self.lbl_hollow_info.setText(""); return
        info = estimate_material_saved(self.mesh_handler.mesh, self.spin_wall.value())
        self.lbl_hollow_info.setText(f"Saves ~{info['saving_pct']:.0f}% material  ({info['volume_solid_cm3']}→{info['volume_shell_cm3']} cm³)")

    def _export_parts(self):
        leaves = self.part_tree.get_all_leaves()
        if not leaves: return
        out_dir = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if not out_dir: return

        fmt        = self.combo_format.currentText().lower()
        do_hollow  = self.chk_hollow.isChecked()
        wall       = self.spin_wall.value()
        do_pins    = self.chk_pins.isChecked()
        pin_radius = self.spin_pin_radius.value()
        do_numbers = self.chk_part_numbers.isChecked()

        self._busy(True, f"Exporting {len(leaves)} parts in parallel…")
        self.btn_export.setEnabled(False)

        self._exp_w = ExportWorker(leaves, out_dir, fmt, do_hollow, wall,
                                    do_pins, pin_radius, do_numbers)
        self._exp_w.progress.connect(
            lambda d,t: self._update_progress(d, t, "Exporting"))
        self._exp_w.finished.connect(self._export_done)
        self._exp_w.error.connect(
            lambda e: (self._busy(False),
                       self.btn_export.setEnabled(True),
                       QMessageBox.critical(self, "Export Error", e)))
        self._exp_w.start()

    def _export_done(self, n_files, out_dir):
        self._busy(False); self.btn_export.setEnabled(True)
        self.lbl_export_result.setText(f"✓ {n_files} files → {out_dir}")
        self.status_bar.showMessage(f"Exported {n_files} parts.")
        QMessageBox.information(self, "Export Complete",
            f"Exported {n_files} part files to:\n{out_dir}")

    def _export_bambu(self):
        """Export all parts as a Bambu Studio-ready .3mf file."""
        leaves = self.part_tree.get_all_leaves()
        if not leaves:
            QMessageBox.warning(self, "No Parts", "Slice the model first."); return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Bambu .3mf", "", "Bambu Studio 3MF (*.3mf)")
        if not path: return
        if not path.endswith('.3mf'): path += '.3mf'

        profile = self._get_printer_profile()
        self._busy(True, f"Packing {len(leaves)} parts onto {profile.build_x:.0f}×{profile.build_y:.0f}mm plates…")
        self.btn_bambu_export.setEnabled(False)

        self._bambu_w = BambuExportWorker(
            leaves, path,
            plate_w=float(profile.usable_x),
            plate_h=float(profile.usable_y))
        self._bambu_w.finished.connect(self._bambu_done)
        self._bambu_w.error.connect(
            lambda e: (self._busy(False),
                       self.btn_bambu_export.setEnabled(True),
                       QMessageBox.critical(self, "Bambu Export Error", e)))
        self._bambu_w.start()

    def _bambu_done(self, ok, msg, plate_assignments):
        self._busy(False); self.btn_bambu_export.setEnabled(True)
        self._last_plate_assignments = plate_assignments
        self.btn_pdf_export.setEnabled(True)
        if ok:
            n_plates = len(plate_assignments)
            self.lbl_export_result.setText(
                f"✓ Bambu .3mf — {n_plates} plates")
            self.status_bar.showMessage(msg.split('\n')[0])
            QMessageBox.information(self, "Bambu Export Complete", msg)
        else:
            QMessageBox.critical(self, "Bambu Export Failed", msg)

    def _export_pdf(self):
        """Generate assembly guide PDF."""
        leaves = self.part_tree.get_all_leaves()
        if not leaves:
            QMessageBox.warning(self, "No Parts", "Slice the model first."); return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Assembly Guide PDF", "", "PDF Files (*.pdf)")
        if not path: return
        if not path.endswith('.pdf'): path += '.pdf'

        plate_assignments = getattr(self, '_last_plate_assignments', [])
        project_name = os.path.splitext(
            os.path.basename(self.mesh_handler.file_path or "Build"))[0]

        self._busy(True, "Generating assembly guide PDF…")
        self.btn_pdf_export.setEnabled(False)

        # Compute real estimates for the PDF
        from core.slicer import SlicedPart as SP
        class FakePart:
            def __init__(self, p): self.mesh=p.mesh; self.label=p.label; self.grid_index=(0,0,0); self.joint_type='flat'
        fake = [FakePart(p) for p in leaves]
        ests = estimate_all(fake, wall_thickness_mm=self.spin_wall.value(),
                            material=self.combo_material.currentText())
        summ = total_summary(ests)

        printer = self._get_printer_profile().name
        self._pdf_w = PdfExportWorker(
            leaves, path, plate_assignments, project_name,
            printer_name=printer,
            total_filament_g=summ['total_filament_g'],
            total_time_str=summ['total_time_str'],
            material=self.combo_material.currentText())
        self._pdf_w.finished.connect(self._pdf_done)
        self._pdf_w.error.connect(
            lambda e: (self._busy(False),
                       self.btn_pdf_export.setEnabled(True),
                       QMessageBox.critical(self, "PDF Error", e)))
        self._pdf_w.start()

    def _pdf_done(self, ok, path):
        self._busy(False); self.btn_pdf_export.setEnabled(True)
        if ok:
            self.lbl_export_result.setText(f"✓ PDF → {os.path.basename(path)}")
            self.status_bar.showMessage("Assembly guide PDF saved.")
            QMessageBox.information(self, "PDF Complete",
                f"Assembly guide saved to:\n{path}")
        else:
            QMessageBox.critical(self, "PDF Failed", "Could not generate PDF.")

    # ═══════════════════════════════════════════════════════
    # PROJECT SAVE / LOAD (simplified for new system)
    # ═══════════════════════════════════════════════════════

    def _save_project(self):
        if self.mesh_handler.mesh is None:
            QMessageBox.warning(self,"Nothing to save","Import a model first."); return
        path, _ = QFileDialog.getSaveFileName(self,"Save Project","","3D Print Slicer Project (*.ksproject)")
        if not path: return
        settings = {'wall_thickness': self.spin_wall.value(), 'hollow': self.chk_hollow.isChecked(),
                    'material': self.combo_material.currentText(), 'export_format': self.combo_format.currentText()}
        # Use a simplified slicer-compatible save
        from core.slicer import Slicer
        dummy_slicer = Slicer()
        dummy_slicer.cut_planes = []
        dummy_slicer.default_cut_size = self.spin_auto_size.value()
        ok = save_project(path, self.mesh_handler, dummy_slicer, settings)
        if ok: self.status_bar.showMessage(f"Saved: {os.path.basename(path)}")
        else: QMessageBox.critical(self,"Save Failed","Could not save project.")

    def _load_project(self):
        path, _ = QFileDialog.getOpenFileName(self,"Load Project","","3D Print Slicer Project (*.ksproject)")
        if not path: return
        data = load_project(path)
        if data is None:
            QMessageBox.critical(self,"Load Failed","Could not read project file."); return
        src = data.get('source_file','')
        if src and os.path.exists(src):
            self.mesh_handler.load(src)
            rz = data.get('resize',{})
            if rz:
                self.mesh_handler.target_x = rz.get('x', self.mesh_handler.target_x)
                self.mesh_handler.target_y = rz.get('y', self.mesh_handler.target_y)
                self.mesh_handler.target_z = rz.get('z', self.mesh_handler.target_z)
                self.mesh_handler.apply_resize()
            self.part_tree.load_mesh(self.mesh_handler.mesh)
            self._refresh_tree()
            verts, faces = self.mesh_handler.get_vertex_array()
            self.viewport.set_mesh(verts, faces, self.mesh_handler.get_bounds())
            self._lock_signals = True
            dims = self.mesh_handler.get_dimensions_mm()
            self.spin_x.setValue(round(dims[0],1)); self.spin_y.setValue(round(dims[1],1)); self.spin_z.setValue(round(dims[2],1))
            self._lock_signals = False
            self._update_pct_labels()
        settings = data.get('settings',{})
        if settings:
            self.spin_wall.setValue(settings.get('wall_thickness',3.0))
            self.chk_hollow.setChecked(settings.get('hollow',False))
            self.combo_material.setCurrentText(settings.get('material','PETG'))
            self.combo_format.setCurrentText(settings.get('export_format','STL'))
        self.status_bar.showMessage(f"Loaded: {os.path.basename(path)}")

    # ═══════════════════════════════════════════════════════
    # HELPERS
    # ═══════════════════════════════════════════════════════

    def _colour_split(self):
        """Split the loaded mesh into parts based on colour/material regions."""
        if self.mesh_handler.mesh is None:
            return
        self._busy(True, "Detecting colour regions…")
        try:
            raw = getattr(self.mesh_handler, 'raw_loaded', None)
            root_label = os.path.splitext(
                os.path.basename(self.mesh_handler.file_path or "Model"))[0]

            # Try scene-based split first (OBJ with materials / 3MF with components)
            if raw is not None and has_multi_geometry(raw):
                ok, msg = split_scene_by_geometry(raw, self.part_tree, root_label)
            else:
                ok, msg = split_by_colour(
                    self.mesh_handler.mesh, self.part_tree, root_label,
                    tolerance=self.spin_colour_tol.value())

            self._busy(False)

            if not ok:
                QMessageBox.information(self, "No Colour Data",
                    f"{msg}\n\nThis works best with OBJ files that have materials,\n"
                    "or 3MF files with multiple coloured bodies.")
                return

            self._refresh_tree()
            self._refresh_viewport()
            self._refresh_parts_tables()
            self.btn_export.setEnabled(True)
            self.btn_dryfit.setEnabled(True)
            n = len(self.part_tree.get_all_leaves())
            self.lbl_colour_info.setText(f"✓ {msg}")
            self.status_bar.showMessage(f"Colour split: {n} regions")

        except Exception as e:
            self._busy(False)
            QMessageBox.critical(self, "Colour Split Error", str(e))

    def _toggle_dryfit(self):
        """Toggle between dry-fit (assembled) view and exploded view."""
        if not hasattr(self, '_dryfit_mode'):
            self._dryfit_mode = False
        self._dryfit_mode = not self._dryfit_mode

        if self._dryfit_mode:
            self.btn_dryfit.setText("Back to Edit View")
            if _HAS_QTA:
                self.btn_dryfit.setIcon(qta.icon('fa5s.edit', color='#50e080'))
            self.btn_dryfit.setStyleSheet("background:#1a3020;color:#50e080;border:1px solid #2a5030;border-radius:5px;")
            self.status_bar.showMessage("Dry-fit view: all parts shown in assembled position")
            # In dry-fit mode all parts are shown, viewport shows assembled state
            self.part_tree.show_all()
        else:
            self.btn_dryfit.setText("Preview Assembly (Dry-Fit)")
            self._set_btn_icon(self.btn_dryfit, 'preview')
            self.btn_dryfit.setStyleSheet("")
            self.status_bar.showMessage("Edit view")

        self._refresh_tree()
        self._refresh_viewport()

    def _run_bond_analysis(self):
        """Analyse bonding surface area for all cut faces."""
        leaves = self.part_tree.get_all_leaves()
        if not leaves:
            self.lbl_bond.setText("No parts — slice first."); return
        from core.bond_analysis import analyse_all_parts, bond_summary
        analyses = analyse_all_parts(leaves)
        summary = bond_summary(analyses)
        details = []
        for a in analyses:
            if a['min_quality'] < 70 and a['worst_face']:
                wf = a['worst_face']
                details.append(
                    f"  {a['label']}: {wf['area_mm2']}mm² ({wf['width_mm']}×{wf['height_mm']}mm) "
                    f"— quality {wf['bond_quality']}% — {wf['suggestion']}")
        detail_str = "\n".join(details[:6]) if details else ""
        self.lbl_bond.setText(f"{summary}\n{detail_str}" if detail_str else summary)
        self.status_bar.showMessage("Bond analysis complete.")

    def _run_printability_check(self):
        """Run combined printability check on all parts."""
        leaves = self.part_tree.get_all_leaves()
        if not leaves:
            self.status_bar.showMessage("No parts to check."); return
        self._busy(True, "Running printability checks…")
        try:
            from core.printability import check_all_parts, printability_summary
            profile = self._get_printer_profile()
            bv = (profile.usable_x, profile.usable_y, profile.usable_z)
            results = check_all_parts(leaves, build_volume=bv)
            summary = printability_summary(results)
            # Build detailed message
            details = []
            for r in results:
                if r['issues']:
                    issues_str = "; ".join(i['message'] for i in r['issues'][:3])
                    details.append(f"  {r['label']} [{r['colour'].upper()}]: {issues_str}")
            detail_str = "\n".join(details[:8])
            self._busy(False)
            QMessageBox.information(self, "Printability Check",
                f"{summary}\n\n{detail_str}" if detail_str else summary)
            self.status_bar.showMessage("Printability check complete.")
        except Exception as e:
            self._busy(False)
            QMessageBox.warning(self, "Check Error", str(e))

    def _run_wall_check(self):
        """Check wall thickness on all parts."""
        leaves = self.part_tree.get_all_leaves()
        if not leaves:
            self.status_bar.showMessage("No parts to check."); return
        self._busy(True, "Checking wall thickness…")
        try:
            from core.wall_thickness import analyse_all_parts as wt_analyse
            results = wt_analyse(leaves, min_thickness=1.5)
            lines = []
            for r in results:
                if r.get('pct_thin', 0) > 0:
                    lines.append(f"  {r['label']}: {r['pct_thin']}% thin (min {r['min_found_mm']}mm)")
                else:
                    lines.append(f"  {r['label']}: OK (min {r.get('min_found_mm', '?')}mm)")
            self._busy(False)
            msg = "\n".join(lines[:12])
            QMessageBox.information(self, "Wall Thickness", msg)
            self.status_bar.showMessage("Wall thickness check complete.")
        except Exception as e:
            self._busy(False)
            QMessageBox.warning(self, "Check Error", str(e))

    def _preview_face_labels(self):
        """Show label markers on cut faces in the viewport."""
        leaves = self.part_tree.get_all_leaves()
        if not leaves:
            self.status_bar.showMessage("No parts — slice first."); return
        from core.face_labels import generate_face_labels
        labels = generate_face_labels(leaves)
        if not labels:
            self.viewport.face_label_markers = []
            self.viewport.update()
            self.status_bar.showMessage("No matching cut faces found."); return

        # Collect all label markers for the viewport
        all_markers = []
        total = 0
        for part in leaves:
            if part.id in labels:
                for centre, normal, text in labels[part.id]:
                    all_markers.append((centre, normal, text))
                    total += 1

        # Push to viewport for rendering
        self.viewport.face_label_markers = all_markers
        self.viewport.update()

        pairs = total // 2
        self.status_bar.showMessage(
            f"Label preview: {pairs} matched joint pairs shown on model. "
            f"Click Preview Labels again to refresh, or export to emboss.")

    def _export_tolerance_test(self):
        """Export a tolerance test STL for the current joint type."""
        out_dir = QFileDialog.getExistingDirectory(self, "Save Tolerance Test To")
        if not out_dir: return
        from core.tolerance_test import export_tolerance_test
        idx = self.combo_joint_type.currentIndex()
        type_map = {0:'round_dowel', 1:'round_dowel', 2:'rect_dowel',
                    3:'round_dowel', 4:'magnet', 5:'snap_fit'}
        joint_type = type_map.get(idx, 'round_dowel')
        path = export_tolerance_test(
            out_dir,
            joint_type=joint_type,
            base_radius=self.spin_rod_radius.value(),
            rect_width=self.spin_rect_w.value(),
            rect_height=self.spin_rect_h.value(),
            magnet_diameter=self.spin_magnet_dia.value() if hasattr(self, 'spin_magnet_dia') else 6.0,
            magnet_depth=self.spin_magnet_depth.value() if hasattr(self, 'spin_magnet_depth') else 3.2,
        )
        if path:
            self.status_bar.showMessage(f"Tolerance test exported: {os.path.basename(path)}")
            QMessageBox.information(self, "Tolerance Test",
                f"Exported tolerance test to:\n{path}\n\n"
                "Print it, test the fit at each tolerance,\n"
                "then use the best value in 3D Print Slicer.")
        else:
            QMessageBox.warning(self, "Error", "Could not generate tolerance test.")

    def _run_overhang_analysis(self):
        """Analyse overhangs for all leaf parts."""
        leaves = self.part_tree.get_all_leaves()
        if not leaves:
            self.lbl_overhang.setText("No parts — slice first."); return
        from core.overhang_analysis import analyse_all_parts, overhang_summary
        analyses = analyse_all_parts(leaves)
        summ = overhang_summary(analyses)
        # Show per-part results
        lines = []
        for a in analyses:
            icon = "⚠" if a['needs_support'] else "✓"
            lines.append(f"{icon} {a['label']}: {a['overhang_pct']}% overhang — {a['suggestion']}")
        summary_line = (f"\n{summ['parts_needing_support']}/{summ['total_parts']} parts need supports"
                        f" (avg {summ['avg_overhang_pct']}% overhang)")
        self.lbl_overhang.setText("\n".join(lines[:8]) + summary_line)
        self.status_bar.showMessage(
            f"Overhang analysis: {summ['parts_needing_support']} of {summ['total_parts']} parts need supports")

    def _orient_flat_down(self):
        """Rotate selected part so its largest flat face is on the bottom."""
        sel = self.part_tree.selected_part
        if sel is None or not sel.is_leaf:
            self.status_bar.showMessage("Select a leaf part first."); return
        self.part_tree.push_mesh_snapshot([sel], "orient flat")
        self.btn_undo.setEnabled(True); self._update_undo_tooltip()
        try:
            from exports.bambu_export import _orient_flat_down
            sel.mesh = _orient_flat_down(sel.mesh)
            self._vbo_invalidate_part(sel)
            self._refresh_viewport()
            self.status_bar.showMessage(f"Oriented {sel.label} flat-down.")
        except Exception as e:
            self.status_bar.showMessage(f"Orientation error: {e}")

    def _orient_rotate(self, axis, degrees):
        """Rotate selected part by degrees around the given axis (0=X, 1=Y, 2=Z)."""
        sel = self.part_tree.selected_part
        if sel is None or not sel.is_leaf:
            self.status_bar.showMessage("Select a leaf part first."); return
        self.part_tree.push_mesh_snapshot([sel], f"rotate {['X','Y','Z'][axis]} {degrees}°")
        self.btn_undo.setEnabled(True); self._update_undo_tooltip()
        import trimesh
        angle = np.radians(degrees)
        rot = trimesh.transformations.rotation_matrix(angle, [1 if axis==0 else 0,
                                                               1 if axis==1 else 0,
                                                               1 if axis==2 else 0])
        sel.mesh.apply_transform(rot)
        # Re-center on Z=0
        sel.mesh.apply_translation([0, 0, -sel.mesh.bounds[0][2]])
        self._vbo_invalidate_part(sel)
        self._refresh_viewport()
        self.status_bar.showMessage(f"Rotated {sel.label} {degrees}° around {['X','Y','Z'][axis]}.")

    def _on_dowel_placed(self, pos, normal):
        """A dowel marker was placed by clicking on the model."""
        sel = self.part_tree.selected_part
        if sel is None:
            self.status_bar.showMessage("Select a part first to place dowels.")
            return
        if not hasattr(sel, '_manual_dowel_positions'):
            sel._manual_dowel_positions = []
        pos_arr = np.array(pos, dtype=float)
        nrm_arr = np.array(normal, dtype=float)
        sel._manual_dowel_positions.append((pos_arr, nrm_arr))

        # Auto-match: find adjacent part and add mirrored marker
        adj = self._find_adjacent_part(sel, pos_arr, nrm_arr)
        if adj:
            if not hasattr(adj, '_manual_dowel_positions'):
                adj._manual_dowel_positions = []
            # Mirror: same position, opposite normal
            adj._manual_dowel_positions.append((pos_arr.copy(), -nrm_arr))

        # Update viewport markers (show markers for selected part)
        self.viewport.dowel_markers = list(sel._manual_dowel_positions)
        self.viewport.update()
        n = len(sel._manual_dowel_positions)
        adj_msg = f" (mirrored to {adj.label})" if adj else ""
        self.lbl_dowel_count.setText(f"{n} marker{'s' if n != 1 else ''}")
        self.status_bar.showMessage(
            f"Dowel placed at ({pos_arr[0]:.1f}, {pos_arr[1]:.1f}, {pos_arr[2]:.1f}){adj_msg}")

    def _find_adjacent_part(self, source_part, position, normal):
        """Find the neighbouring part that shares the cut face at this position."""
        leaves = self.part_tree.get_all_leaves()
        pos = np.array(position)
        nrm = np.array(normal)
        for part in leaves:
            if part.id == source_part.id:
                continue
            # Check if this part's bounds contain the dowel position
            # (with tolerance since the position is on the cut face boundary)
            bounds = part.mesh.bounds
            tol = 5.0  # mm tolerance for boundary detection
            if (pos[0] >= bounds[0][0] - tol and pos[0] <= bounds[1][0] + tol and
                pos[1] >= bounds[0][1] - tol and pos[1] <= bounds[1][1] + tol and
                pos[2] >= bounds[0][2] - tol and pos[2] <= bounds[1][2] + tol):
                return part
        return None

    def _save_export_preset(self):
        """Save current export settings as a named preset."""
        import json
        name, ok = QFileDialog.getSaveFileName(
            self, "Save Export Preset", "", "Preset Files (*.json)")
        if not name or not ok: return
        if not name.endswith('.json'): name += '.json'
        preset = {
            'format': self.combo_format.currentText(),
            'material': self.combo_material.currentText(),
            'hollow': self.chk_hollow.isChecked(),
            'wall_thickness': self.spin_wall.value(),
            'pins': self.chk_pins.isChecked(),
            'pin_radius': self.spin_pin_radius.value(),
            'pin_depth': self.spin_pin_depth.value(),
            'part_numbers': self.chk_part_numbers.isChecked(),
            'joint_type': self.combo_joint_type.currentIndex(),
            'printer': self.combo_printer.currentText(),
            'cut_size': self.spin_auto_size.value(),
        }
        try:
            with open(name, 'w') as f:
                json.dump(preset, f, indent=2)
            basename = os.path.basename(name).replace('.json', '')
            if self.combo_export_preset.findText(basename) < 0:
                self.combo_export_preset.addItem(basename)
            self.combo_export_preset.setCurrentText(basename)
            self.status_bar.showMessage(f"Preset saved: {basename}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    def _load_export_preset(self):
        """Load export settings from a preset file."""
        import json
        name, _ = QFileDialog.getOpenFileName(
            self, "Load Export Preset", "", "Preset Files (*.json)")
        if not name: return
        try:
            with open(name, 'r') as f:
                preset = json.load(f)
            if 'format' in preset:
                self.combo_format.setCurrentText(preset['format'])
            if 'material' in preset:
                self.combo_material.setCurrentText(preset['material'])
            if 'hollow' in preset:
                self.chk_hollow.setChecked(preset['hollow'])
            if 'wall_thickness' in preset:
                self.spin_wall.setValue(preset['wall_thickness'])
            if 'pins' in preset:
                self.chk_pins.setChecked(preset['pins'])
            if 'pin_radius' in preset:
                self.spin_pin_radius.setValue(preset['pin_radius'])
            if 'pin_depth' in preset:
                self.spin_pin_depth.setValue(preset['pin_depth'])
            if 'part_numbers' in preset:
                self.chk_part_numbers.setChecked(preset['part_numbers'])
            if 'joint_type' in preset:
                self.combo_joint_type.setCurrentIndex(preset['joint_type'])
            if 'printer' in preset:
                self.combo_printer.setCurrentText(preset['printer'])
            if 'cut_size' in preset:
                self.spin_auto_size.setValue(preset['cut_size'])
            basename = os.path.basename(name).replace('.json', '')
            if self.combo_export_preset.findText(basename) < 0:
                self.combo_export_preset.addItem(basename)
            self.combo_export_preset.setCurrentText(basename)
            self.status_bar.showMessage(f"Preset loaded: {basename}")
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))

    def _toggle_manual_dowel_mode(self, active):
        """Toggle manual dowel placement mode with a step-by-step HUD overlay
        in the viewport so the workflow is discoverable without docs."""
        self.viewport.manual_dowel_mode = active
        if active:
            sel = self.part_tree.selected_part
            if sel is None or not sel.is_leaf:
                self.status_bar.showMessage("⚠ Select a leaf part in the tree FIRST, then enable manual placement.")
                self.lbl_status.setText("⚠ Select a part first!")
                self.lbl_status.setStyleSheet("color:#f05050;font-weight:700;")
                self.chk_manual_dowels.setChecked(False)
                self.viewport.manual_dowel_mode = False
                return
            self.viewport.setCursor(Qt.CrossCursor)
            self.lbl_status.setText(f"📌 DOWEL PLACEMENT — click on {sel.label}")
            self.lbl_status.setStyleSheet("color:#40d0a0;font-weight:700;")
            self.status_bar.showMessage(
                f"Click on cut faces of {sel.label} to place dowel markers. "
                f"Drag placed markers to fine-tune. Uncheck the box when done.")
            # Step-by-step HUD — this removes the "not intuitive" problem.
            self.viewport.set_hud(
                "DOWEL PLACEMENT\n"
                "─────────────────────\n"
                f"Part:  {sel.label}\n\n"
                "1.  Click on a flat cut-face to drop a marker\n"
                "2.  Drag any marker to slide it along the surface\n"
                "3.  Repeat to add more (they mirror to the mate)\n"
                "4.  Un-tick ‘Manual Placement’ to commit",
                color='#6ae0b0')
            # Show existing markers for this part
            if hasattr(sel, '_manual_dowel_positions'):
                self.viewport.dowel_markers = list(sel._manual_dowel_positions)
            else:
                self.viewport.dowel_markers = []
            self.viewport._schedule_refresh(markers=True)
        else:
            self.viewport.setCursor(Qt.ArrowCursor)
            self.lbl_status.setStyleSheet("")
            self.lbl_status.setText(self._get_status_text())
            self.viewport.set_hud(None)
            self.status_bar.showMessage("Dowel placement off.")
            # Persist dragged-final positions back onto the part.
            sel = self.part_tree.selected_part
            if sel is not None:
                sel._manual_dowel_positions = list(self.viewport.dowel_markers)
        self._update_mode_indicator()

    def _clear_manual_dowels(self):
        """Remove all manual dowel markers from the selected part."""
        sel = self.part_tree.selected_part
        if sel:
            sel._manual_dowel_positions = []
        self.viewport.dowel_markers = []
        self.viewport.update()
        self.lbl_dowel_count.setText("0 markers")
        self.status_bar.showMessage("Manual dowel markers cleared.")

    # ═══════════════════════════════════════════════════════
    # UNDO HISTORY PANEL
    # ═══════════════════════════════════════════════════════

    def _refresh_history(self):
        """Refresh the undo history table."""
        stack = self.part_tree._undo_stack
        self.history_list.setRowCount(len(stack))
        for i, record in enumerate(stack):
            action = record.get('action', '')
            if action == 'split':
                part = record.get('part')
                label = part.label if part else '?'
                detail = f"Split {label}"
                action_str = "Cut"
            elif action == 'mesh_edit':
                label = record.get('label', 'edit')
                n = len(record.get('snapshots', []))
                detail = f"{label} ({n} part{'s' if n>1 else ''})"
                action_str = "Edit"
            else:
                action_str = action
                detail = ""
            self.history_list.setItem(i, 0, QTableWidgetItem(str(i+1)))
            act_item = QTableWidgetItem(action_str)
            act_item.setForeground(QColor("#80c0ff" if action == 'split' else "#f0a020"))
            self.history_list.setItem(i, 1, act_item)
            self.history_list.setItem(i, 2, QTableWidgetItem(detail))
        self.lbl_history_count.setText(f"{len(stack)} action{'s' if len(stack) != 1 else ''}")

    def _undo_to_selected(self):
        """Undo all actions down to the selected row in the history table."""
        row = self.history_list.currentRow()
        if row < 0: return
        stack_len = len(self.part_tree._undo_stack)
        undos_needed = stack_len - row
        if undos_needed <= 0: return
        for _ in range(undos_needed):
            desc = self.part_tree.undo()
            if desc is None: break
        self.btn_undo.setEnabled(self.part_tree.can_undo())
        self._update_undo_tooltip()
        self._refresh_tree()
        self._refresh_viewport()
        self._refresh_parts_tables()
        self._refresh_history()
        self.status_bar.showMessage(f"Undone {undos_needed} action(s) — back to step {row}")

    # ═══════════════════════════════════════════════════════
    # DRAG AND DROP
    # ═══════════════════════════════════════════════════════

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            for url in e.mimeData().urls():
                path = url.toLocalFile().lower()
                if path.endswith(('.stl', '.obj', '.3mf')):
                    e.acceptProposedAction()
                    return
        e.ignore()

    def dropEvent(self, e):
        for url in e.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith(('.stl', '.obj', '.3mf')):
                self._busy(True, "Loading model…")
                self.btn_import.setEnabled(False)
                self._load_w = LoadWorker(self.mesh_handler, path)
                self._load_w.finished.connect(self._import_done)
                self._load_w.error.connect(
                    lambda e: (self._busy(False), self.btn_import.setEnabled(True),
                               QMessageBox.critical(self, "Import Failed", e)))
                self._load_w.start()
                return

    # ═══════════════════════════════════════════════════════
    # AUTO-ORIENT ALL PARTS
    # ═══════════════════════════════════════════════════════

    def _orient_all_parts(self):
        """Rotate every leaf part so its largest flat face is on the bottom."""
        leaves = self.part_tree.get_all_leaves()
        if not leaves:
            self.status_bar.showMessage("No parts to orient."); return
        # Save undo
        self.part_tree.push_mesh_snapshot(leaves, "orient all")
        self.btn_undo.setEnabled(True); self._update_undo_tooltip()
        count = 0
        for part in leaves:
            try:
                oriented = self._orient_flat_down(part.mesh)
                if oriented is not part.mesh:
                    part.mesh = oriented
                    count += 1
            except Exception:
                pass
        self._refresh_viewport()
        self._refresh_history()
        self.status_bar.showMessage(f"Oriented {count} of {len(leaves)} parts flat-face-down.")

    def _orient_flat_down(self, mesh):
        """Rotate mesh so its largest flat face is on the bottom."""
        import trimesh
        try:
            if not hasattr(mesh, 'facets') or len(mesh.facets) == 0:
                return mesh
            best = int(np.argmax(mesh.facets_area))
            normal = mesh.facets_normal[best]
            target = np.array([0., 0., -1.])
            if np.allclose(normal, target, atol=0.05):
                return mesh
            rot = trimesh.geometry.align_vectors(normal, target)
            oriented = mesh.copy()
            oriented.apply_transform(rot)
            oriented.apply_translation([0, 0, -oriented.bounds[0][2]])
            return oriented
        except Exception:
            return mesh

    # ═══════════════════════════════════════════════════════
    # MEASUREMENT TOOL
    # ═══════════════════════════════════════════════════════

    def _toggle_measure_mode(self, active):
        """Toggle measurement mode — click two points to measure distance."""
        self._measure_mode = active
        self._measure_points = []
        self.viewport.measure_points = []
        self.viewport._measure_active = active
        self.viewport.update()
        if active:
            self.viewport.setCursor(Qt.CrossCursor)
            self.lbl_measure.setText("Click first point on model…")
            self.lbl_status.setText("📏 MEASURE MODE")
            self.lbl_status.setStyleSheet("color:#f0d040;font-weight:700;")
            self.status_bar.showMessage("Measure mode: click two points on the model")
        else:
            self.viewport.setCursor(Qt.ArrowCursor)
            self.lbl_measure.setText("")
            self.lbl_status.setStyleSheet("")
            self.lbl_status.setText(self._get_status_text())
            self.status_bar.showMessage("Measure mode off")

    def _get_status_text(self):
        """Get the default status label text."""
        if self.mesh_handler.file_path:
            return os.path.basename(self.mesh_handler.file_path)
        return "No model loaded"

    def _on_measure_click(self, pos, normal):
        """Handle a measurement click from the viewport."""
        if not self._measure_mode: return
        self._measure_points.append(np.array(pos))
        self.viewport.measure_points = [p.copy() for p in self._measure_points]
        self.viewport.update()

        if len(self._measure_points) == 1:
            self.lbl_measure.setText("Click second point…")
        elif len(self._measure_points) >= 2:
            p1, p2 = self._measure_points[0], self._measure_points[1]
            dist = float(np.linalg.norm(p2 - p1))
            dx = abs(p2[0] - p1[0]); dy = abs(p2[1] - p1[1]); dz = abs(p2[2] - p1[2])
            self.lbl_measure.setText(
                f"Distance: {dist:.1f} mm\n"
                f"  ΔX={dx:.1f}  ΔY={dy:.1f}  ΔZ={dz:.1f}")
            self.status_bar.showMessage(f"Measured: {dist:.1f}mm")
            # Reset for next measurement but keep showing result
            self._measure_points = []

    # ═══════════════════════════════════════════════════════
    # CONTEXT MENU + DOUBLE-CLICK + HOVER
    # ═══════════════════════════════════════════════════════

    def _on_viewport_part_picked(self, part):
        """Right-click on a part in the viewport — show context menu."""
        if part is None: return
        menu = QMenu(self)
        menu.setStyleSheet("QMenu{background:#283848;color:#e0e8f0;border:1px solid #3a5060;}"
                           "QMenu::item:selected{background:#3a6090;}")

        # Select this part
        act_sel = menu.addAction(f"Select: {part.label}")
        act_sel.triggered.connect(lambda: self._viewport_select_part(part))
        menu.addSeparator()

        # Solo — hide everything else
        act_solo = menu.addAction("Solo (show only this part)")
        act_solo.triggered.connect(lambda: self._viewport_solo_part(part))

        # Show all
        act_show = menu.addAction("Show All Parts")
        act_show.triggered.connect(self._show_all_parts)

        menu.addSeparator()
        act_hide = menu.addAction("Hide This Part")
        act_hide.triggered.connect(lambda: self._viewport_hide_part(part))

        if part.is_leaf:
            menu.addSeparator()
            act_cut = menu.addAction(_qicon('cut') if _HAS_QTA else QIcon(),
                                      _icon_label('cut', 'Cut This Part'))
            act_cut.triggered.connect(lambda: self._context_cut_part(part))
            act_orient = menu.addAction(_qicon('flat') if _HAS_QTA else QIcon(), "Orient Flat Down")
            act_orient.triggered.connect(lambda: self._context_orient(part))

        menu.exec_(QCursor.pos())

    def _viewport_select_part(self, part):
        """Select a part from viewport click. Only updates the tree selection
        (which cascades into viewport update via _tree_selection_changed);
        avoids double-calling set_selected_part and a full tree rebuild."""
        self.part_tree.select(part)
        # Just move the selection in the tree widget — _tree_selection_changed
        # handles viewport + info panel + cut preview in one pass.
        self._select_tree_item_by_id(part.id)
        self.status_bar.showMessage(f"Selected: {part.label}")

    def _viewport_solo_part(self, part):
        """Solo a part from viewport right-click — hide everything else."""
        target_ids = {p.id for p in (part.all_leaves() if not part.is_leaf else [part])}
        for p in self.part_tree.get_all_leaves():
            p.visible = (p.id in target_ids)
        self.part_tree.select(part)
        self.viewport.set_selected_part(part.id)
        self._refresh_tree()
        self._refresh_viewport()
        self.status_bar.showMessage(f"Solo: {part.label}")

    def _viewport_hide_part(self, part):
        """Hide a part from viewport right-click."""
        part.visible = False
        self._refresh_tree()
        self._refresh_viewport()

    def _tree_context_menu(self, pos):
        """Right-click context menu on parts tree."""
        item = self.parts_tree_widget.itemAt(pos)
        if item is None: return
        part_id = item.data(0, Qt.UserRole)
        part = self.part_tree.find_by_id(part_id)
        if part is None: return

        menu = QMenu(self)
        menu.setStyleSheet("QMenu{background:#1a2030;color:#c0c8d8;border:1px solid #2a3548;}"
                           "QMenu::item:selected{background:#2a4878;}")

        if part.is_leaf:
            act_cut = menu.addAction(_qicon('cut') if _HAS_QTA else QIcon(),
                                      _icon_label('cut', 'Cut This Part'))
            act_cut.triggered.connect(lambda: self._context_cut_part(part))
            menu.addSeparator()
            act_solo = menu.addAction("Solo (hide others)")
            act_solo.triggered.connect(lambda: self._context_solo(part))
            act_hide = menu.addAction("Hide" if part.visible else "Show")
            act_hide.triggered.connect(lambda: self._context_toggle_vis(part))
            act_wf = menu.addAction("Wireframe" if not getattr(part, '_wireframe', False) else "Solid")
            act_wf.triggered.connect(lambda: self._context_toggle_wf(part))
            menu.addSeparator()
            act_joint = menu.addAction("Set Joint Type…")
            act_joint.triggered.connect(lambda: self._context_set_joint(part))
            act_orient = menu.addAction("Orient Flat Down")
            act_orient.triggered.connect(lambda: self._context_orient(part))
        if self.part_tree.can_undo():
            menu.addSeparator()
            act_undo = menu.addAction("↩ Undo Last")
            act_undo.triggered.connect(self._undo)

        menu.exec_(self.parts_tree_widget.mapToGlobal(pos))

    def _context_cut_part(self, part):
        self.part_tree.select(part)
        self._refresh_tree()
        if hasattr(self, '_sections'):
            s = self._sections.get('cut')
            if s: s.expand()
        self.status_bar.showMessage(f"Selected {part.label} — set cut position and click Cut")

    def _context_solo(self, part):
        target_ids = {p.id for p in (part.all_leaves() if not part.is_leaf else [part])}
        for p in self.part_tree.get_all_leaves():
            p.visible = (p.id in target_ids)
        self._refresh_tree(); self._refresh_viewport()

    def _context_toggle_vis(self, part):
        part.visible = not part.visible
        self._refresh_tree(); self._refresh_viewport()

    def _context_toggle_wf(self, part):
        part._wireframe = not getattr(part, '_wireframe', False)
        self._refresh_tree(); self._refresh_viewport()

    def _context_set_joint(self, part):
        self.part_tree.select(part)
        self._refresh_tree()
        if hasattr(self, '_sections'):
            s = self._sections.get('joints')
            if s: s.expand()

    def _context_orient(self, part):
        self.part_tree.push_mesh_snapshot([part], f"orient {part.label}")
        self.btn_undo.setEnabled(True); self._update_undo_tooltip()
        part.mesh = self._orient_flat_down(part.mesh)
        self._refresh_viewport(); self._vbo_invalidate_part(part)

    def _vbo_invalidate_part(self, part):
        """Trigger re-render for a specific part (PyVista handles caching)."""
        self.viewport.update()

    def _tree_double_clicked(self, item, column):
        """Double-click centres the viewport on the clicked part."""
        part_id = item.data(0, Qt.UserRole)
        part = self.part_tree.find_by_id(part_id)
        if part is None: return
        try:
            bounds = part.get_bounds()
            centre = (bounds[0] + bounds[1]) / 2.0
            size = float(np.max(bounds[1] - bounds[0]))
            # Animate to target
            self._animate_camera(-float(centre[0]), -float(centre[1]), -size * 2.5)
        except Exception:
            pass

    def _animate_camera(self, target_px, target_py, target_zoom):
        """Move camera to focus on a point. PyVista handles smooth rendering."""
        try:
            self.viewport.plotter.camera.focal_point = (target_px, target_py, 0)
            self.viewport.plotter.camera.position = (
                target_px, target_py, abs(target_zoom))
            self.viewport.plotter.render()
        except Exception:
            pass

    def _tree_item_hovered(self, item, column):
        """Hover preview — temporarily show a hidden part when hovering its name."""
        # Restart the clear timer — if mouse stops hovering, preview clears after 300ms
        self._hover_timer.stop()
        self._hover_timer.start(300)

        if item is None:
            return
        part_id = item.data(0, Qt.UserRole)
        if part_id == self._hover_highlight_id:
            return  # same item, skip

        # End previous preview first
        self._end_hover_preview()

        part = self.part_tree.find_by_id(part_id)
        if part is None or not part.is_leaf:
            return

        self._hover_highlight_id = part_id

        # If part is hidden, temporarily show it as a preview
        if not part.visible:
            self._hover_was_hidden = True
            part.visible = True
            self._refresh_viewport()
        else:
            self._hover_was_hidden = False

    def _end_hover_preview(self):
        """End hover preview — hide the temporarily shown part."""
        if self._hover_highlight_id and getattr(self, '_hover_was_hidden', False):
            part = self.part_tree.find_by_id(self._hover_highlight_id)
            if part and part.is_leaf:
                part.visible = False
                self._refresh_viewport()
        self._hover_highlight_id = None
        self._hover_was_hidden = False

    def _clear_hover_highlight(self):
        self._end_hover_preview()
        self.viewport._hover_part_id = None
        self.viewport.update()

    # ═══════════════════════════════════════════════════════
    # BUILD VOLUME TOGGLE + QUICK-CUT BAR COLOUR
    # ═══════════════════════════════════════════════════════

    def _toggle_build_volume(self, show):
        """Toggle build volume outline in viewport."""
        if show:
            profile = self._get_printer_profile()
            self.viewport.build_volume = (
                float(profile.usable_x),
                float(profile.usable_y),
                float(profile.usable_z))
        else:
            self.viewport.build_volume = None
        self.viewport.update()

    def _update_qc_bar_colour(self, idx):
        """Colour the quick-cut bar background based on cut mode."""
        colours = {
            0: "#10121a",  # Full — default dark
            1: "#1a1420",  # Angled — slight purple
            2: "#101a1a",  # Section — slight teal
            3: "#1a1810",  # Groove — slight orange
            4: "#101a10",  # Natural — slight green
        }
        col = colours.get(idx, "#10121a")
        # Find the cut bar widget and update its style
        for child in self.findChildren(QWidget, "cutBar"):
            child.setStyleSheet(f"background:{col};border-top:1px solid #1a1d2a;border-bottom:1px solid #1a1d2a;")

    # ═══════════════════════════════════════════════════════
    # SELECTION-AWARE HELPERS
    # ═══════════════════════════════════════════════════════

    def _get_target_parts(self):
        """Get the parts that the current action should apply to.
        Returns (parts_list, description) based on selection state."""
        sel = self.part_tree.selected_part
        leaves = self.part_tree.get_all_leaves()

        if sel and sel.is_leaf:
            return [sel], f"'{sel.label}'"
        elif sel and not sel.is_leaf:
            # Non-leaf selected = all its children
            children = sel.all_leaves()
            return children, f"all {len(children)} parts under '{sel.label}'"
        elif leaves:
            return leaves, f"all {len(leaves)} parts"
        elif self.part_tree.root:
            return [self.part_tree.root], "the model"
        return [], "nothing"

    def _get_target_mesh(self):
        """Get the single mesh to operate on.
        If a leaf part is selected, returns that part's mesh.
        Otherwise returns the root mesh."""
        sel = self.part_tree.selected_part
        if sel and sel.is_leaf:
            return sel.mesh, sel.label
        elif self.mesh_handler.mesh is not None:
            return self.mesh_handler.mesh, "model"
        return None, ""

    def _run_threaded(self, func, args, busy_msg, on_done, on_error=None):
        """Run a function in a background thread with busy indicator.
        func: callable to run
        args: tuple of arguments
        busy_msg: status bar message while running
        on_done: callback(result) when finished
        on_error: callback(error_str) on failure
        """
        self._busy(True, busy_msg)
        self._task_worker = TaskWorker(func, *args)
        self._task_worker.finished.connect(lambda r: (self._busy(False), on_done(r)))
        self._task_worker.error.connect(
            lambda e: (self._busy(False),
                       on_error(e) if on_error else
                       QMessageBox.warning(self, "Error", e)))
        self._task_worker.start()

    def _busy(self, state, msg=""):
        self.progress_bar.setVisible(state)
        if state:
            self.progress_bar.setRange(0, 0)
            self.status_bar.showMessage(msg)
            self.setWindowTitle(f"3D Print Slicer — {msg}")
        else:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            self.setWindowTitle(self._original_title)

    def _toggle_selected_visibility(self):
        """Keyboard shortcut: toggle visibility of selected part."""
        sel = self.part_tree.selected_part
        if sel and sel.is_leaf:
            sel.visible = not sel.visible
            self._refresh_tree()
            self._refresh_viewport()

    def _delete_selected_preview_cut(self):
        """Keyboard shortcut: delete selected preview cut plane."""
        idx = self._active_preview_idx
        if idx < 0 or idx >= len(self._preview_planes): return
        self._preview_planes.pop(idx)
        self._active_preview_idx = min(idx, len(self._preview_planes) - 1)
        self._refresh_preview_list()
        self.viewport.set_preview_cuts(self._preview_planes, self._active_preview_idx)
        self.status_bar.showMessage(f"Deleted cut plane. {len(self._preview_planes)} remaining.")

    def _explode_changed(self, value):
        """Update viewport explode factor from slider."""
        self.viewport.explode_factor = value / 100.0
        self.viewport.update()

    def _snap_config_changed(self, idx):
        """Update viewport snap angle from combo."""
        snap_map = {0: 5.0, 1: 10.0, 2: 15.0, 3: 30.0, 4: 45.0, 5: 0.0}
        self.viewport.snap_step = snap_map.get(idx, 15.0)
        self.viewport.update()
