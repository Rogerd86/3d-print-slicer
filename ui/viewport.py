"""
Viewport3D — PyVista-based 3D viewport embedded in PyQt5.

Replaces the old OpenGL immediate-mode renderer with VTK/PyVista:
  - Proper shader-based rendering (handles millions of triangles)
  - Built-in orbit/pan/zoom that works correctly
  - Reliable mesh picking (click to select parts)
  - Cut plane widgets
  - No manual matrix math

Mouse controls (VTK standard):
  Left-click drag  = orbit (rotate view)
  Right-click drag = zoom
  Middle-click drag = pan
  Scroll wheel     = zoom
  Left-click       = select part / place marker
  Right-click (no drag) = context menu on part
"""
import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
from PyQt5.QtCore import Qt, pyqtSignal, QPoint
import colorsys


class Viewport3D(QWidget):
    """PyVista-based 3D viewport that matches the old OpenGL viewport's interface."""

    # Signals — same as old viewport
    cut_moved = pyqtSignal(float)
    cut_rotated = pyqtSignal(float, float, float)
    dowel_placed = pyqtSignal(object, object)
    measure_clicked = pyqtSignal(object, object)
    faces_selected = pyqtSignal(object)
    part_left_clicked = pyqtSignal(object)
    part_right_clicked = pyqtSignal(object)

    # 50 colours using golden-angle hue spacing
    @staticmethod
    def _generate_colors():
        golden = 0.618033988749895
        cols = []
        for i in range(50):
            h = (i * golden) % 1.0
            s = 0.55 + (i % 3) * 0.15
            v = 0.70 + (i % 2) * 0.15
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            cols.append((r, g, b))
        return cols
    COLORS = _generate_colors()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(400, 400)

        # Create PyVista plotter embedded in Qt
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plotter = QtInteractor(self)
        self.plotter.set_background('#1e2128')
        layout.addWidget(self.plotter.interactor)

        # Scene data
        self.parts = []
        self.all_parts = []
        self.selected_part_id = None
        self.mesh_bounds = None
        self.vertices = None
        self.faces = None

        # Cut planes
        self.preview_cuts = []
        self.preview_cut = None
        self.active_cut_idx = -1

        # Joint preview
        self.joint_preview_meshes = []

        # Display state
        self.show_wireframe = False
        self.show_cut_preview = True
        self.show_selection_wireframe = False
        self.show_grid = True
        self._heatmap_rgba = None
        self._crease_lines = []

        # Explode view
        self.explode_factor = 0.0
        self.orbit_centre = np.array([0.0, 0.0, 0.0])

        # Selection state
        self.selection_mode = False
        self.selection_brush_radius = 25.0
        self.selected_faces = set()
        self.manual_dowel_mode = False
        self.dowel_markers = []
        self.face_label_markers = []
        self.measure_points = []
        self.build_volume = None
        self.snap_step = 15.0
        self.snap_to_creases = True
        self.crease_snap_points = []
        self._hover_part_id = None
        self._measure_active = False

        # Actor tracking — maps part.id to pyvista actor
        self._part_actors = {}
        self._preview_actors = []
        self._joint_actors = []
        self._marker_actors = []
        self._grid_actor = None
        self._build_vol_actor = None

        # Track mouse for right-click detection
        self._rmb_start = None

        # Rotate-object + dowel-drag state
        self._rotate_obj_mode = False
        self._r_key_held = False
        self._obj_drag_active = False
        self._dowel_drag_active = False
        self._hud_actor = None

        # Setup initial scene
        self._setup_scene()

        # Install VTK event filter so we can intercept LMB-drag when in
        # rotate-object / dowel-drag modes.
        try:
            self._install_interaction_filter()
        except Exception:
            pass

        # Connect pick callback — use surface picking for reliable point coordinates
        try:
            self.plotter.enable_surface_point_picking(
                callback=self._on_pick, show_point=False, show_message=False)
        except (AttributeError, TypeError):
            # Fallback for older PyVista versions
            try:
                self.plotter.enable_point_picking(
                    callback=self._on_pick, show_point=False)
            except Exception:
                pass

    def _setup_scene(self):
        """Initial scene setup."""
        self.plotter.add_axes()
        if self.show_grid:
            self._draw_grid()

    def set_hud(self, text, color='#ffd56a'):
        """Show a corner text overlay with step-by-step instructions.
        Pass text=None (or empty) to clear."""
        if self._hud_actor is not None:
            try: self.plotter.remove_actor(self._hud_actor)
            except Exception: pass
            self._hud_actor = None
        if not text:
            try: self.plotter.render()
            except Exception: pass
            return
        try:
            self._hud_actor = self.plotter.add_text(
                text, position='upper_left', font_size=10,
                color=color, shadow=True, name='__hud__')
            self.plotter.render()
        except Exception:
            pass

    def _add_mesh(self, *args, **kwargs):
        """Wrapper for plotter.add_mesh that never resets the camera and
        defers rendering. Keeps camera put and batches renders into a single
        final call from _refresh_scene."""
        kwargs.setdefault('reset_camera', False)
        kwargs.setdefault('render', False)
        return self.plotter.add_mesh(*args, **kwargs)

    def _schedule_refresh(self, parts=False, cuts=False, joints=False,
                          markers=False, build_vol=False, all=False):
        """Mark sections dirty and queue a single coalesced rebuild pass.

        Instead of rebuilding the whole scene on every setter call, each
        section tracks its own dirty flag. A QTimer(0) merges every call in
        the current event-loop tick into ONE pass, and the pass only touches
        the sections that actually changed."""
        from PyQt5.QtCore import QTimer
        if all:
            parts = cuts = joints = markers = build_vol = True
        self._dirty_parts     = getattr(self, '_dirty_parts', False)     or parts
        self._dirty_cuts      = getattr(self, '_dirty_cuts', False)      or cuts
        self._dirty_joints    = getattr(self, '_dirty_joints', False)    or joints
        self._dirty_markers   = getattr(self, '_dirty_markers', False)   or markers
        self._dirty_build_vol = getattr(self, '_dirty_build_vol', False) or build_vol
        if getattr(self, '_refresh_timer', None) is None:
            self._refresh_timer = QTimer(self)
            self._refresh_timer.setSingleShot(True)
            self._refresh_timer.setInterval(0)
            self._refresh_timer.timeout.connect(self._apply_dirty_refresh)
        if not self._refresh_timer.isActive():
            self._refresh_timer.start()

    def _apply_dirty_refresh(self):
        """Run the coalesced refresh pass — only touches dirty sections."""
        touched = False
        if getattr(self, '_dirty_parts', False):
            self._rebuild_parts(); touched = True
        if getattr(self, '_dirty_cuts', False):
            self._rebuild_cuts(); touched = True
        if getattr(self, '_dirty_joints', False):
            self._rebuild_joints(); touched = True
        if getattr(self, '_dirty_markers', False):
            self._rebuild_markers(); touched = True
        if getattr(self, '_dirty_build_vol', False):
            self._draw_build_volume(); touched = True
        if self.show_grid and not self._grid_actor:
            self._draw_grid(); touched = True
        self._dirty_parts = self._dirty_cuts = False
        self._dirty_joints = self._dirty_markers = False
        self._dirty_build_vol = False
        if touched:
            try: self.plotter.render()
            except Exception: pass

    def _rebuild_parts(self):
        """Rebuild part actors only."""
        for actor in self._part_actors.values():
            try: self.plotter.remove_actor(actor)
            except Exception: pass
        self._part_actors.clear()
        if self.parts:
            self._draw_parts()
        elif self.vertices is not None:
            self._draw_simple_mesh()
        # Face-selection overlay sits on top of parts, so rebuild it too.
        self._rebuild_face_selection()

    def _rebuild_cuts(self):
        """Rebuild preview-cut actors only."""
        for actor in self._preview_actors:
            try: self.plotter.remove_actor(actor)
            except Exception: pass
        self._preview_actors.clear()
        if self.show_cut_preview and self.preview_cuts:
            self._draw_preview_cuts()

    def _rebuild_joints(self):
        """Rebuild joint preview actors only."""
        for actor in self._joint_actors:
            try: self.plotter.remove_actor(actor)
            except Exception: pass
        self._joint_actors.clear()
        if self.joint_preview_meshes:
            self._draw_joint_previews()

    def _rebuild_markers(self):
        """Rebuild dowel / measure / label markers only."""
        for actor in self._marker_actors:
            try: self.plotter.remove_actor(actor)
            except Exception: pass
        self._marker_actors.clear()
        if self.dowel_markers:
            self._draw_dowel_markers()
        if self.face_label_markers:
            self._draw_face_labels()
        if self.measure_points:
            self._draw_measure_points()

    def _rebuild_face_selection(self):
        try:
            self.plotter.remove_actor('face_selection')
        except Exception:
            pass
        if self.selected_faces:
            self._draw_face_selection()

    # ═══════════════════════════════════════════════════════
    # DRAWING
    # ═══════════════════════════════════════════════════════

    def _draw_grid(self):
        """Draw a ground plane grid at Z=0 (build plate)."""
        if self._grid_actor:
            try: self.plotter.remove_actor(self._grid_actor)
            except Exception: pass
            self._grid_actor = None
        try:
            # Grid at Z=0 — matches the build plate / ground plane
            grid = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1),
                            i_size=600, j_size=600, i_resolution=12, j_resolution=12)
            self._grid_actor = self._add_mesh(
                grid, color='#333844', style='wireframe',
                line_width=1, opacity=0.2, name='grid')
        except Exception:
            self._grid_actor = None

    def _draw_build_volume(self):
        """Draw or remove printer build volume wireframe box."""
        # Always try to remove the old one first
        if self._build_vol_actor:
            try:
                self.plotter.remove_actor(self._build_vol_actor)
            except Exception:
                pass
            self._build_vol_actor = None

        # Only draw if build_volume is set
        if not self.build_volume:
            return
        try:
            bx, by, bz = self.build_volume
            box = pv.Box(bounds=(-bx/2, bx/2, -by/2, by/2, 0, bz))
            self._build_vol_actor = self._add_mesh(
                box, color='#4080c0', style='wireframe',
                line_width=1.5, opacity=0.25, name='build_vol')
        except Exception:
            self._build_vol_actor = None

    def _refresh_scene(self):
        """Full scene refresh — kept for backward compat. Marks everything
        dirty and runs the coalesced pass synchronously."""
        self._dirty_parts = True
        self._dirty_cuts = True
        self._dirty_joints = True
        self._dirty_markers = True
        self._dirty_build_vol = True
        self._apply_dirty_refresh()

    def _draw_simple_mesh(self):
        """Draw the root mesh (before slicing)."""
        if self.vertices is None or self.faces is None:
            return
        # Build pyvista mesh from vertices/faces
        n_faces = len(self.faces)
        pv_faces = np.column_stack([
            np.full(n_faces, 3, dtype=np.int32),
            self.faces.astype(np.int32)
        ]).flatten()
        mesh = pv.PolyData(self.vertices.astype(np.float64), pv_faces)

        if self._heatmap_rgba is not None and len(self._heatmap_rgba) == len(self.vertices):
            # Heatmap overlay
            mesh.point_data['colors'] = (self._heatmap_rgba[:, :3] * 255).astype(np.uint8)
            actor = self._add_mesh(mesh, scalars='colors', rgb=True,
                                          name='root_mesh', show_edges=self.show_wireframe)
        else:
            style = 'wireframe' if self.show_wireframe else 'surface'
            actor = self._add_mesh(mesh, color='#66bbee',
                                          style=style, name='root_mesh')
        self._part_actors['root'] = actor

    def _draw_parts(self):
        """Draw all leaf parts with distinct colours."""
        model_centroid = np.zeros(3)
        if self.explode_factor > 0.001 and self.mesh_bounds is not None:
            model_centroid = (self.mesh_bounds[0] + self.mesh_bounds[1]) / 2.0
            scene_size = float(np.max(self.mesh_bounds[1] - self.mesh_bounds[0]))
            explode_dist = self.explode_factor * scene_size * 0.5

        for part in self.parts:
            if hasattr(part, 'visible') and not part.visible:
                continue
            mesh = part.mesh
            if not len(mesh.faces):
                continue

            # Convert trimesh to pyvista
            n_faces = len(mesh.faces)
            pv_faces = np.column_stack([
                np.full(n_faces, 3, dtype=np.int32),
                mesh.faces.astype(np.int32)
            ]).flatten()
            verts = mesh.vertices.astype(np.float64).copy()

            # Apply explode offset
            if self.explode_factor > 0.001 and self.mesh_bounds is not None:
                part_centroid = (mesh.bounds[0] + mesh.bounds[1]) / 2.0
                direction = part_centroid - model_centroid
                dn = np.linalg.norm(direction)
                if dn > 0.001:
                    direction /= dn
                verts += direction * explode_dist

            pv_mesh = pv.PolyData(verts, pv_faces)

            # Colour
            is_sel = (part.id == self.selected_part_id)
            is_wf = getattr(part, '_wireframe', False)
            if hasattr(part, '_source_colour') and part._source_colour:
                r, g, b = part._source_colour
                color = (int(r*255), int(g*255), int(b*255))
            else:
                c = self.COLORS[part.color_idx % len(self.COLORS)]
                color = (int(c[0]*255), int(c[1]*255), int(c[2]*255))

            style = 'wireframe' if is_wf else 'surface'
            show_edges = is_wf or (is_sel and self.show_selection_wireframe)
            edge_color = '#ffdd33' if is_sel else None
            opacity = 1.0

            actor = self._add_mesh(
                pv_mesh, color=color, style=style,
                show_edges=show_edges, edge_color=edge_color,
                line_width=1.5 if show_edges else 1.0,
                opacity=opacity, name=f'part_{part.id}')
            self._part_actors[part.id] = actor

    def _draw_preview_cuts(self):
        """Draw cut plane previews.

        Plane sizing is generous (2× the max extent) so a Full cut visibly
        spans the whole object. Plane centre is anchored to the MODEL
        centre and then slid along the cut normal by the requested offset —
        this way the plane covers the object even when the model sits on
        the ground (Z=0) rather than around the origin."""
        if not self.preview_cuts or self.mesh_bounds is None:
            return
        mins, maxs = np.asarray(self.mesh_bounds[0]), np.asarray(self.mesh_bounds[1])
        model_centre = (mins + maxs) * 0.5
        extent = float(np.max(maxs - mins))
        plane_side = extent * 1.4  # 40% margin past the widest side

        for i, cut in enumerate(self.preview_cuts):
            is_active = (i == self.active_cut_idx)
            normal = np.asarray(cut.get_normal(), dtype=float)
            # How far along the normal from the model centre is the cut?
            cut_origin = np.asarray(cut.get_origin(), dtype=float)
            offset_along_n = float(np.dot(cut_origin - model_centre, normal))
            # Anchor the plane at model_centre + offset * normal so it
            # straddles the model regardless of where the model sits.
            plane_centre = model_centre + normal * offset_along_n

            plane = pv.Plane(center=plane_centre, direction=normal,
                            i_size=plane_side, j_size=plane_side)

            if is_active:
                color = '#ff8800'
                opacity = 0.25
            elif getattr(cut, 'pinned', False):
                color = '#f0c040'
                opacity = 0.15
            else:
                color = '#5588ee'
                opacity = 0.10

            actor = self._add_mesh(
                plane, color=color, opacity=opacity,
                show_edges=True, edge_color=color,
                line_width=2.0 if is_active else 1.0,
                name=f'cut_{i}')
            self._preview_actors.append(actor)

    def _draw_joint_previews(self):
        """Draw joint preview meshes as cyan wireframes."""
        for j, mesh in enumerate(self.joint_preview_meshes):
            try:
                n_faces = len(mesh.faces)
                pv_faces = np.column_stack([
                    np.full(n_faces, 3, dtype=np.int32),
                    mesh.faces.astype(np.int32)
                ]).flatten()
                pv_mesh = pv.PolyData(mesh.vertices.astype(np.float64), pv_faces)
                actor = self._add_mesh(
                    pv_mesh, color='#18e8cc', style='wireframe',
                    line_width=1.5, opacity=0.7, name=f'joint_{j}')
                self._joint_actors.append(actor)
            except Exception:
                pass

    def _draw_dowel_markers(self):
        """Draw manual dowel placement markers."""
        for k, (pos, normal) in enumerate(self.dowel_markers):
            sphere = pv.Sphere(radius=3.0, center=pos)
            actor = self._add_mesh(
                sphere, color='#00eedd', opacity=0.8, name=f'dowel_{k}')
            self._marker_actors.append(actor)

    def _draw_face_labels(self):
        """Draw face label text markers."""
        label_colors = {
            'A': '#ff6666', 'B': '#66ff66', 'C': '#6699ff',
            'D': '#ffcc33', 'E': '#ff66cc', 'F': '#66ffee',
        }
        for k, (centre, normal, text) in enumerate(self.face_label_markers):
            letter = text[0].upper() if text else 'A'
            color = label_colors.get(letter, '#cccccc')
            sphere = pv.Sphere(radius=4.0, center=centre)
            actor = self._add_mesh(
                sphere, color=color, opacity=0.8, name=f'label_{k}')
            self._marker_actors.append(actor)
            # Add text label
            try:
                self.plotter.add_point_labels(
                    [centre], [text], font_size=14, text_color=color,
                    point_color=color, point_size=0, shape=None,
                    name=f'label_text_{k}', reset_camera=False, render=False)
            except TypeError:
                # Older pyvista versions without reset_camera/render kwargs
                self.plotter.add_point_labels(
                    [centre], [text], font_size=14, text_color=color,
                    point_color=color, point_size=0, shape=None,
                    name=f'label_text_{k}')

    def _draw_measure_points(self):
        """Draw measurement points and line."""
        for k, pt in enumerate(self.measure_points):
            sphere = pv.Sphere(radius=3.0, center=pt)
            actor = self._add_mesh(
                sphere, color='#ff3333', opacity=0.9, name=f'measure_{k}')
            self._marker_actors.append(actor)
        if len(self.measure_points) >= 2:
            line = pv.Line(self.measure_points[0], self.measure_points[1])
            actor = self._add_mesh(
                line, color='#ffff33', line_width=3.0, name='measure_line')
            self._marker_actors.append(actor)

    def _draw_face_selection(self):
        """Draw selected faces as orange overlay."""
        # Find the mesh to highlight on
        mesh = None
        if self.parts:
            sel_part = next((p for p in self.parts if p.id == self.selected_part_id), None)
            if sel_part:
                mesh = sel_part.mesh
        elif self.vertices is not None and self.faces is not None:
            import trimesh
            mesh = trimesh.Trimesh(vertices=self.vertices, faces=self.faces)

        if mesh is None or not self.selected_faces:
            return

        sel_list = [fi for fi in self.selected_faces if 0 <= fi < len(mesh.faces)]
        if not sel_list:
            return

        sel_faces = mesh.faces[sel_list]
        n_sel = len(sel_faces)
        pv_faces = np.column_stack([
            np.full(n_sel, 3, dtype=np.int32),
            sel_faces.astype(np.int32)
        ]).flatten()
        sel_mesh = pv.PolyData(mesh.vertices.astype(np.float64), pv_faces)
        actor = self._add_mesh(
            sel_mesh, color='#ff8800', opacity=0.4,
            show_edges=True, edge_color='#ff9933',
            line_width=1.5, name='face_selection')
        self._marker_actors.append(actor)

    # ═══════════════════════════════════════════════════════
    # PICKING / INTERACTION
    # ═══════════════════════════════════════════════════════

    def _on_pick(self, *args):
        """Callback when user clicks a point on a mesh.
        PyVista passes different args depending on picker type and version."""
        # Extract the point from whatever args PyVista gives us
        point = None
        for arg in args:
            if arg is None:
                continue
            try:
                arr = np.asarray(arg, dtype=float)
                if arr.shape == (3,):
                    point = arr
                    break
                elif arr.ndim == 2 and arr.shape[1] == 3 and len(arr) > 0:
                    point = arr[0]
                    break
            except (TypeError, ValueError):
                continue
        if point is None:
            return
        pos = np.array(point, dtype=float)

        # Find which part was clicked — use simple distance to face centres
        best_part = None
        best_dist = float('inf')
        best_normal = np.array([0, 0, 1.0])

        for part in self.parts:
            if hasattr(part, 'visible') and not part.visible:
                continue
            try:
                mesh = part.mesh
                if len(mesh.faces) == 0:
                    continue
                # Find nearest face centre
                face_centres = mesh.triangles_center
                dists = np.linalg.norm(face_centres - pos, axis=1)
                fi = int(np.argmin(dists))
                dist = float(dists[fi])
                if dist < best_dist:
                    best_dist = dist
                    best_part = part
                    best_normal = mesh.face_normals[fi].copy()
            except Exception as e:
                print(f"Pick error on {getattr(part, 'label', '?')}: {e}")

        # For manual dowel mode, also try the selected part specifically
        # (it might not be in self.parts if it's the root before slicing)
        if self.manual_dowel_mode and best_part is None and self.selected_part_id:
            for part in self.all_parts:
                if part.id == self.selected_part_id:
                    try:
                        face_centres = part.mesh.triangles_center
                        dists = np.linalg.norm(face_centres - pos, axis=1)
                        fi = int(np.argmin(dists))
                        best_part = part
                        best_normal = part.mesh.face_normals[fi].copy()
                        best_dist = float(dists[fi])
                    except Exception:
                        pass
                    break

        # Handle different modes
        if self._measure_active:
            self.measure_clicked.emit(pos, best_normal)
        elif self.manual_dowel_mode:
            # In dowel mode, emit regardless of which part (handler checks selected)
            if best_dist < 100:  # reasonable click distance
                self.dowel_placed.emit(pos, best_normal)
        elif self.selection_mode and best_part and best_part.id == self.selected_part_id:
            from core.region_repair import select_faces_near_point
            new_sel = select_faces_near_point(
                best_part.mesh, pos, self.selection_brush_radius)
            self.selected_faces |= new_sel
            self.faces_selected.emit(self.selected_faces)
            # Face-selection overlay is a single actor — no need to rebuild parts.
            self._rebuild_face_selection()
            try: self.plotter.render()
            except Exception: pass
        elif best_part:
            self.part_left_clicked.emit(best_part)

    # ═══════════════════════════════════════════════════════
    # ROTATE-OBJECT MODE (hold R or toggle button)
    # ═══════════════════════════════════════════════════════

    def set_rotate_object_mode(self, on):
        """When True, LMB drag rotates the selected part around its own
        centroid instead of orbiting the camera."""
        self._rotate_obj_mode = bool(on)
        self._install_interaction_filter()

    def _install_interaction_filter(self):
        if getattr(self, '_interaction_filter_installed', False):
            return
        try:
            self.plotter.interactor.installEventFilter(self)
            # Track key state too — R key = momentary rotate-object mode.
            self.plotter.interactor.setFocusPolicy(Qt.StrongFocus)
            self._interaction_filter_installed = True
        except Exception:
            pass

    def _reset_all_modes(self):
        """Nuclear option — clear every sticky interaction mode and any
        in-flight drag state. Wired to ESC + an explicit UI button."""
        self._rotate_obj_mode = False
        self._r_key_held = False
        self.selection_mode = False
        self.manual_dowel_mode = False
        self._measure_active = False
        self._dowel_drag_active = False
        self._obj_drag_active = False
        self._pick_press_pos = None
        try:
            self.setCursor(Qt.ArrowCursor)
        except Exception:
            pass
        self.set_hud(None)

    def _rotate_active(self):
        """Rotate-object mode is active if either the toggle is on OR the
        R key is currently held."""
        return (getattr(self, '_rotate_obj_mode', False)
                or getattr(self, '_r_key_held', False))

    def _get_selected_part(self):
        if self.selected_part_id is None:
            return None
        for p in self.parts:
            if p.id == self.selected_part_id:
                return p
        return None

    def _start_obj_rotation(self, qpos):
        part = self._get_selected_part()
        if part is None or not hasattr(part, 'mesh'):
            return False
        actor = self._part_actors.get(part.id)
        if actor is None:
            return False
        import vtk
        self._obj_drag_active = True
        self._obj_drag_part = part
        self._obj_drag_actor = actor
        self._obj_drag_last = (qpos.x(), qpos.y())
        # Identity cumulative rotation — applied as the actor's user transform.
        self._obj_drag_matrix = np.eye(4)
        # Cache centroid for rotation pivot.
        b = part.mesh.bounds
        self._obj_drag_pivot = np.array([
            (b[0][0] + b[1][0]) * 0.5,
            (b[0][1] + b[1][1]) * 0.5,
            (b[0][2] + b[1][2]) * 0.5,
        ])
        return True

    def _update_obj_rotation(self, qpos):
        if not getattr(self, '_obj_drag_active', False):
            return
        lx, ly = self._obj_drag_last
        dx = qpos.x() - lx
        dy = qpos.y() - ly
        self._obj_drag_last = (qpos.x(), qpos.y())
        # Rotation sensitivity: 0.4°/pixel feels natural.
        deg_per_px = 0.4
        # Horizontal = rotate around world Z.
        # Vertical   = rotate around camera right vector (screen-aligned).
        try:
            cam = self.plotter.camera
            view_up = np.array(cam.up, dtype=float)
            view_dir = np.array(cam.focal_point, dtype=float) - np.array(cam.position, dtype=float)
            nd = np.linalg.norm(view_dir)
            if nd > 1e-6:
                view_dir /= nd
            right = np.cross(view_dir, view_up)
            nr = np.linalg.norm(right)
            if nr > 1e-6:
                right /= nr
            else:
                right = np.array([1.0, 0.0, 0.0])
        except Exception:
            right = np.array([1.0, 0.0, 0.0])

        ang_z = np.deg2rad(-dx * deg_per_px)   # horizontal drag → yaw
        ang_r = np.deg2rad(-dy * deg_per_px)   # vertical drag → pitch
        Rz = self._axis_angle_matrix(np.array([0, 0, 1.0]), ang_z)
        Rr = self._axis_angle_matrix(right, ang_r)
        delta = Rz @ Rr
        # Pre-multiply so screen axes stay consistent during sustained drag.
        self._obj_drag_matrix = delta @ self._obj_drag_matrix
        self._apply_drag_user_transform()

    def _axis_angle_matrix(self, axis, angle):
        """Build a 4x4 rotation matrix around `axis` by `angle` (radians)
        that pivots around self._obj_drag_pivot."""
        ax = axis / (np.linalg.norm(axis) + 1e-12)
        c = np.cos(angle); s = np.sin(angle); t = 1 - c
        x, y, z = ax
        R3 = np.array([
            [t*x*x + c,   t*x*y - s*z, t*x*z + s*y],
            [t*x*y + s*z, t*y*y + c,   t*y*z - s*x],
            [t*x*z - s*y, t*y*z + s*x, t*z*z + c],
        ])
        p = self._obj_drag_pivot
        M = np.eye(4)
        M[:3, :3] = R3
        M[:3, 3] = p - R3 @ p
        return M

    def _apply_drag_user_transform(self):
        """Push the current drag matrix onto the actor's UserTransform — GPU
        rotation, no mesh rebuild, smooth to the eye."""
        import vtk
        vm = vtk.vtkMatrix4x4()
        for i in range(4):
            for j in range(4):
                vm.SetElement(i, j, float(self._obj_drag_matrix[i, j]))
        tr = vtk.vtkTransform()
        tr.SetMatrix(vm)
        try:
            self._obj_drag_actor.SetUserTransform(tr)
            self.plotter.render()
        except Exception:
            pass

    def _finish_obj_rotation(self):
        """Commit the drag transform to the trimesh so it sticks, then reset
        the actor's UserTransform and do a single lightweight re-draw."""
        if not getattr(self, '_obj_drag_active', False):
            return
        part = self._obj_drag_part
        M = self._obj_drag_matrix.copy()
        # Clear drag state FIRST so any follow-up events go through cleanly.
        self._obj_drag_active = False
        try:
            import vtk
            tr = vtk.vtkTransform()
            tr.Identity()
            self._obj_drag_actor.SetUserTransform(tr)
        except Exception:
            pass
        # Bake the transform into the mesh vertices.
        try:
            part.mesh.apply_transform(M)
        except Exception:
            pass
        # Record cumulative rotation on the part so "Reset Orientation" can
        # undo it. Stack with any previous rotation.
        try:
            prev = getattr(part, '_user_rotation', None)
            if prev is None:
                part._user_rotation = M.copy()
            else:
                part._user_rotation = M @ np.asarray(prev, dtype=float)
        except Exception:
            pass
        # Only this one part needs redrawing.
        self._schedule_refresh(parts=True)

    # ═══════════════════════════════════════════════════════
    # EVENT FILTER — intercepts VTK mouse when rotate-object is active
    # ═══════════════════════════════════════════════════════

    def eventFilter(self, obj, ev):
        from PyQt5.QtCore import QEvent
        try:
            interactor = self.plotter.interactor
        except Exception:
            return super().eventFilter(obj, ev)
        if obj is not interactor:
            return super().eventFilter(obj, ev)

        t = ev.type()
        # Track R key for momentary rotate mode (press-and-hold).
        if t == QEvent.KeyPress and ev.key() == Qt.Key_R and not ev.isAutoRepeat():
            self._r_key_held = True
            return False
        if t == QEvent.KeyRelease and ev.key() == Qt.Key_R and not ev.isAutoRepeat():
            self._r_key_held = False
            return False
        # ESC unconditionally clears every sticky viewport mode. This is the
        # big "I'm stuck — get me out" button — without it the user can end
        # up trapped in rotate-object / dowel / selection mode if a key
        # release event is missed (e.g. alt-tab while R is held).
        if t == QEvent.KeyPress and ev.key() == Qt.Key_Escape:
            self._reset_all_modes()
            return False
        # Safety net: any time the interactor LOSES FOCUS, assume held keys
        # are released. This is what prevents "R stays True forever" after
        # the user alt-tabs with R held, then comes back and finds the
        # whole scene following their cursor.
        if t == QEvent.FocusOut:
            self._r_key_held = False
            if getattr(self, '_obj_drag_active', False):
                try: self._finish_obj_rotation()
                except Exception: self._obj_drag_active = False
            self._dowel_drag_active = False

        if self._rotate_active():
            if t == QEvent.MouseButtonPress and ev.button() == Qt.LeftButton:
                if self._start_obj_rotation(ev.pos()):
                    return True  # swallow → VTK doesn't orbit
            elif t == QEvent.MouseMove and getattr(self, '_obj_drag_active', False):
                self._update_obj_rotation(ev.pos())
                return True
            elif t == QEvent.MouseButtonRelease and ev.button() == Qt.LeftButton:
                if getattr(self, '_obj_drag_active', False):
                    self._finish_obj_rotation()
                    return True

        # ─── LMB click-to-pick in selection / dowel / measure modes ───
        # PyVista's point picker is bound to the 'P' key by default, which
        # is a terrible UX default. We intercept LMB press in these modes,
        # raycast the cursor → world position, call _on_pick ourselves,
        # and swallow the event so VTK's interactor doesn't also orbit.
        in_pick_mode = (
            getattr(self, 'selection_mode', False) or
            getattr(self, 'manual_dowel_mode', False) or
            getattr(self, '_measure_active', False)
        )
        if in_pick_mode:
            if t == QEvent.MouseButtonPress and ev.button() == Qt.LeftButton:
                # Dowel drag starts here — if we press on a marker, let the
                # existing dowel-drag path handle it and swallow.
                if getattr(self, 'manual_dowel_mode', False):
                    self._maybe_start_dowel_drag(ev.pos())
                    if getattr(self, '_dowel_drag_active', False):
                        return True
                # Remember press pos — we only want to PICK on clean clicks,
                # not at the end of a drag.
                self._pick_press_pos = (ev.pos().x(), ev.pos().y())
                return True  # swallow press so VTK doesn't start orbiting
            elif t == QEvent.MouseMove and getattr(self, '_dowel_drag_active', False):
                self._update_dowel_drag(ev.pos())
                return True
            elif t == QEvent.MouseButtonRelease and ev.button() == Qt.LeftButton:
                if getattr(self, '_dowel_drag_active', False):
                    self._dowel_drag_active = False
                    return True
                # Commit as a click only if movement was small (not a drag).
                pp = getattr(self, '_pick_press_pos', None)
                if pp is not None:
                    dx = abs(ev.pos().x() - pp[0])
                    dy = abs(ev.pos().y() - pp[1])
                    self._pick_press_pos = None
                    if dx < 5 and dy < 5:
                        self._do_click_pick(ev.pos())
                return True  # swallow — VTK shouldn't treat it as orbit

        return super().eventFilter(obj, ev)

    def _do_click_pick(self, qpos):
        """Raycast the viewport at qpos, find the world point on the nearest
        surface, and feed it into _on_pick so the existing mode handling
        (selection brush / dowel place / measure) runs."""
        try:
            import vtk
            picker = vtk.vtkCellPicker()
            picker.SetTolerance(0.0005)
            ren = self.plotter.renderer
            h = ren.GetSize()[1]
            picker.Pick(qpos.x(), h - qpos.y(), 0, ren)
            if picker.GetCellId() < 0:
                return
            hit = picker.GetPickPosition()
            self._on_pick(hit)
        except Exception as e:
            print(f"Click-pick failed: {e}")

    def _update_dowel_drag(self, qpos):
        """Raycast the current cursor position onto the part the marker is
        attached to and move the marker there. Keeps it glued to the
        surface, which is what the user expects."""
        idx = getattr(self, '_dowel_drag_idx', None)
        part = getattr(self, '_dowel_drag_part', None)
        if idx is None or part is None or idx >= len(self.dowel_markers):
            return
        # Project screen → world using VTK's picker on this part's actor only.
        try:
            import vtk
            picker = vtk.vtkCellPicker()
            picker.SetTolerance(0.005)
            actor = self._part_actors.get(part.id)
            if actor is None: return
            picker.AddPickList(actor)
            picker.PickFromListOn()
            ren = self.plotter.renderer
            h = ren.GetSize()[1]
            picker.Pick(qpos.x(), h - qpos.y(), 0, ren)
            if picker.GetCellId() < 0:
                return
            hit = np.array(picker.GetPickPosition(), dtype=float)
            # Re-snap to the nearest face centre + use that face's normal.
            centres = part.mesh.triangles_center
            fi = int(np.argmin(np.linalg.norm(centres - hit, axis=1)))
            normal = part.mesh.face_normals[fi].copy()
            self.dowel_markers[idx] = (hit, normal)
            self._schedule_refresh(markers=True)
        except Exception:
            pass

    def mousePressEvent(self, e):
        """Track right-click start for context menu detection, and LMB
        presses on dowel markers for drag-to-move."""
        if e.button() == Qt.RightButton:
            self._rmb_start = e.pos()
        elif e.button() == Qt.LeftButton and getattr(self, 'manual_dowel_mode', False):
            self._maybe_start_dowel_drag(e.pos())
        super().mousePressEvent(e)

    def _maybe_start_dowel_drag(self, qpos):
        """If the click landed on an existing dowel marker, begin a drag."""
        if not self.dowel_markers:
            return
        try:
            import vtk
            picker = vtk.vtkPropPicker()
            ren = self.plotter.renderer
            h = ren.GetSize()[1]
            picker.Pick(qpos.x(), h - qpos.y(), 0, ren)
            hit_actor = picker.GetActor()
            if hit_actor is None:
                return
            # Is hit_actor one of our dowel markers?
            for k, (pos, _n) in enumerate(self.dowel_markers):
                # We don't cache per-marker actors individually by id, so
                # fall back to screen-space proximity to known marker pos.
                # Any marker within 15 world units of the pick position
                # while in dowel mode = start drag.
                pick_pos = np.array(picker.GetPickPosition(), dtype=float)
                if np.linalg.norm(pick_pos - pos) < 15.0:
                    # Find which part the marker lives on (nearest part).
                    best_part, best_d = None, float('inf')
                    for p in self.parts:
                        if not hasattr(p, 'mesh'): continue
                        centres = p.mesh.triangles_center
                        d = float(np.min(np.linalg.norm(centres - pos, axis=1)))
                        if d < best_d:
                            best_d, best_part = d, p
                    if best_part is not None:
                        self._dowel_drag_active = True
                        self._dowel_drag_idx = k
                        self._dowel_drag_part = best_part
                    return
        except Exception:
            pass

    def mouseReleaseEvent(self, e):
        """Detect short right-click for part context menu. Uses a cheap
        face-centre distance test (same as _on_pick) instead of
        mesh.nearest.vertex, which is O(n) per part and was the main cause
        of right-click lag."""
        if e.button() == Qt.RightButton and self._rmb_start is not None:
            dx = abs(e.x() - self._rmb_start.x())
            dy = abs(e.y() - self._rmb_start.y())
            if dx < 5 and dy < 5:
                try:
                    picked = self.plotter.pick_click_position()
                except Exception:
                    picked = None
                if picked is not None:
                    pos = np.array(picked, dtype=float)
                    best_part = None
                    best_dist = float('inf')
                    for part in self.parts:
                        if hasattr(part, 'visible') and not part.visible:
                            continue
                        mesh = getattr(part, 'mesh', None)
                        if mesh is None or len(mesh.faces) == 0:
                            continue
                        try:
                            # Broad-phase: bbox distance is ~free
                            b = mesh.bounds
                            cx = (b[0][0] + b[1][0]) * 0.5
                            cy = (b[0][1] + b[1][1]) * 0.5
                            cz = (b[0][2] + b[1][2]) * 0.5
                            bb_dist = np.linalg.norm(np.array([cx, cy, cz]) - pos)
                            if bb_dist > best_dist + float(np.max(b[1] - b[0])):
                                continue
                            # Narrow-phase: nearest face centre
                            centres = mesh.triangles_center
                            d = float(np.min(np.linalg.norm(centres - pos, axis=1)))
                            if d < best_dist:
                                best_dist = d
                                best_part = part
                        except Exception:
                            continue
                    if best_part is not None and best_dist < 50:
                        self.part_right_clicked.emit(best_part)
        self._rmb_start = None
        super().mouseReleaseEvent(e)

    # ═══════════════════════════════════════════════════════
    # DATA SETTERS (same interface as old viewport)
    # ═══════════════════════════════════════════════════════

    def set_mesh(self, vertices, faces, bounds=None):
        # Loading a (new) root mesh — always fit the camera to the model.
        self.vertices = vertices
        self.faces = faces
        self.parts = []; self.all_parts = []
        self.mesh_bounds = bounds
        self._part_actors.clear()
        if bounds is not None:
            self.orbit_centre = ((bounds[0] + bounds[1]) / 2.0).copy()
        self._refresh_scene()
        if bounds is not None:
            try: self.plotter.reset_camera()
            except Exception: pass

    def set_parts(self, parts, all_parts=None, bounds=None):
        first_load = (not self.parts and self.vertices is None)
        self.parts = parts
        self.all_parts = all_parts or parts
        self.vertices = None; self.faces = None
        if bounds is not None:
            self.mesh_bounds = bounds
        # Only the part section + any cut/joint previews that depend on bounds.
        self._schedule_refresh(parts=True, cuts=True)
        if first_load and bounds is not None:
            try: self.plotter.reset_camera()
            except Exception: pass

    def set_selected_part(self, part_id):
        """Lightweight selection update — only toggles the edge highlight on
        the affected actors instead of rebuilding the whole scene."""
        if self.selected_part_id == part_id:
            return
        old_id = self.selected_part_id
        self.selected_part_id = part_id

        # If we haven't drawn parts yet, nothing to update visually.
        if not self._part_actors:
            return

        want_edges = bool(self.show_selection_wireframe)
        for pid in (old_id, part_id):
            if pid is None: continue
            actor = self._part_actors.get(pid)
            if actor is None: continue
            try:
                prop = actor.GetProperty()
                is_sel = (pid == self.selected_part_id)
                # Preserve wireframe-mode parts
                part = next((p for p in self.parts if p.id == pid), None)
                is_wf = bool(getattr(part, '_wireframe', False)) if part else False
                show_edges = is_wf or (is_sel and want_edges)
                prop.SetEdgeVisibility(1 if show_edges else 0)
                if is_sel and want_edges:
                    prop.SetEdgeColor(1.0, 0.867, 0.2)  # #ffdd33
                prop.SetLineWidth(1.5 if show_edges else 1.0)
            except Exception:
                pass
        try:
            self.plotter.render()
        except Exception:
            pass

    def set_preview_cuts(self, cuts, active_idx=-1):
        self.preview_cuts = cuts
        self.active_cut_idx = active_idx
        self.preview_cut = cuts[active_idx] if 0 <= active_idx < len(cuts) else None
        self._schedule_refresh(cuts=True)

    def set_preview_cut(self, cut):
        self.preview_cut = cut
        if cut is not None and cut not in self.preview_cuts:
            self.preview_cuts = [cut]
            self.active_cut_idx = 0
        self._schedule_refresh(cuts=True)

    def set_joint_preview(self, meshes):
        self.joint_preview_meshes = meshes if meshes else []
        self._schedule_refresh(joints=True)

    def set_heatmap(self, scores_rgba):
        self._heatmap_rgba = scores_rgba
        # Heatmap lives on the part actors themselves → need part rebuild.
        self._schedule_refresh(parts=True)

    def set_crease_lines(self, seam_list):
        self._crease_lines = seam_list
        self._schedule_refresh(markers=True)

    def clear_heatmap(self):
        self._heatmap_rgba = None
        self._crease_lines = []
        self._schedule_refresh(parts=True, markers=True)

    def set_cut_planes(self, planes):
        self.preview_cuts = planes if planes else []
        self._schedule_refresh(cuts=True)

    def clear(self):
        self.vertices = None; self.faces = None
        self.parts = []; self.all_parts = []
        self.preview_cut = None; self.preview_cuts = []
        self.joint_preview_meshes = []
        self._heatmap_rgba = None; self._crease_lines = []
        self.face_label_markers = []; self.dowel_markers = []
        self.measure_points = []; self.selected_faces = set()
        self.mesh_bounds = None
        self._part_actors.clear()
        self._preview_actors.clear()
        self._joint_actors.clear()
        self._marker_actors.clear()
        self._grid_actor = None
        self._build_vol_actor = None
        try:
            self.plotter.clear()
            self._setup_scene()
        except Exception:
            pass

    def update(self):
        """Refresh everything — coalesced. Callers that know which section
        changed should use _schedule_refresh(section=True) directly."""
        self._schedule_refresh(all=True)

    def render_only(self):
        """Lightweight re-render without rebuilding actors."""
        try:
            self.plotter.render()
        except Exception:
            pass

    def keyPressEvent(self, e):
        """Only forwards to super when `e` is a real QKeyEvent.

        The R key (momentary rotate-object) and mouse interactions are
        handled by the VTK-interactor event filter, so this path is rarely
        triggered. Kept defensively because callers sometimes pass synth
        objects; in that case we just swallow rather than crash."""
        try:
            from PyQt5.QtGui import QKeyEvent
            if isinstance(e, QKeyEvent):
                if e.key() == Qt.Key_F:
                    # Focus on selected part
                    for part in self.parts:
                        if part.id == self.selected_part_id:
                            try:
                                bounds = part.mesh.bounds
                                centre = (bounds[0] + bounds[1]) / 2.0
                                self.plotter.camera.focal_point = centre
                                self.plotter.reset_camera()
                            except Exception:
                                pass
                            break
                super().keyPressEvent(e)
        except Exception:
            pass

    def setCursor(self, cursor):
        """Forward cursor changes to the plotter widget."""
        try:
            self.plotter.interactor.setCursor(cursor)
        except Exception:
            super().setCursor(cursor)
