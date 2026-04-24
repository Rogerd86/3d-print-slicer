"""
Viewport3D — OpenGL 3D view with cut plane gizmo.

Mouse controls:
  Right-click drag  = orbit (rotate view)
  Middle-click drag = pan
  Scroll wheel      = zoom
  Left-click        = select gizmo handle / part

Gizmo handles on active cut plane:
  Yellow arrow  = drag to translate along cut normal
  Red ring      = drag to rotate around X
  Green ring    = drag to rotate around Y
  Blue ring     = drag to rotate around Z
"""
import numpy as np
from PyQt5.QtWidgets import QOpenGLWidget, QSizePolicy
from PyQt5.QtCore import Qt, QPoint, pyqtSignal
from PyQt5.QtGui import QColor, QCursor

try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    HAS_OPENGL = True
except ImportError:
    HAS_OPENGL = False

HIT_NONE   = 0
HIT_ARROW  = 1
HIT_RING_X = 2
HIT_RING_Y = 3
HIT_RING_Z = 4


class Viewport3D(QOpenGLWidget):
    cut_moved   = pyqtSignal(float)
    cut_rotated = pyqtSignal(float, float, float)
    dowel_placed = pyqtSignal(object, object)  # position(3,), normal(3,)
    measure_clicked = pyqtSignal(object, object)  # position(3,), normal(3,)
    faces_selected = pyqtSignal(object)  # set of face indices
    part_right_clicked = pyqtSignal(object)  # part object (right-click context menu)
    part_left_clicked = pyqtSignal(object)   # part object (left-click select)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(400, 400)

        # Camera
        self.rot_x = 25.0
        self.rot_y = -35.0
        self.zoom  = -600.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        # Orbit centre — camera rotates around this point
        self.orbit_centre = np.array([0.0, 0.0, 0.0])

        # Mouse state
        self._last_pos   = QPoint()
        self._lmb_down   = False   # left mouse = gizmo
        self._rmb_down   = False   # right mouse = orbit
        self._mmb_down   = False   # middle = pan
        self._gizmo_drag = HIT_NONE

        # Scene data
        self.parts     = []
        self.all_parts = []
        self.selected_part_id = None
        self.mesh_bounds = None
        self.vertices = None
        self.faces    = None

        # Cut planes
        self.preview_cuts   = []
        self.preview_cut    = None
        self.active_cut_idx = -1

        # Joint preview geometry (list of trimesh meshes to draw as wireframe)
        self.joint_preview_meshes = []

        # Gizmo state
        self.gizmo_hit  = HIT_NONE
        self.show_wireframe   = False
        self.show_cut_preview = True
        self.show_grid        = True
        self.show_selection_wireframe = False  # yellow wireframe on selected part
        self._heatmap_rgba    = None    # per-vertex (N,4) float32 or None
        self._crease_lines    = []      # list of seam dicts

        # Explode view
        self.explode_factor = 0.0  # 0.0 = assembled, 1.0 = fully exploded

        # Manual dowel placement
        self.manual_dowel_mode = False
        self.dowel_markers = []  # list of (position_xyz, normal_xyz)

        # Face label preview markers: list of (centre, normal, label_text)
        self.face_label_markers = []

        # Measurement points: list of np.array positions (0, 1 or 2)
        self.measure_points = []

        # Build volume outline: (x, y, z) in mm, or None to hide
        self.build_volume = None

        # Hover highlight
        self._hover_part_id = None

        # Face selection for region-based repair
        self.selection_mode = False    # True = brush selection active
        self.selection_brush_radius = 25.0  # mm — larger default for easier selection
        self.selected_faces = set()   # set of face indices
        self._sel_vbo = None          # cached selection overlay

        # Crease snap data: list of {axis, position, midpoint}
        self.crease_snap_points = []
        self.snap_to_creases = True
        self._snapped_crease = None  # currently snapped crease for highlight

        # Configurable snap angle step (degrees, 0 = off)
        self.snap_step = 15.0

        # VBO caches — invalidated when data changes
        self._vbo_simple = None    # (flat_verts, flat_normals, n_tris)
        self._vbo_parts  = {}      # part.id -> (flat_verts, flat_normals, flat_colors, n_tris)
        self._vbo_heatmap = None   # (flat_verts, flat_colors, n_tris)
        self._vbo_grid = None      # (grid_verts, grid_colors, n_lines)

        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)

    # ── OpenGL ────────────────────────────────────────────────────

    def initializeGL(self):
        if not HAS_OPENGL: return
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING); glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glShadeModel(GL_SMOOTH); glEnable(GL_NORMALIZE)
        glLightfv(GL_LIGHT0, GL_POSITION, [2.0, 3.0, 4.0, 0.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE,  [0.85, 0.85, 0.85, 1.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT,  [0.20, 0.20, 0.20, 1.0])
        glClearColor(0.12, 0.13, 0.15, 1.0)

    def resizeGL(self, w, h):
        if not HAS_OPENGL: return
        h = h or 1
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION); glLoadIdentity()
        gluPerspective(45.0, w/h, 1.0, 20000.0)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        if not HAS_OPENGL: return
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        # Camera: zoom back, then pan
        glTranslatef(self.pan_x, self.pan_y, self.zoom)
        # Orbit: rotate around the orbit centre point
        glRotatef(self.rot_x, 1, 0, 0)
        glRotatef(self.rot_y, 0, 0, 1)
        oc = self.orbit_centre
        glTranslatef(-float(oc[0]), -float(oc[1]), -float(oc[2]))

        if self.show_grid:    self._draw_grid()
        if self.build_volume:  self._draw_build_volume()
        if self.parts:        self._draw_parts()
        elif self.vertices is not None:
            if self._heatmap_rgba is not None:
                self._draw_heatmap_mesh()
            else:
                self._draw_simple_mesh()

        if self._crease_lines: self._draw_crease_lines()
        if self.dowel_markers: self._draw_dowel_markers()
        if self.face_label_markers: self._draw_face_labels()
        if self.measure_points: self._draw_measure_points()
        if self.selected_faces: self._draw_face_selection()

        if self.show_cut_preview:
            self._draw_preview_cuts()
            if self.preview_cut is not None:
                self._draw_gizmo(self.preview_cut)

        # Draw joint previews (dowel holes, dovetail slots shown as wireframe)
        if self.joint_preview_meshes:
            self._draw_joint_previews()

        # Save matrices for hit-testing in mouse events
        self._save_matrices()

    # ── VBO-style array building ──────────────────────────────────

    def _build_simple_vbo(self):
        """Pre-compute flat vertex/normal arrays for the simple mesh."""
        if self.vertices is None or self.faces is None:
            self._vbo_simple = None
            return
        verts = self.vertices
        faces = self.faces
        v0 = verts[faces[:, 0]]; v1 = verts[faces[:, 1]]; v2 = verts[faces[:, 2]]
        norms = np.cross(v1 - v0, v2 - v0)
        lens = np.linalg.norm(norms, axis=1, keepdims=True)
        norms = norms / np.where(lens == 0, 1, lens)
        # Expand per-face normals to per-vertex (3 verts per face, same normal)
        flat_norms = np.repeat(norms, 3, axis=0).astype(np.float32)
        flat_verts = verts[faces.flatten()].astype(np.float32)
        self._vbo_simple = (flat_verts, flat_norms, len(faces))

    def _build_heatmap_vbo(self):
        """Pre-compute flat vertex/colour arrays for heatmap mesh."""
        if self.vertices is None or self.faces is None or self._heatmap_rgba is None:
            self._vbo_heatmap = None
            return
        rgba = self._heatmap_rgba
        if len(rgba) != len(self.vertices):
            self._vbo_heatmap = None
            return
        verts = self.vertices
        faces = self.faces
        flat_verts = verts[faces.flatten()].astype(np.float32)
        flat_colors = rgba[faces.flatten()].astype(np.float32)
        self._vbo_heatmap = (flat_verts, flat_colors, len(faces))

    def _build_part_vbo(self, part):
        """Pre-compute arrays for a single part mesh."""
        mesh = part.mesh
        if not len(mesh.faces):
            return None
        verts = mesh.vertices
        faces = mesh.faces
        v0 = verts[faces[:, 0]]; v1 = verts[faces[:, 1]]; v2 = verts[faces[:, 2]]
        norms = np.cross(v1 - v0, v2 - v0)
        lens = np.linalg.norm(norms, axis=1, keepdims=True)
        norms = norms / np.where(lens == 0, 1, lens)
        flat_norms = np.repeat(norms, 3, axis=0).astype(np.float32)
        flat_verts = verts[faces.flatten()].astype(np.float32)
        return (flat_verts, flat_norms, len(faces))

    # ── Drawing (array-based) ─────────────────────────────────────

    def _draw_simple_mesh(self):
        if self.vertices is None or self.faces is None: return
        if self._vbo_simple is None:
            self._build_simple_vbo()
        if self._vbo_simple is None: return

        flat_verts, flat_norms, n_tris = self._vbo_simple
        glColor3f(0.4, 0.75, 0.95)
        if self.show_wireframe:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, flat_verts)
        glNormalPointer(GL_FLOAT, 0, flat_norms)
        glDrawArrays(GL_TRIANGLES, 0, n_tris * 3)
        glDisableClientState(GL_NORMAL_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)

        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

    def _draw_heatmap_mesh(self):
        """Draw mesh with per-vertex seam quality colours (green=good, red=bad)."""
        if self.vertices is None or self.faces is None: return
        if self._heatmap_rgba is None or len(self._heatmap_rgba) != len(self.vertices):
            self._draw_simple_mesh(); return

        if self._vbo_heatmap is None:
            self._build_heatmap_vbo()
        if self._vbo_heatmap is None:
            self._draw_simple_mesh(); return

        flat_verts, flat_colors, n_tris = self._vbo_heatmap
        glDisable(GL_LIGHTING)

        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, flat_verts)
        glColorPointer(4, GL_FLOAT, 0, flat_colors)
        glDrawArrays(GL_TRIANGLES, 0, n_tris * 3)
        glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)

        glEnable(GL_LIGHTING)

    def _draw_crease_lines(self):
        """Draw natural crease lines as bright cyan lines."""
        if not self._crease_lines: return
        # Build flat array of line vertices
        n_lines = len(self._crease_lines)
        line_verts = np.zeros((n_lines * 2, 3), dtype=np.float32)
        valid = 0
        for seam in self._crease_lines:
            try:
                line_verts[valid * 2] = seam['v0']
                line_verts[valid * 2 + 1] = seam['v1']
                valid += 1
            except Exception:
                pass

        if valid == 0: return

        glDisable(GL_LIGHTING)
        glColor4f(0.0, 0.9, 0.9, 0.9)
        glLineWidth(2.5)

        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, line_verts[:valid * 2])
        glDrawArrays(GL_LINES, 0, valid * 2)
        glDisableClientState(GL_VERTEX_ARRAY)

        glLineWidth(1.0)
        glEnable(GL_LIGHTING)

    def _draw_parts(self):
        # 50 distinct colours using golden-angle hue spacing
        # Each consecutive colour is ~137.5° apart in hue — guarantees
        # no two adjacent parts look similar regardless of part count
        if not hasattr(Viewport3D, '_COLORS_CACHE'):
            import colorsys
            golden = 0.618033988749895
            cols = []
            for i in range(50):
                h = (i * golden) % 1.0
                s = 0.55 + (i % 3) * 0.15   # vary saturation 0.55-0.85
                v = 0.70 + (i % 2) * 0.15   # vary brightness 0.70-0.85
                r, g, b = colorsys.hsv_to_rgb(h, s, v)
                cols.append((r, g, b))
            Viewport3D._COLORS_CACHE = cols
        COLORS = Viewport3D._COLORS_CACHE

        # Compute model centroid for explode
        explode = self.explode_factor
        model_centroid = np.zeros(3)
        if explode > 0.001 and self.mesh_bounds is not None:
            model_centroid = (self.mesh_bounds[0] + self.mesh_bounds[1]) / 2.0
            scene_size = float(np.max(self.mesh_bounds[1] - self.mesh_bounds[0]))
            explode_dist = explode * scene_size * 0.5

        for part in self.parts:
            if hasattr(part, 'visible') and not part.visible: continue
            mesh = part.mesh
            if not len(mesh.faces): continue

            # Get or build cached arrays
            if part.id not in self._vbo_parts:
                vbo = self._build_part_vbo(part)
                if vbo is None: continue
                self._vbo_parts[part.id] = vbo
            flat_verts, flat_norms, n_tris = self._vbo_parts[part.id]

            is_sel = (part.id == self.selected_part_id)
            is_hover = (part.id == self._hover_part_id and not is_sel)
            # Always use the part's actual colour for the solid fill
            if hasattr(part, '_source_colour') and part._source_colour:
                r, g, b = part._source_colour
            else:
                r, g, b = COLORS[part.color_idx % len(COLORS)]
            # Brighten hovered parts
            if is_hover:
                r = min(1.0, r + 0.15); g = min(1.0, g + 0.15); b = min(1.0, b + 0.15)
            glColor3f(r, g, b)

            # Apply explode offset
            if explode > 0.001 and self.mesh_bounds is not None:
                part_centroid = (mesh.bounds[0] + mesh.bounds[1]) / 2.0
                direction = part_centroid - model_centroid
                dn = np.linalg.norm(direction)
                if dn > 0.001:
                    direction /= dn
                offset = direction * explode_dist
                glPushMatrix()
                glTranslatef(float(offset[0]), float(offset[1]), float(offset[2]))

            # Check per-part wireframe mode
            is_wf = getattr(part, '_wireframe', False)

            if is_wf:
                # Wireframe-only mode for this part
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE); glLineWidth(1.2)
            else:
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

            glEnableClientState(GL_VERTEX_ARRAY)
            glEnableClientState(GL_NORMAL_ARRAY)
            glVertexPointer(3, GL_FLOAT, 0, flat_verts)
            glNormalPointer(GL_FLOAT, 0, flat_norms)
            glDrawArrays(GL_TRIANGLES, 0, n_tris * 3)
            glDisableClientState(GL_NORMAL_ARRAY)
            glDisableClientState(GL_VERTEX_ARRAY)

            if is_wf:
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL); glLineWidth(1.0)

            if is_sel and self.show_selection_wireframe:
                # Draw wireframe outline ON TOP of the solid (yellow highlight)
                # Only when show_selection_wireframe is enabled
                glDisable(GL_LIGHTING); glColor4f(1, 0.9, 0.2, 0.5)
                glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE); glLineWidth(1.5)
                glEnableClientState(GL_VERTEX_ARRAY)
                glVertexPointer(3, GL_FLOAT, 0, flat_verts)
                glDrawArrays(GL_TRIANGLES, 0, n_tris * 3)
                glDisableClientState(GL_VERTEX_ARRAY)
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL); glLineWidth(1.0)
                glDisable(GL_BLEND); glEnable(GL_LIGHTING)
            elif is_sel:
                # Subtle highlight: just brighten the part slightly (no wireframe)
                glDisable(GL_LIGHTING)
                glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                glColor4f(1.0, 0.9, 0.3, 0.12)  # faint yellow glow
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
                glEnableClientState(GL_VERTEX_ARRAY)
                glVertexPointer(3, GL_FLOAT, 0, flat_verts)
                glDrawArrays(GL_TRIANGLES, 0, n_tris * 3)
                glDisableClientState(GL_VERTEX_ARRAY)
                glDisable(GL_BLEND); glEnable(GL_LIGHTING)

            if explode > 0.001 and self.mesh_bounds is not None:
                glPopMatrix()

    def _draw_preview_cuts(self):
        if not self.preview_cuts or self.mesh_bounds is None: return
        mins, maxs = self.mesh_bounds[0], self.mesh_bounds[1]
        size = float(np.max(maxs - mins)) * 0.65
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        for i, cut in enumerate(self.preview_cuts):
            is_active  = (i == self.active_cut_idx)
            is_locked  = getattr(cut, 'pinned', False)
            normal = cut.get_normal(); origin = cut.get_origin()
            if is_active:   fill=(1.0,0.5,0.05,0.30); edge=(1.0,0.65,0.1,1.0); lw=2.5
            elif is_locked: fill=(0.95,0.75,0.1,0.18); edge=(0.95,0.80,0.15,0.8); lw=1.8
            else:           fill=(0.3,0.55,0.95,0.10); edge=(0.4,0.65,0.95,0.5); lw=1.0

            # Section cuts show a sized rectangle; full/free cuts show a large plane
            if getattr(cut, 'mode', 'full') == 'section':
                self._draw_section_rect(cut, fill, edge, lw)
            else:
                self._draw_plane_quad(normal, origin, size, fill, edge, lw)
        glDisable(GL_BLEND); glEnable(GL_LIGHTING)

    def _draw_plane_quad(self, normal, origin, size, fill, edge, lw):
        n = np.array(normal, dtype=float)
        h = np.array([0, 0, 1.0]) if abs(n[2]) < 0.9 else np.array([0, 1.0, 0])
        u = np.cross(n, h); u /= np.linalg.norm(u)
        v = np.cross(n, u); v /= np.linalg.norm(v)
        corners = [origin+(u+v)*size, origin+(-u+v)*size, origin+(-u-v)*size, origin+(u-v)*size]
        glColor4f(*fill)
        glBegin(GL_QUADS); [glVertex3fv(c) for c in corners]; glEnd()
        glColor4f(*edge); glLineWidth(lw)
        glBegin(GL_LINE_LOOP); [glVertex3fv(c) for c in corners]; glEnd()
        glLineWidth(1.0)

    def _draw_section_rect(self, cut, fill, edge, lw):
        """Draw a sized rectangle for section cuts showing actual cut area."""
        origin = cut.get_origin()
        try:
            u, v = cut.get_plane_axes()
        except Exception:
            normal = cut.get_normal()
            h = np.array([0,0,1.0]) if abs(normal[2]) < 0.9 else np.array([0,1.0,0])
            u = np.cross(normal, h); u /= max(np.linalg.norm(u), 1e-9)
            v = np.cross(normal, u); v /= max(np.linalg.norm(v), 1e-9)
        hw = getattr(cut, 'section_w', 100) / 2.0
        hh = getattr(cut, 'section_h', 100) / 2.0
        corners = [origin + u*hw + v*hh, origin - u*hw + v*hh,
                   origin - u*hw - v*hh, origin + u*hw - v*hh]
        # Fill
        glColor4f(*fill)
        glBegin(GL_QUADS); [glVertex3fv(c) for c in corners]; glEnd()
        # Edge with dashed-style emphasis
        glColor4f(*edge); glLineWidth(lw + 1)
        glBegin(GL_LINE_LOOP); [glVertex3fv(c) for c in corners]; glEnd()
        # Corner markers
        glPointSize(6.0)
        glBegin(GL_POINTS); [glVertex3fv(c) for c in corners]; glEnd()
        glPointSize(1.0)
        glLineWidth(1.0)

    def _draw_joint_previews(self):
        """Draw joint preview meshes (dowel holes, dovetails) as cyan wireframes."""
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glColor4f(0.1, 0.9, 0.8, 0.7)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        glLineWidth(1.5)
        for mesh in self.joint_preview_meshes:
            try:
                verts = mesh.vertices.astype(np.float32)
                faces = mesh.faces
                flat_verts = verts[faces.flatten()].astype(np.float32)
                glEnableClientState(GL_VERTEX_ARRAY)
                glVertexPointer(3, GL_FLOAT, 0, flat_verts)
                glDrawArrays(GL_TRIANGLES, 0, len(faces) * 3)
                glDisableClientState(GL_VERTEX_ARRAY)
            except Exception:
                pass
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glLineWidth(1.0)
        glDisable(GL_BLEND); glEnable(GL_LIGHTING)

    def _draw_gizmo(self, cut):
        if cut is None: return
        origin = cut.get_origin()
        normal = cut.get_normal()
        if self.mesh_bounds is not None:
            scene_size = float(np.max(self.mesh_bounds[1] - self.mesh_bounds[0]))
        else:
            scene_size = 200.0
        gs = scene_size * 0.18
        rs = scene_size * 0.15

        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Translate arrow (yellow)
        ahit = (self.gizmo_hit == HIT_ARROW)
        ac = (1.0,1.0,0.4,1.0) if ahit else (1.0,0.85,0.1,0.9)
        glColor4f(*ac); glLineWidth(4.0 if ahit else 3.0)
        tip  = origin + normal * gs
        tip2 = origin - normal * gs
        glBegin(GL_LINES); glVertex3fv(origin); glVertex3fv(tip);  glEnd()
        glBegin(GL_LINES); glVertex3fv(origin); glVertex3fv(tip2); glEnd()
        self._draw_cone(tip,  normal,   gs*0.10, gs*0.20, ac)
        self._draw_cone(tip2, -normal,  gs*0.10, gs*0.20, ac)

        # Rotation rings with current angle indicators
        ring_angles = {
            HIT_RING_X: cut.rot_u if hasattr(cut, 'rot_u') else 0,
            HIT_RING_Y: cut.rot_v if hasattr(cut, 'rot_v') else 0,
            HIT_RING_Z: cut.rot_n if hasattr(cut, 'rot_n') else 0,
        }
        for ax, col, hit_id in [
            (np.array([1.,0,0]), (0.9,0.2,0.2,0.85), HIT_RING_X),
            (np.array([0,1.,0]), (0.2,0.85,0.2,0.85), HIT_RING_Y),
            (np.array([0,0,1.]), (0.2,0.45,0.95,0.85), HIT_RING_Z),
        ]:
            is_hit = (self.gizmo_hit == hit_id)
            is_dragging = (self._gizmo_drag == hit_id)
            c = (min(col[0]+0.2,1), min(col[1]+0.2,1), min(col[2]+0.2,1), 1.0) if is_hit else col
            # Show current angle indicator only when dragging this ring
            cur_angle = ring_angles.get(hit_id) if is_dragging else None
            self._draw_ring(origin, ax, rs, c, lw=4.0 if is_hit else 2.5,
                           current_angle=cur_angle)

        glLineWidth(1.0); glDisable(GL_BLEND); glEnable(GL_LIGHTING)

    def _draw_cone(self, tip, direction, radius, length, color):
        glColor4f(*color)
        n = np.array(direction, dtype=float); n /= max(np.linalg.norm(n), 1e-9)
        h = np.array([0, 0, 1.0]) if abs(n[2]) < 0.9 else np.array([0, 1.0, 0])
        u = np.cross(n, h); u /= np.linalg.norm(u); v = np.cross(n, u); v /= np.linalg.norm(v)
        base = tip - n * length; segs = 12
        glBegin(GL_TRIANGLE_FAN); glVertex3fv(tip)
        for i in range(segs + 1):
            a = 2 * np.pi * i / segs; glVertex3fv(base + radius * (np.cos(a) * u + np.sin(a) * v))
        glEnd()

    def _draw_ring(self, centre, axis, radius, color, lw=2.5, current_angle=None):
        """Draw a rotation ring with Bambu-style snap tick marks."""
        n = np.array(axis, dtype=float); n /= max(np.linalg.norm(n), 1e-9)
        h = np.array([0, 0, 1.0]) if abs(n[2]) < 0.9 else np.array([0, 1.0, 0])
        u = np.cross(n, h); u /= np.linalg.norm(u); v = np.cross(n, u); v /= np.linalg.norm(v)

        # Main ring
        glColor4f(*color); glLineWidth(lw)
        segs = 48
        glBegin(GL_LINE_LOOP)
        for i in range(segs):
            a = 2 * np.pi * i / segs
            glVertex3fv(centre + radius * (np.cos(a) * u + np.sin(a) * v))
        glEnd()

        # Snap tick marks at 15° intervals
        tick_small = radius * 0.06  # small tick length
        tick_large = radius * 0.12  # large tick for 0/45/90
        tick_step = max(5, int(self.snap_step)) if self.snap_step > 0 else 15
        for deg in range(0, 360, tick_step):
            a = np.radians(deg)
            pt_inner = centre + radius * (np.cos(a) * u + np.sin(a) * v)
            is_major = (deg % 45 == 0)
            tick_len = tick_large if is_major else tick_small
            tick_dir = np.cos(a) * u + np.sin(a) * v
            pt_outer = pt_inner + tick_dir * tick_len

            if is_major:
                glColor4f(color[0], color[1], color[2], 1.0)
                glLineWidth(2.5)
            else:
                glColor4f(color[0]*0.7, color[1]*0.7, color[2]*0.7, 0.6)
                glLineWidth(1.5)
            glBegin(GL_LINES)
            glVertex3fv(pt_inner); glVertex3fv(pt_outer)
            glEnd()

        # If actively dragging, show current angle indicator
        if current_angle is not None:
            a = np.radians(current_angle)
            pt = centre + radius * 1.15 * (np.cos(a) * u + np.sin(a) * v)
            pt0 = centre + radius * 0.85 * (np.cos(a) * u + np.sin(a) * v)
            glColor4f(1, 1, 1, 0.9); glLineWidth(3.0)
            glBegin(GL_LINES); glVertex3fv(pt0); glVertex3fv(pt); glEnd()

        glLineWidth(1.0)

    def _build_grid_vbo(self):
        """Pre-compute grid vertex/colour arrays."""
        size = 400; step = 50
        lines = []
        for i in range(-size, size + step, step):
            lines.extend([[i, -size, 0], [i, size, 0]])
            lines.extend([[-size, i, 0], [size, i, 0]])
        grid_verts = np.array(lines, dtype=np.float32)
        grid_colors = np.full((len(grid_verts), 3), [0.20, 0.22, 0.26], dtype=np.float32)
        # Axis lines
        axis_verts = np.array([
            [0,0,0],[120,0,0], [0,0,0],[0,120,0], [0,0,0],[0,0,120]
        ], dtype=np.float32)
        axis_colors = np.array([
            [0.8,0.2,0.2],[0.8,0.2,0.2],
            [0.2,0.8,0.2],[0.2,0.8,0.2],
            [0.2,0.4,0.9],[0.2,0.4,0.9],
        ], dtype=np.float32)
        self._vbo_grid = (grid_verts, grid_colors, len(grid_verts),
                          axis_verts, axis_colors, len(axis_verts))

    def _draw_grid(self):
        if self._vbo_grid is None:
            self._build_grid_vbo()
        gv, gc, gn, av, ac, an = self._vbo_grid
        glDisable(GL_LIGHTING); glLineWidth(1.0)
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, gv)
        glColorPointer(3, GL_FLOAT, 0, gc)
        glDrawArrays(GL_LINES, 0, gn)
        # Axis lines (thicker)
        glLineWidth(2.0)
        glVertexPointer(3, GL_FLOAT, 0, av)
        glColorPointer(3, GL_FLOAT, 0, ac)
        glDrawArrays(GL_LINES, 0, an)
        glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)
        # Axis labels
        glColor3f(0.8, 0.2, 0.2)
        self._draw_text_3d(130, 0, 0, "X")
        glColor3f(0.2, 0.8, 0.2)
        self._draw_text_3d(0, 130, 0, "Y")
        glColor3f(0.2, 0.4, 0.9)
        self._draw_text_3d(0, 0, 130, "Z")
        glLineWidth(1.0); glEnable(GL_LIGHTING)

    def _draw_text_3d(self, x, y, z, text):
        """Draw simple text at a 3D position using line strokes."""
        s = 8.0  # size
        glLineWidth(2.0)
        glBegin(GL_LINES)
        if text == "X":
            glVertex3f(x-s, y-s, z); glVertex3f(x+s, y+s, z)
            glVertex3f(x+s, y-s, z); glVertex3f(x-s, y+s, z)
        elif text == "Y":
            glVertex3f(x-s, y+s, z); glVertex3f(x, y, z)
            glVertex3f(x+s, y+s, z); glVertex3f(x, y, z)
            glVertex3f(x, y, z); glVertex3f(x, y-s, z)
        elif text == "Z":
            glVertex3f(x-s, y, z+s); glVertex3f(x+s, y, z+s)
            glVertex3f(x+s, y, z+s); glVertex3f(x-s, y, z-s)
            glVertex3f(x-s, y, z-s); glVertex3f(x+s, y, z-s)
        glEnd()
        glLineWidth(1.0)

    def _draw_build_volume(self):
        """Draw printer build volume as a translucent wireframe box."""
        if not self.build_volume: return
        bx, by, bz = self.build_volume
        hx, hy = bx / 2, by / 2
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        # Wireframe box
        glColor4f(0.3, 0.6, 0.9, 0.3); glLineWidth(1.5)
        glBegin(GL_LINES)
        # Bottom rectangle
        for x1, y1, x2, y2 in [(-hx,-hy, hx,-hy), (hx,-hy, hx,hy),
                                  (hx,hy, -hx,hy), (-hx,hy, -hx,-hy)]:
            glVertex3f(x1, y1, 0); glVertex3f(x2, y2, 0)
        # Top rectangle
        for x1, y1, x2, y2 in [(-hx,-hy, hx,-hy), (hx,-hy, hx,hy),
                                  (hx,hy, -hx,hy), (-hx,hy, -hx,-hy)]:
            glVertex3f(x1, y1, bz); glVertex3f(x2, y2, bz)
        # Verticals
        for x, y in [(-hx,-hy), (hx,-hy), (hx,hy), (-hx,hy)]:
            glVertex3f(x, y, 0); glVertex3f(x, y, bz)
        glEnd()
        # Translucent bottom face
        glColor4f(0.2, 0.4, 0.7, 0.08)
        glBegin(GL_QUADS)
        glVertex3f(-hx, -hy, 0); glVertex3f(hx, -hy, 0)
        glVertex3f(hx, hy, 0); glVertex3f(-hx, hy, 0)
        glEnd()
        glLineWidth(1.0); glDisable(GL_BLEND); glEnable(GL_LIGHTING)

    def _draw_dowel_markers(self):
        """Draw manually placed dowel markers as small spheres."""
        if not self.dowel_markers: return
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glColor4f(0.0, 0.9, 0.8, 0.9)
        # Draw small diamonds at each marker position
        marker_size = 3.0
        for pos, normal in self.dowel_markers:
            p = np.array(pos, dtype=float)
            n = np.array(normal, dtype=float)
            n /= max(np.linalg.norm(n), 1e-9)
            # Cross to get perpendicular axes
            h = np.array([0,0,1.0]) if abs(n[2]) < 0.9 else np.array([0,1.0,0])
            u = np.cross(n, h); u /= max(np.linalg.norm(u), 1e-9)
            v = np.cross(n, u); v /= max(np.linalg.norm(v), 1e-9)
            # Draw a diamond shape
            pts = [p + u*marker_size, p + v*marker_size,
                   p - u*marker_size, p - v*marker_size]
            glBegin(GL_LINE_LOOP)
            for pt in pts: glVertex3fv(pt)
            glEnd()
            # Normal indicator line
            glBegin(GL_LINES)
            glVertex3fv(p); glVertex3fv(p + n * marker_size * 2)
            glEnd()
        glDisable(GL_BLEND); glEnable(GL_LIGHTING)

    def _draw_face_selection(self):
        """Draw selected faces as a translucent orange overlay."""
        if not self.selected_faces: return
        # Get the mesh to highlight on
        mesh = None
        if self.parts:
            sel_part = next((p for p in self.parts if p.id == self.selected_part_id), None)
            if sel_part:
                mesh = sel_part.mesh
        elif self.vertices is not None and self.faces is not None:
            mesh = type('M', (), {'vertices': self.vertices, 'faces': self.faces})()

        if mesh is None: return

        # Build arrays for selected faces only
        sel_list = [fi for fi in self.selected_faces
                    if 0 <= fi < len(mesh.faces)]
        if not sel_list: return

        sel_faces = np.array(mesh.faces)[sel_list]
        flat_verts = np.array(mesh.vertices)[sel_faces.flatten()].astype(np.float32)

        glDisable(GL_LIGHTING); glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glColor4f(1.0, 0.5, 0.0, 0.35)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, flat_verts)
        glDrawArrays(GL_TRIANGLES, 0, len(sel_list) * 3)
        glDisableClientState(GL_VERTEX_ARRAY)

        # Wireframe outline
        glColor4f(1.0, 0.6, 0.0, 0.7)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE); glLineWidth(1.5)
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, flat_verts)
        glDrawArrays(GL_TRIANGLES, 0, len(sel_list) * 3)
        glDisableClientState(GL_VERTEX_ARRAY)

        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL); glLineWidth(1.0)
        glDisable(GL_BLEND); glEnable(GL_DEPTH_TEST); glEnable(GL_LIGHTING)

    def _draw_measure_points(self):
        """Draw measurement points and line between them."""
        if not self.measure_points: return
        glDisable(GL_LIGHTING); glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Draw crosshairs at each point
        size = 5.0
        for pt in self.measure_points:
            p = np.array(pt, dtype=float)
            glColor4f(1.0, 0.2, 0.2, 1.0); glLineWidth(3.0)
            glBegin(GL_LINES)
            glVertex3fv(p + np.array([-size, 0, 0])); glVertex3fv(p + np.array([size, 0, 0]))
            glVertex3fv(p + np.array([0, -size, 0])); glVertex3fv(p + np.array([0, size, 0]))
            glVertex3fv(p + np.array([0, 0, -size])); glVertex3fv(p + np.array([0, 0, size]))
            glEnd()

        # Draw line between two points
        if len(self.measure_points) >= 2:
            p1, p2 = self.measure_points[0], self.measure_points[1]
            glColor4f(1.0, 1.0, 0.0, 0.9); glLineWidth(2.5)
            glBegin(GL_LINES)
            glVertex3fv(p1); glVertex3fv(p2)
            glEnd()

        glLineWidth(1.0)
        glDisable(GL_BLEND); glEnable(GL_DEPTH_TEST); glEnable(GL_LIGHTING)

    def _draw_face_labels(self):
        """Draw face label text (A1, B2, etc.) on cut faces as coloured shapes."""
        if not self.face_label_markers: return
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Colour cycle for different letter groups
        label_colors = {
            'A': (1.0, 0.4, 0.4, 0.9),  # red
            'B': (0.4, 1.0, 0.4, 0.9),  # green
            'C': (0.4, 0.6, 1.0, 0.9),  # blue
            'D': (1.0, 0.8, 0.2, 0.9),  # yellow
            'E': (1.0, 0.4, 0.8, 0.9),  # pink
            'F': (0.4, 1.0, 0.9, 0.9),  # cyan
        }

        for centre, normal, text in self.face_label_markers:
            c = np.array(centre, dtype=float)
            n = np.array(normal, dtype=float)
            n /= max(np.linalg.norm(n), 1e-9)

            # Get face-local axes
            helper = np.array([0, 0, 1.0]) if abs(n[2]) < 0.9 else np.array([0, 1.0, 0])
            u = np.cross(n, helper); u /= max(np.linalg.norm(u), 1e-9)
            v = np.cross(n, u); v /= max(np.linalg.norm(v), 1e-9)

            # Position slightly offset from face surface
            pos = c + n * 0.5

            # Pick colour from first character
            letter = text[0].upper() if text else 'A'
            col = label_colors.get(letter, (0.8, 0.8, 0.8, 0.9))
            glColor4f(*col)

            # Draw the label as a bordered rectangle with diagonal for the number
            size = 8.0
            # Background rectangle
            corners = [
                pos - u * size - v * size,
                pos + u * size - v * size,
                pos + u * size + v * size,
                pos - u * size + v * size,
            ]
            glLineWidth(2.5)
            glBegin(GL_LINE_LOOP)
            for pt in corners: glVertex3fv(pt)
            glEnd()

            # Draw letter as simple strokes
            if len(text) >= 1:
                ch = text[0].upper()
                s = size * 0.7
                if ch == 'A':
                    pts = [pos - u*s - v*s, pos - v*s + v*s*2, pos + u*s - v*s]
                    glBegin(GL_LINE_STRIP)
                    for pt in pts: glVertex3fv(pt)
                    glEnd()
                    glBegin(GL_LINES)
                    glVertex3fv(pos - u*s*0.5); glVertex3fv(pos + u*s*0.5)
                    glEnd()
                elif ch == 'B':
                    glBegin(GL_LINE_STRIP)
                    glVertex3fv(pos - u*s - v*s); glVertex3fv(pos - u*s + v*s)
                    glVertex3fv(pos + u*s*0.5 + v*s); glVertex3fv(pos + u*s*0.5)
                    glVertex3fv(pos - u*s); glVertex3fv(pos + u*s*0.5)
                    glVertex3fv(pos + u*s*0.5 - v*s); glVertex3fv(pos - u*s - v*s)
                    glEnd()
                elif ch == 'C':
                    glBegin(GL_LINE_STRIP)
                    glVertex3fv(pos + u*s + v*s); glVertex3fv(pos - u*s + v*s)
                    glVertex3fv(pos - u*s - v*s); glVertex3fv(pos + u*s - v*s)
                    glEnd()
                elif ch == 'D':
                    glBegin(GL_LINE_STRIP)
                    glVertex3fv(pos - u*s - v*s); glVertex3fv(pos - u*s + v*s)
                    glVertex3fv(pos + u*s*0.5 + v*s*0.5)
                    glVertex3fv(pos + u*s*0.5 - v*s*0.5)
                    glVertex3fv(pos - u*s - v*s)
                    glEnd()
                else:
                    # Generic: draw an X for unknown letters
                    glBegin(GL_LINES)
                    glVertex3fv(pos - u*s - v*s); glVertex3fv(pos + u*s + v*s)
                    glVertex3fv(pos + u*s - v*s); glVertex3fv(pos - u*s + v*s)
                    glEnd()

            # Draw number part (digit after letter)
            if len(text) >= 2 and text[1].isdigit():
                digit = int(text[1])
                # Offset to the right of the letter
                dpos = pos + u * size * 1.5
                ds = size * 0.5
                # Simple: draw the digit as a small number indicator
                # 1 = single line, 2 = two lines
                glBegin(GL_LINES)
                if digit == 1:
                    glVertex3fv(dpos - v*ds); glVertex3fv(dpos + v*ds)
                elif digit == 2:
                    glVertex3fv(dpos - v*ds - u*ds*0.3); glVertex3fv(dpos + v*ds - u*ds*0.3)
                    glVertex3fv(dpos - v*ds + u*ds*0.3); glVertex3fv(dpos + v*ds + u*ds*0.3)
                else:
                    for k in range(digit):
                        off = (k - (digit-1)/2) * ds * 0.5
                        glVertex3fv(dpos + u*off - v*ds*0.3)
                        glVertex3fv(dpos + u*off + v*ds*0.3)
                glEnd()

            glLineWidth(1.0)

        glDisable(GL_BLEND); glEnable(GL_LIGHTING)

    # ── Gizmo hit testing ─────────────────────────────────────────

    def _gizmo_hit_test(self, mx, my):
        if self.preview_cut is None: return HIT_NONE
        cut = self.preview_cut
        origin = cut.get_origin(); normal = cut.get_normal()
        if self.mesh_bounds is not None:
            gs = float(np.max(self.mesh_bounds[1] - self.mesh_bounds[0])) * 0.18
            rs = float(np.max(self.mesh_bounds[1] - self.mesh_bounds[0])) * 0.15
        else:
            gs = 36.0; rs = 30.0

        tip = origin + normal * gs; tip2 = origin - normal * gs
        s0 = self._proj(origin); st = self._proj(tip); st2 = self._proj(tip2)
        if s0 is None: return HIT_NONE

        # Arrow hit
        if st and st2:
            d1 = self._seg_dist(mx, my, s0, st); d2 = self._seg_dist(mx, my, s0, st2)
            if min(d1, d2) < 22: return HIT_ARROW

        # Ring hits — check each ring, pick the closest
        ring_hits = []
        for ax, hit_id in [(np.array([1.,0,0]),HIT_RING_X),
                            (np.array([0,1.,0]),HIT_RING_Y),
                            (np.array([0,0,1.]),HIT_RING_Z)]:
            d = self._ring_dist(mx, my, origin, ax, rs)
            if d < 28:
                ring_hits.append((d, hit_id))
        if ring_hits:
            ring_hits.sort()
            return ring_hits[0][1]

        return HIT_NONE

    def _ring_dist(self, mx, my, centre, axis, radius):
        """Min distance from mouse to ring screen projection."""
        n = np.array(axis, dtype=float); n /= max(np.linalg.norm(n), 1e-9)
        h = np.array([0, 0, 1.0]) if abs(n[2]) < 0.9 else np.array([0, 1.0, 0])
        u = np.cross(n, h); u /= np.linalg.norm(u); v = np.cross(n, u); v /= np.linalg.norm(v)
        segs = 32; min_d = float('inf')
        for i in range(segs):
            a = 2 * np.pi * i / segs
            pt = centre + radius * (np.cos(a) * u + np.sin(a) * v)
            sp = self._proj(pt)
            if sp: min_d = min(min_d, np.hypot(mx - sp[0], my - sp[1]))
        return min_d

    def _seg_dist(self, px, py, a, b):
        ax, ay = a; bx, by = b; dx, dy = bx - ax, by - ay
        if dx == 0 and dy == 0: return np.hypot(px - ax, py - ay)
        t = max(0, min(1, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
        return np.hypot(px - (ax + t * dx), py - (ay + t * dy))

    def _save_matrices(self):
        """Save current GL matrices after paintGL for use in mouse events."""
        try:
            self._saved_mv = glGetDoublev(GL_MODELVIEW_MATRIX)
            self._saved_pr = glGetDoublev(GL_PROJECTION_MATRIX)
            self._saved_vp = glGetIntegerv(GL_VIEWPORT)
        except Exception:
            pass

    def _proj(self, pt):
        try:
            # Use saved matrices if available (works outside paintGL)
            if hasattr(self, '_saved_mv') and self._saved_mv is not None:
                mv = self._saved_mv; pr = self._saved_pr; vp = self._saved_vp
                wx, wy, wz = gluProject(float(pt[0]), float(pt[1]), float(pt[2]), mv, pr, vp)
                return (wx, self.height() - wy)
            mv = glGetDoublev(GL_MODELVIEW_MATRIX); pr = glGetDoublev(GL_PROJECTION_MATRIX)
            vp = glGetIntegerv(GL_VIEWPORT)
            wx, wy, wz = gluProject(float(pt[0]), float(pt[1]), float(pt[2]), mv, pr, vp)
            return (wx, self.height() - wy)
        except Exception:
            return None

    # ── Gizmo drag ────────────────────────────────────────────────

    def _snap_angle(self, angle, snap_step=None, snap_threshold=3.0):
        """Snap angle to nearest multiple of snap_step if within threshold."""
        if snap_step is None:
            snap_step = self.snap_step
        if snap_step <= 0:
            return angle  # snapping disabled
        from PyQt5.QtWidgets import QApplication
        modifiers = QApplication.keyboardModifiers()
        if modifiers & Qt.ShiftModifier:
            return angle
        nearest = round(angle / snap_step) * snap_step
        if abs(angle - nearest) < snap_threshold:
            return nearest
        return angle

    def _snap_to_crease(self, position, axis, threshold=10.0):
        """Snap a cut position to the nearest crease edge if within threshold."""
        if not self.snap_to_creases or not self.crease_snap_points:
            self._snapped_crease = None
            return position
        ax_idx = {'x': 0, 'y': 1, 'z': 2}.get(axis, 0)
        best_dist = threshold
        best_pos = position
        best_crease = None
        for cp in self.crease_snap_points:
            if cp.get('axis', '') != axis:
                continue
            crease_pos = cp.get('position', 0)
            dist = abs(position - crease_pos)
            if dist < best_dist:
                best_dist = dist
                best_pos = crease_pos
                best_crease = cp
        self._snapped_crease = best_crease
        return best_pos

    def _apply_gizmo_drag(self, dx, dy):
        cut = self.preview_cut
        if cut is None: return
        if self._gizmo_drag == HIT_ARROW:
            delta = self._screen_to_world(dx, dy, cut.get_normal())
            ax_idx = {'x':0, 'y':1, 'z':2}.get(cut.axis, 0)
            new_pos = float(cut.position[ax_idx]) + delta
            # Snap to nearby crease edges
            new_pos = self._snap_to_crease(new_pos, cut.axis)
            self.cut_moved.emit(new_pos)
        elif self._gizmo_drag in (HIT_RING_X, HIT_RING_Y, HIT_RING_Z):
            # Switch to free mode so rotation takes effect
            if cut.mode == 'full':
                cut.mode = 'free'
            deg = dx * 0.5
            if self._gizmo_drag == HIT_RING_X:
                raw = max(-89, min(89, cut.rot_u + deg))
                ru = self._snap_angle(raw)
                rv = cut.rot_v
                rn = cut.rot_n
            elif self._gizmo_drag == HIT_RING_Y:
                ru = cut.rot_u
                raw = max(-89, min(89, cut.rot_v + deg))
                rv = self._snap_angle(raw)
                rn = cut.rot_n
            else:  # RING_Z — spin around cut normal
                ru = cut.rot_u
                rv = cut.rot_v
                rn = self._snap_angle(cut.rot_n + deg)
            self.cut_rotated.emit(ru, rv, rn)

    def _screen_to_world(self, dx, dy, world_axis):
        rx, ry = np.radians(self.rot_x), np.radians(self.rot_y)
        cam_r = np.array([np.cos(ry), np.sin(ry), 0])
        cam_u = np.array([-np.sin(ry) * np.sin(rx), np.cos(ry) * np.sin(rx), np.cos(rx)])
        ax = np.array(world_axis, dtype=float)
        ln = np.linalg.norm(ax)
        if ln > 0: ax /= ln
        sx = np.dot(ax, cam_r); sy = np.dot(ax, cam_u)
        sl = np.hypot(sx, sy)
        if sl < 0.001: return 0.0
        scale = abs(self.zoom) / (max(self.height(), 1) * 0.6)
        return (dx * sx - dy * sy) / sl * scale

    # ── Data setters (invalidate VBO caches) ──────────────────────

    def set_mesh(self, vertices, faces, bounds=None):
        self.vertices = vertices; self.faces = faces
        self.parts = []; self.all_parts = []; self.mesh_bounds = bounds
        self._vbo_simple = None
        self._vbo_heatmap = None
        self._vbo_parts = {}
        if bounds is not None:
            c = (bounds[0] + bounds[1]) / 2; sz = float(np.max(bounds[1] - bounds[0]))
            self.zoom = -sz * 2.5
            self.pan_x = 0.0; self.pan_y = 0.0
            self.orbit_centre = c.copy()
        self.update()

    def set_parts(self, parts, all_parts=None, bounds=None):
        self.parts = parts; self.all_parts = all_parts or parts
        self.vertices = None; self.faces = None
        self._vbo_simple = None
        self._vbo_heatmap = None
        self._vbo_parts = {}
        if bounds is not None: self.mesh_bounds = bounds
        self.update()

    def set_selected_part(self, part_id):
        self.selected_part_id = part_id
        # Move orbit centre to the selected part's centroid
        for part in self.parts:
            if part.id == part_id:
                try:
                    c = (part.mesh.bounds[0] + part.mesh.bounds[1]) / 2.0
                    self.orbit_centre = c.copy()
                except Exception:
                    pass
                break
        self.update()

    def set_preview_cuts(self, cuts, active_idx=-1):
        self.preview_cuts = cuts; self.active_cut_idx = active_idx
        self.preview_cut = cuts[active_idx] if 0 <= active_idx < len(cuts) else None
        self.update()

    def set_preview_cut(self, cut):
        self.preview_cut = cut
        if cut is not None and cut not in self.preview_cuts:
            self.preview_cuts = [cut]; self.active_cut_idx = 0
        self.update()

    def set_joint_preview(self, meshes):
        """Set joint preview meshes (dowel/dovetail visualisation)."""
        self.joint_preview_meshes = meshes if meshes else []
        self.update()

    def set_heatmap(self, scores_rgba: np.ndarray):
        """
        Set per-vertex colour data for seam heatmap overlay.
        scores_rgba: (N, 4) float32 array, one RGBA per vertex of the root mesh.
        Pass None to clear.
        """
        self._heatmap_rgba = scores_rgba
        self._vbo_heatmap = None
        self.update()

    def set_crease_lines(self, seam_list: list):
        """
        Set crease edge lines to draw (natural seam suggestions).
        seam_list: list of dicts with 'v0','v1' world positions.
        """
        self._crease_lines = seam_list
        self.update()

    def clear_heatmap(self):
        self._heatmap_rgba = None
        self._vbo_heatmap = None
        self._crease_lines = []
        self.update()

    def set_cut_planes(self, planes):
        self.preview_cuts = planes if planes else []; self.update()

    def clear(self):
        self.vertices = None; self.faces = None
        self.parts = []; self.all_parts = []
        self.preview_cut = None; self.preview_cuts = []
        self.joint_preview_meshes = []
        self._heatmap_rgba = None; self._crease_lines = []
        self._vbo_simple = None; self._vbo_heatmap = None; self._vbo_parts = {}
        self.face_label_markers = []; self.dowel_markers = []
        self.mesh_bounds = None; self.update()

    # ── Mouse events ───────────────────────────────────────────────

    def _unproject(self, mx, my, depth=0.5):
        """Unproject screen coords to world ray (origin, direction)."""
        try:
            mv = glGetDoublev(GL_MODELVIEW_MATRIX)
            pr = glGetDoublev(GL_PROJECTION_MATRIX)
            vp = glGetIntegerv(GL_VIEWPORT)
            wy = self.height() - my
            x0, y0, z0 = gluUnProject(float(mx), float(wy), 0.0, mv, pr, vp)
            x1, y1, z1 = gluUnProject(float(mx), float(wy), 1.0, mv, pr, vp)
            origin = np.array([x0, y0, z0])
            direction = np.array([x1 - x0, y1 - y0, z1 - z0])
            direction /= max(np.linalg.norm(direction), 1e-9)
            return origin, direction
        except Exception:
            return None, None

    def _raycast_parts(self, mx, my, only_selected=False):
        """Cast a ray through screen point and find the closest VISIBLE part hit.
        If only_selected=True, only tests the selected part.
        Returns (hit_position, face_normal, part) or (None, None, None)."""
        origin, direction = self._unproject(mx, my)
        if origin is None:
            return None, None, None

        best_dist = float('inf')
        best_pos = None
        best_normal = None
        best_part = None

        for part in self.parts:
            # Skip hidden parts
            if hasattr(part, 'visible') and not part.visible:
                continue
            # If only_selected, skip parts that aren't the selected one
            if only_selected and part.id != self.selected_part_id:
                continue
            mesh = part.mesh
            if not len(mesh.faces):
                continue
            try:
                locations, ray_idx, face_idx = mesh.ray.intersects_location(
                    [origin], [direction])
                if len(locations) > 0:
                    dists = np.linalg.norm(locations - origin, axis=1)
                    closest = int(np.argmin(dists))
                    if dists[closest] < best_dist:
                        best_dist = dists[closest]
                        best_pos = locations[closest]
                        best_normal = mesh.face_normals[face_idx[closest]]
                        best_part = part
            except Exception:
                pass

        return best_pos, best_normal, best_part

    def mousePressEvent(self, e):
        self._last_pos = e.pos()
        if e.button() == Qt.LeftButton:
            self._lmb_down = True
            # Ensure GL context is current for projection calculations
            self.makeCurrent()

            # Selection brush mode — paint-select faces (only on selected part)
            if self.selection_mode and self.parts:
                pos, normal, part = self._raycast_parts(e.x(), e.y(), only_selected=True)
                if pos is not None and part is not None:
                    from core.region_repair import select_faces_near_point
                    new_sel = select_faces_near_point(
                        part.mesh, pos, self.selection_brush_radius)
                    from PyQt5.QtWidgets import QApplication
                    mods = QApplication.keyboardModifiers()
                    if mods & Qt.ShiftModifier:
                        # Shift+click = remove from selection
                        self.selected_faces -= new_sel
                    elif mods & Qt.ControlModifier:
                        # Ctrl+click = flood-fill select similar normals
                        from core.region_repair import select_faces_by_normal_similarity
                        if new_sel:
                            seed = next(iter(new_sel))
                            flood = select_faces_by_normal_similarity(part.mesh, seed)
                            self.selected_faces |= flood
                    else:
                        # Normal click = add to selection
                        self.selected_faces |= new_sel
                    self.faces_selected.emit(self.selected_faces)
                    self.update()
                    return

            # Measure mode — raycast to get point
            if getattr(self, '_measure_active', False) and self.parts:
                pos, normal, part = self._raycast_parts(e.x(), e.y())
                if pos is not None:
                    self.measure_clicked.emit(pos, normal)
                    self.update()
                    return

            # Manual dowel placement mode — raycast ONLY selected part
            if self.manual_dowel_mode and self.parts:
                pos, normal, part = self._raycast_parts(e.x(), e.y(), only_selected=True)
                if pos is not None:
                    self.dowel_placed.emit(pos, normal)
                    self.update()
                    return

            hit = self._gizmo_hit_test(e.x(), e.y())
            if hit != HIT_NONE:
                self._gizmo_drag = hit
                self.setCursor(Qt.ClosedHandCursor)
                return
            self._gizmo_drag = HIT_NONE

            # Left-click on a part = select it (when not in any special mode)
            if self.parts and not self.selection_mode:
                pos, normal, part = self._raycast_parts(e.x(), e.y())
                if part is not None:
                    self.part_left_clicked.emit(part)
        elif e.button() == Qt.RightButton:
            self._rmb_down = True
            self._rmb_start = e.pos()  # track start for click vs drag
            self.setCursor(Qt.SizeAllCursor)
        elif e.button() == Qt.MiddleButton:
            self._mmb_down = True

    def mouseMoveEvent(self, e):
        dx = e.x() - self._last_pos.x(); dy = e.y() - self._last_pos.y()
        self.makeCurrent()

        # Selection brush drag-painting (only on selected part)
        if self._lmb_down and self.selection_mode and self.parts:
            pos, normal, part = self._raycast_parts(e.x(), e.y(), only_selected=True)
            if pos is not None and part is not None:
                from core.region_repair import select_faces_near_point
                new_sel = select_faces_near_point(
                    part.mesh, pos, self.selection_brush_radius)
                self.selected_faces |= new_sel
                self.update()
            self._last_pos = e.pos(); return

        if self._gizmo_drag != HIT_NONE:
            self._apply_gizmo_drag(dx, dy)
            self._last_pos = e.pos(); return

        if self._rmb_down:   # right = orbit
            self.rot_x += dy * 0.4; self.rot_y += dx * 0.4
        elif self._mmb_down: # middle = pan
            self.pan_x += dx * 0.5; self.pan_y -= dy * 0.5

        # Gizmo hover (only when not dragging orbit)
        if not self._rmb_down and not self._mmb_down:
            hit = self._gizmo_hit_test(e.x(), e.y())
            if hit != self.gizmo_hit:
                self.gizmo_hit = hit
                if not self._rmb_down:
                    self.setCursor(Qt.PointingHandCursor if hit != HIT_NONE else Qt.ArrowCursor)

        self._last_pos = e.pos(); self.update()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._lmb_down = False; self._gizmo_drag = HIT_NONE
        elif e.button() == Qt.RightButton:
            self._rmb_down = False
            # Short right-click (no drag) = pick part
            if hasattr(self, '_rmb_start'):
                dx = abs(e.x() - self._rmb_start.x())
                dy = abs(e.y() - self._rmb_start.y())
                if dx < 5 and dy < 5:  # barely moved = click not drag
                    self.makeCurrent()
                    pos, normal, part = self._raycast_parts(e.x(), e.y())
                    if part is not None:
                        self.part_right_clicked.emit(part)
        elif e.button() == Qt.MiddleButton:
            self._mmb_down = False
        self.setCursor(Qt.ArrowCursor)

    def wheelEvent(self, e):
        # Zoom toward mouse cursor position
        mx = e.x(); my = e.y()
        w = self.width(); h = max(self.height(), 1)

        # How far the mouse is from screen centre (in screen fraction)
        fx = (mx - w / 2.0) / (w / 2.0)
        fy = -(my - h / 2.0) / (h / 2.0)

        delta = e.angleDelta().y() * 0.4
        old_zoom = self.zoom
        self.zoom += delta

        # Scale factor: how much the view size changed
        # At zoom distance Z, a world unit takes (Z_new/Z_old) screen pixels
        if abs(old_zoom) > 1.0:
            scale = delta / old_zoom
            # Pan correction: move toward cursor by the zoom fraction
            self.pan_x += fx * scale * abs(old_zoom) * 0.5
            self.pan_y += fy * scale * abs(old_zoom) * 0.5

        self.update()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_R:
            self.rot_x = 25.0; self.rot_y = -35.0
            self.pan_x = 0; self.pan_y = 0
            # Reset orbit to scene centre
            if self.mesh_bounds is not None:
                self.orbit_centre = ((self.mesh_bounds[0] + self.mesh_bounds[1]) / 2.0).copy()
                sz = float(np.max(self.mesh_bounds[1] - self.mesh_bounds[0]))
                self.zoom = -sz * 2.5
            self.update()
        elif e.key() == Qt.Key_F:
            # F = focus on selected part (like Blender's numpad period)
            for part in self.parts:
                if part.id == self.selected_part_id:
                    try:
                        c = (part.mesh.bounds[0] + part.mesh.bounds[1]) / 2.0
                        sz = float(np.max(part.mesh.bounds[1] - part.mesh.bounds[0]))
                        self.orbit_centre = c.copy()
                        self.pan_x = 0; self.pan_y = 0
                        self.zoom = -sz * 3.0
                    except Exception:
                        pass
                    break
            self.update()
