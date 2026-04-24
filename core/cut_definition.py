"""
CutDefinition: describes a single cut operation.

Modes:
  'full'    — full-width plane, axis-aligned
  'free'    — full-width plane at arbitrary angle (uses rot_u, rot_v)
  'section' — rectangular region cut
"""
import numpy as np
from typing import Optional, Tuple
import trimesh
from trimesh.intersections import slice_mesh_plane


class CutDefinition:
    def __init__(self, mode: str = 'full'):
        self.mode = mode

        # Position — world-space point the plane passes through
        self.position = np.array([0.0, 0.0, 0.0])

        # Axis for 'full' / 'free' mode: 'x', 'y', 'z'
        self.axis = 'x'

        # Rotation offsets in degrees, applied relative to cut plane's own axes
        # rot_u = tilt around the plane's U axis (perpendicular to normal, horizontal)
        # rot_v = tilt around the plane's V axis (perpendicular to normal, vertical)
        # These are intuitive: dragging any ring tilts the cut plane visibly
        self.rot_u = 0.0   # tilt left/right
        self.rot_v = 0.0   # tilt up/down
        self.rot_n = 0.0   # spin around own normal (less common)

        # Section cut bounds
        self.section_w = 100.0
        self.section_h = 100.0

        # Groove cut parameters
        self.groove_teeth = 6      # number of teeth
        self.groove_depth = 4.0    # mm depth of each tooth
        self.groove_width = 8.0    # mm width of each tooth

        # Preview state
        self.pinned = False

    @property
    def effective_position(self) -> float:
        ax_idx = {'x': 0, 'y': 1, 'z': 2}.get(self.axis, 0)
        return float(self.position[ax_idx])

    def get_normal(self) -> np.ndarray:
        """Compute cut plane normal. Always works regardless of axis."""
        # Base normal from axis
        base = {'x': np.array([1.,0,0]), 'y': np.array([0,1.,0]),
                'z': np.array([0,0,1.])}.get(self.axis, np.array([1.,0,0]))

        if self.mode == 'full':
            return base

        # For free/section: build plane axes and apply rotations
        n = base.copy()
        helper = np.array([0,0,1.0]) if abs(n[2]) < 0.9 else np.array([0,1.,0])
        u = np.cross(n, helper); u /= np.linalg.norm(u)
        v = np.cross(n, u);     v /= np.linalg.norm(v)

        # Apply rot_u: rotate normal around U axis (tilt left/right)
        ru = np.radians(self.rot_u)
        n = (n * np.cos(ru)
             + np.cross(u, n) * np.sin(ru)
             + u * np.dot(u, n) * (1 - np.cos(ru)))
        n /= np.linalg.norm(n)

        # Apply rot_v: rotate around V axis (tilt up/down)
        rv = np.radians(self.rot_v)
        helper2 = np.array([0,0,1.0]) if abs(n[2]) < 0.9 else np.array([0,1.,0])
        u2 = np.cross(n, helper2); u2 /= max(np.linalg.norm(u2), 1e-9)
        v2 = np.cross(n, u2);     v2 /= max(np.linalg.norm(v2), 1e-9)
        n = (n * np.cos(rv)
             + np.cross(v2, n) * np.sin(rv)
             + v2 * np.dot(v2, n) * (1 - np.cos(rv)))
        n /= max(np.linalg.norm(n), 1e-9)

        return n

    def get_plane_axes(self) -> tuple:
        """Return the U and V axes of the cut plane (for section cut rectangle)."""
        n = self.get_normal()
        helper = np.array([0,0,1.0]) if abs(n[2]) < 0.9 else np.array([0,1.,0])
        u = np.cross(n, helper); u /= max(np.linalg.norm(u), 1e-9)
        v = np.cross(n, u);     v /= max(np.linalg.norm(v), 1e-9)

        # Apply rot_n: spin the U/V axes around the normal
        if abs(self.rot_n) > 0.01:
            rn = np.radians(self.rot_n)
            cos_rn, sin_rn = np.cos(rn), np.sin(rn)
            u_new = u * cos_rn + v * sin_rn
            v_new = -u * sin_rn + v * cos_rn
            u, v = u_new, v_new

        return u, v

    def get_origin(self) -> np.ndarray:
        return self.position.copy()

    def apply_to_mesh(self, mesh: trimesh.Trimesh) -> Tuple[
            Optional[trimesh.Trimesh], Optional[trimesh.Trimesh]]:
        try:
            normal = self.get_normal()
            origin = self.get_origin()
            if self.mode in ('full', 'free'):
                return self._full_cut(mesh, normal, origin)
            elif self.mode == 'groove':
                return self._groove_cut(mesh, normal, origin)
            elif self.mode == 'section':
                return self._section_cut(mesh, normal, origin)
        except Exception as e:
            print(f"Cut error: {e}")
            return None, None

    def _full_cut(self, mesh, normal, origin):
        try:
            side_a = slice_mesh_plane(mesh, normal, origin, cap=True)
            side_b = slice_mesh_plane(mesh, -normal, origin, cap=True)
            side_a = side_a if (side_a is not None and len(side_a.faces) > 0) else None
            side_b = side_b if (side_b is not None and len(side_b.faces) > 0) else None
            return side_a, side_b
        except Exception as e:
            print(f"Full cut error: {e}")
            return None, None

    def _section_cut(self, mesh, normal, origin):
        try:
            # Use get_plane_axes() which includes rot_n spin
            u, v = self.get_plane_axes()
            hw = self.section_w / 2.0; hh = self.section_h / 2.0
            clip_planes = [
                ( u, origin + u*hw), (-u, origin - u*hw),
                ( v, origin + v*hh), (-v, origin - v*hh),
            ]
            section = mesh.copy()
            for cn, co in clip_planes:
                if section is not None and len(section.faces) > 0:
                    section = slice_mesh_plane(section, cn, co, cap=True)
            if section is None or len(section.faces) == 0:
                return mesh.copy(), None
            sec_a = slice_mesh_plane(section,  normal, origin, cap=True)
            sec_b = slice_mesh_plane(section, -normal, origin, cap=True)
            if sec_a is not None and len(sec_a.faces) > 0:
                side_a = trimesh.util.concatenate([mesh.copy(), sec_a])
            else:
                side_a = mesh.copy()
            side_b = sec_b if (sec_b is not None and len(sec_b.faces) > 0) else None
            return side_a, side_b
        except Exception as e:
            print(f"Section cut error: {e}")
            return mesh.copy(), None

    def _groove_cut(self, mesh, normal, origin):
        """
        Groove/zigzag cut — creates interlocking stepped teeth using
        multiple slice operations (no booleans needed).

        Strategy: alternate slicing at two offsets along the normal.
        Teeth are formed by slicing strips at alternating depths.
        Side A gets even teeth, side B gets odd teeth.
        """
        try:
            n = np.array(normal, dtype=float)
            n /= max(np.linalg.norm(n), 1e-9)
            helper = np.array([0, 0, 1.0]) if abs(n[2]) < 0.9 else np.array([0, 1.0, 0])
            u = np.cross(n, helper); u /= np.linalg.norm(u)

            teeth_count = max(2, self.groove_teeth)
            depth = self.groove_depth  # how far teeth protrude
            width = self.groove_width  # width of each tooth

            # Compute the mesh span along u-axis for spacing
            bounds = mesh.bounds
            face_span = float(np.max(bounds[1] - bounds[0]))
            spacing = face_span / teeth_count

            # Create two offset cut planes
            origin_fwd = origin + n * (depth / 2)   # forward plane
            origin_bak = origin - n * (depth / 2)   # backward plane

            # Start with a copy of the mesh
            # We'll build side_a and side_b by slicing strips
            working = mesh.copy()

            # Slice into strips along u-axis, then assign each strip
            # to the forward or backward plane based on alternation
            strip_parts_a = []
            strip_parts_b = []

            for i in range(teeth_count):
                strip_lo = -face_span / 2 + i * spacing
                strip_hi = strip_lo + spacing

                # Cut out this strip from the mesh
                strip = working.copy()
                # Clip to strip bounds
                strip = slice_mesh_plane(strip, u, origin + u * strip_lo, cap=True)
                if strip is None or len(strip.faces) == 0: continue
                strip = slice_mesh_plane(strip, -u, origin + u * strip_hi, cap=True)
                if strip is None or len(strip.faces) == 0: continue

                # Even strips: cut at forward plane (teeth go into B)
                # Odd strips: cut at backward plane (teeth go into A)
                if i % 2 == 0:
                    part_a = slice_mesh_plane(strip, n, origin_fwd, cap=True)
                    part_b = slice_mesh_plane(strip, -n, origin_fwd, cap=True)
                else:
                    part_a = slice_mesh_plane(strip, n, origin_bak, cap=True)
                    part_b = slice_mesh_plane(strip, -n, origin_bak, cap=True)

                if part_a is not None and len(part_a.faces) > 0:
                    strip_parts_a.append(part_a)
                if part_b is not None and len(part_b.faces) > 0:
                    strip_parts_b.append(part_b)

            # Combine all strips for each side
            if strip_parts_a:
                side_a = trimesh.util.concatenate(strip_parts_a)
            else:
                side_a = None
            if strip_parts_b:
                side_b = trimesh.util.concatenate(strip_parts_b)
            else:
                side_b = None

            # Validate
            if side_a is not None and len(side_a.faces) == 0: side_a = None
            if side_b is not None and len(side_b.faces) == 0: side_b = None

            if side_a is None and side_b is None:
                print("Groove cut produced no geometry, falling back to flat cut")
                return self._full_cut(mesh, normal, origin)

            return side_a, side_b

        except Exception as e:
            print(f"Groove cut error: {e}, falling back to flat cut")
            return self._full_cut(mesh, normal, origin)

    def describe(self) -> str:
        pos_val = self.effective_position
        if self.mode == 'full':
            return f"Full {self.axis.upper()} cut @ {pos_val:.1f}mm"
        elif self.mode == 'free':
            return f"Free {self.axis.upper()} cut @ {pos_val:.1f}mm (tilt U:{self.rot_u:.0f}° V:{self.rot_v:.0f}°)"
        elif self.mode == 'groove':
            return f"Groove {self.axis.upper()} cut @ {pos_val:.1f}mm ({self.groove_teeth} teeth, {self.groove_depth:.0f}mm deep)"
        else:
            return f"Section {self.section_w:.0f}×{self.section_h:.0f}mm @ {pos_val:.1f}mm"
