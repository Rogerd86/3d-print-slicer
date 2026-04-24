"""
Slicer core: cut mesh into printable chunks, with adjustable cut planes and joint generation.
"""
import numpy as np
import trimesh
from trimesh.intersections import slice_mesh_plane
from typing import List, Tuple, Dict, Optional
import copy


class CutPlane:
    """
    Represents a single cut plane.

    - auto_position: where the global default spacing algorithm placed it
    - manual_position: set by the user dragging/typing; overrides auto if not None
    - pinned: True once the user has manually moved this plane
    """
    def __init__(self, axis: str, auto_position: float):
        self.axis = axis                    # 'x', 'y', or 'z'
        self.auto_position = auto_position  # computed by default spacing
        self.manual_position: Optional[float] = None  # set by user
        self.pinned: bool = False           # True = user has manually placed this

    @property
    def effective_position(self) -> float:
        if self.pinned and self.manual_position is not None:
            return self.manual_position
        return self.auto_position

    def set_manual(self, pos: float):
        """User has explicitly placed this cut."""
        self.manual_position = pos
        self.pinned = True

    def unpin(self):
        """Reset to auto position."""
        self.manual_position = None
        self.pinned = False

    def normal(self):
        if self.axis == 'x':
            return np.array([1, 0, 0])
        elif self.axis == 'y':
            return np.array([0, 1, 0])
        else:
            return np.array([0, 0, 1])

    def origin(self):
        p = self.effective_position
        if self.axis == 'x':
            return np.array([p, 0, 0])
        elif self.axis == 'y':
            return np.array([0, p, 0])
        else:
            return np.array([0, 0, p])


class SlicedPart:
    """A single printable chunk after slicing."""
    def __init__(self, mesh: trimesh.Trimesh, grid_index: Tuple[int,int,int], label: str):
        self.mesh = mesh
        self.grid_index = grid_index   # (ix, iy, iz)
        self.label = label
        self.joint_type = "flat"       # flat, dovetail, dowel
        self.joint_faces = []          # which faces have joints: list of (axis, direction)


class Slicer:
    def __init__(self):
        self.source_mesh: Optional[trimesh.Trimesh] = None
        self.build_plate_x = 256.0   # mm  (P2S)
        self.build_plate_y = 256.0   # mm
        self.build_plate_z = 256.0   # mm
        self.default_cut_size = 150.0  # mm — global part size target
        self.cut_planes: List[CutPlane] = []
        self.parts: List[SlicedPart] = []
        self.joint_type = "flat"
        self.joint_size = 5.0
        self.tolerance = 0.3

    def set_mesh(self, mesh: trimesh.Trimesh):
        self.source_mesh = mesh.copy()

    def compute_cut_planes(self, preserve_pinned: bool = False):
        """
        Auto-generate cut planes spaced at default_cut_size along each axis.

        If preserve_pinned=True, any plane the user has manually pinned keeps
        its position; only unpinned planes are recalculated.
        """
        if self.source_mesh is None:
            return

        bounds = self.source_mesh.bounds
        mins = bounds[0]
        maxs = bounds[1]
        dims = maxs - mins

        # Collect existing pinned planes so we can preserve them
        pinned: Dict[str, List[CutPlane]] = {'x': [], 'y': [], 'z': []}
        if preserve_pinned:
            for p in self.cut_planes:
                if p.pinned:
                    pinned[p.axis].append(p)

        self.cut_planes = []

        for axis_idx, axis in enumerate(['x', 'y', 'z']):
            lo = float(mins[axis_idx])
            hi = float(maxs[axis_idx])
            span = hi - lo
            cut_size = self.default_cut_size

            # Number of cuts needed to divide span into cut_size chunks
            n_cuts = int(np.floor(span / cut_size))
            if n_cuts < 1:
                # Model fits in one part along this axis — no cuts needed
                continue

            # Even spacing using cut_size
            for i in range(1, n_cuts + 1):
                auto_pos = lo + i * cut_size
                if auto_pos >= hi:
                    break

                # Check if a pinned plane is already near this position
                existing = next(
                    (p for p in pinned[axis]
                     if abs(p.effective_position - auto_pos) < cut_size * 0.4),
                    None
                )
                if existing:
                    self.cut_planes.append(existing)
                else:
                    self.cut_planes.append(CutPlane(axis, auto_pos))

            # Re-add any pinned planes that weren't matched above
            for p in pinned[axis]:
                if p not in self.cut_planes:
                    self.cut_planes.append(p)

        # Sort each axis group by position
        self.cut_planes.sort(key=lambda p: (p.axis, p.effective_position))

    def move_cut_plane(self, index: int, new_position: float):
        """
        Manually move a single cut plane. Marks it as pinned.
        Other planes are unaffected.
        """
        if 0 <= index < len(self.cut_planes):
            self.cut_planes[index].set_manual(new_position)

    def unpin_cut_plane(self, index: int):
        """Reset a single cut plane back to auto position."""
        if 0 <= index < len(self.cut_planes):
            self.cut_planes[index].unpin()

    def unpin_all(self):
        """Reset all cut planes to auto positions."""
        for p in self.cut_planes:
            p.unpin()

    def add_cut_plane(self, axis: str, position: float) -> int:
        """Manually insert a new cut plane. Returns its index."""
        plane = CutPlane(axis, position)
        plane.set_manual(position)
        self.cut_planes.append(plane)
        self.cut_planes.sort(key=lambda p: (p.axis, p.effective_position))
        return self.cut_planes.index(plane)

    def remove_cut_plane(self, index: int):
        """Delete a cut plane."""
        if 0 <= index < len(self.cut_planes):
            self.cut_planes.pop(index)

    def get_cut_planes_by_axis(self, axis: str) -> List[CutPlane]:
        return [p for p in self.cut_planes if p.axis == axis]

    def get_grid_dimensions(self) -> Tuple[int,int,int]:
        """Returns number of parts along each axis."""
        nx = len(self.get_cut_planes_by_axis('x')) + 1
        ny = len(self.get_cut_planes_by_axis('y')) + 1
        nz = len(self.get_cut_planes_by_axis('z')) + 1
        return nx, ny, nz

    def slice_all(self) -> List[SlicedPart]:
        """Perform all cuts and return list of parts."""
        if self.source_mesh is None:
            return []

        self.parts = []

        # Sort cut planes per axis
        x_cuts = sorted(self.get_cut_planes_by_axis('x'), key=lambda p: p.effective_position)
        y_cuts = sorted(self.get_cut_planes_by_axis('y'), key=lambda p: p.effective_position)
        z_cuts = sorted(self.get_cut_planes_by_axis('z'), key=lambda p: p.effective_position)

        bounds = self.source_mesh.bounds
        mins = bounds[0]
        maxs = bounds[1]

        # Build list of slab boundaries per axis
        def boundaries(cuts, lo, hi):
            positions = [lo] + [c.effective_position for c in cuts] + [hi]
            return list(zip(positions[:-1], positions[1:]))

        x_ranges = boundaries(x_cuts, mins[0], maxs[0])
        y_ranges = boundaries(y_cuts, mins[1], maxs[1])
        z_ranges = boundaries(z_cuts, mins[2], maxs[2])

        part_num = 0
        for ix, (x0, x1) in enumerate(x_ranges):
            for iy, (y0, y1) in enumerate(y_ranges):
                for iz, (z0, z1) in enumerate(z_ranges):
                    part_mesh = self._cut_region(x0, x1, y0, y1, z0, z1)
                    if part_mesh is not None and len(part_mesh.faces) > 0:
                        part_num += 1
                        label = f"Part_{ix+1}_{iy+1}_{iz+1}"
                        part = SlicedPart(part_mesh, (ix, iy, iz), label)
                        part.joint_type = self.joint_type
                        self.parts.append(part)

        return self.parts

    def _cut_region(self, x0, x1, y0, y1, z0, z1) -> Optional[trimesh.Trimesh]:
        """Extract the mesh region within the given bounding box."""
        try:
            mesh = self.source_mesh.copy()

            # Slice with 6 planes (box)
            planes = [
                (np.array([1, 0, 0]),  np.array([x0, 0, 0])),   # left face
                (np.array([-1, 0, 0]), np.array([x1, 0, 0])),   # right face
                (np.array([0, 1, 0]),  np.array([0, y0, 0])),   # front
                (np.array([0, -1, 0]), np.array([0, y1, 0])),   # back
                (np.array([0, 0, 1]),  np.array([0, 0, z0])),   # bottom
                (np.array([0, 0, -1]),np.array([0, 0, z1])),   # top
            ]

            for normal, origin in planes:
                if len(mesh.faces) == 0:
                    return None
                mesh = slice_mesh_plane(mesh, normal, origin, cap=True)
                if mesh is None or len(mesh.faces) == 0:
                    return None

            return mesh
        except Exception as e:
            print(f"Cut error at ({x0:.1f},{y0:.1f},{z0:.1f}): {e}")
            return None

    def add_joints_to_part(self, part: SlicedPart):
        """
        Add joint geometry to cut faces of a part.
        This modifies the part mesh in place.
        Joints: dovetail slots, dowel holes, or flat (no change).
        """
        if part.joint_type == "flat":
            return

        bounds = part.mesh.bounds
        dims = bounds[1] - bounds[0]
        center = (bounds[0] + bounds[1]) / 2

        # Find which faces are cut faces (flat sections from slicing)
        # We detect them by finding large coplanar flat regions
        cut_faces = self._detect_cut_faces(part.mesh)

        for face_info in cut_faces:
            if part.joint_type == "dovetail":
                self._add_dovetail(part, face_info)
            elif part.joint_type == "dowel":
                self._add_dowel_holes(part, face_info)

    def _detect_cut_faces(self, mesh: trimesh.Trimesh) -> List[Dict]:
        """
        Detect flat planar regions (cut faces) on the mesh.
        Returns list of dicts with normal, center, size.
        """
        detected = []
        try:
            facets = mesh.facets
            facet_normals = mesh.facets_normal
            facet_areas = mesh.facets_area

            for i, (facet, normal, area) in enumerate(zip(facets, facet_normals, facet_areas)):
                # Only care about axis-aligned large flat faces (cut faces)
                abs_normal = np.abs(normal)
                is_axis_aligned = np.max(abs_normal) > 0.99
                if is_axis_aligned and area > 50:  # min 50 mm² to be a cut face
                    verts = mesh.vertices[mesh.faces[facet].flatten()]
                    center = verts.mean(axis=0)
                    detected.append({
                        'normal': normal,
                        'center': center,
                        'area': area,
                        'axis': np.argmax(abs_normal)
                    })
        except Exception:
            pass
        return detected

    def _add_dovetail(self, part: SlicedPart, face_info: Dict):
        """Add dovetail slot geometry. Placeholder — returns part unchanged for now."""
        # Full dovetail boolean operations require manifold geometry
        # This is a marker for the export step
        pass

    def _add_dowel_holes(self, part: SlicedPart, face_info: Dict):
        """Add dowel hole markers. Placeholder for export step."""
        pass

    def get_part_count(self) -> int:
        return len(self.parts)

    def get_estimated_part_sizes(self) -> List[Tuple[str, float, float, float]]:
        result = []
        for p in self.parts:
            e = p.mesh.extents
            result.append((p.label, float(e[0]), float(e[1]), float(e[2])))
        return result
