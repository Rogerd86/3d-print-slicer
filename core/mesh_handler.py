"""
Core mesh operations: load, resize (per-axis independent), analyse bounds.
"""
import numpy as np
import trimesh
import os
from typing import Tuple, Optional


class MeshHandler:
    def __init__(self):
        self.mesh: Optional[trimesh.Trimesh] = None
        self.original_mesh: Optional[trimesh.Trimesh] = None
        self.file_path: Optional[str] = None
        self.scale_factor = 1.0
        self._original_dims: Tuple[float, float, float] = (0, 0, 0)
        self.target_x: float = 0.0
        self.target_y: float = 0.0
        self.target_z: float = 0.0

    def load(self, file_path: str) -> bool:
        try:
            loaded = trimesh.load(file_path)  # Don't force='mesh' — preserve Scene
            self.raw_loaded = loaded          # Keep original for colour split detection

            if isinstance(loaded, trimesh.Scene):
                meshes = [m for m in loaded.geometry.values()
                          if isinstance(m, trimesh.Trimesh) and len(m.faces) > 0]
                if not meshes:
                    return False
                combined = trimesh.util.concatenate(meshes)
            elif isinstance(loaded, trimesh.Trimesh):
                combined = loaded
            else:
                return False

            self.mesh = combined
            # Standard positioning: centre X/Y, sit on Z=0
            self._place_on_ground()
            self.original_mesh = self.mesh.copy()
            self.file_path = file_path
            self.scale_factor = 1.0
            e = self.mesh.extents
            self._original_dims = (float(e[0]), float(e[1]), float(e[2]))
            self.target_x = self._original_dims[0]
            self.target_y = self._original_dims[1]
            self.target_z = self._original_dims[2]
            return True
        except Exception as e:
            print(f"Load error: {e}")
            return False

    def apply_resize(self):
        """Apply current target_x/y/z independently per axis."""
        if self.mesh is None or self.original_mesh is None:
            return
        ox, oy, oz = self._original_dims
        if ox == 0 or oy == 0 or oz == 0:
            return
        fx = self.target_x / ox
        fy = self.target_y / oy
        fz = self.target_z / oz
        self.mesh = self.original_mesh.copy()
        scale_matrix = np.diag([fx, fy, fz, 1.0])
        self.mesh.apply_transform(scale_matrix)
        # Standard 3D printing convention: centre X/Y, sit on Z=0
        self._place_on_ground()

    def apply_uniform_resize(self, lock_axis: str = 'x'):
        if self.original_mesh is None:
            return
        ox, oy, oz = self._original_dims
        if lock_axis == 'x' and ox > 0:
            factor = self.target_x / ox
        elif lock_axis == 'y' and oy > 0:
            factor = self.target_y / oy
        else:
            factor = self.target_z / oz if oz > 0 else 1.0
        self.target_x = ox * factor
        self.target_y = oy * factor
        self.target_z = oz * factor
        self.apply_resize()

    def set_target_uniform(self, factor: float):
        ox, oy, oz = self._original_dims
        self.target_x = ox * factor
        self.target_y = oy * factor
        self.target_z = oz * factor

    def pct_change(self) -> Tuple[float, float, float]:
        ox, oy, oz = self._original_dims
        def _pct(cur, orig):
            return ((cur - orig) / orig * 100.0) if orig != 0 else 0.0
        return (_pct(self.target_x, ox), _pct(self.target_y, oy), _pct(self.target_z, oz))

    def original_dims(self) -> Tuple[float, float, float]:
        return self._original_dims

    def get_dimensions_mm(self) -> Tuple[float, float, float]:
        if self.mesh is None:
            return (0.0, 0.0, 0.0)
        e = self.mesh.extents
        return (float(e[0]), float(e[1]), float(e[2]))

    def get_bounds(self):
        if self.mesh is None:
            return None
        return self.mesh.bounds.copy()

    def get_vertex_array(self):
        if self.mesh is None:
            return None, None
        return self.mesh.vertices.copy(), self.mesh.faces.copy()

    def get_triangle_count(self) -> int:
        if self.mesh is None:
            return 0
        return len(self.mesh.faces)

    def is_watertight(self) -> bool:
        if self.mesh is None:
            return False
        return self.mesh.is_watertight

    def _place_on_ground(self):
        """Standard 3D printing position: centre X/Y, bottom at Z=0."""
        if self.mesh is None:
            return
        bounds = self.mesh.bounds
        centre_x = (bounds[0][0] + bounds[1][0]) / 2
        centre_y = (bounds[0][1] + bounds[1][1]) / 2
        bottom_z = bounds[0][2]
        self.mesh.apply_translation([-centre_x, -centre_y, -bottom_z])

    def center_mesh(self):
        if self.mesh is not None:
            self._place_on_ground()

    def subdivide(self, iterations: int = 1) -> tuple:
        """
        Subdivide mesh to increase polygon count and detail.
        Each iteration quadruples the face count.
        Returns (new_face_count, success).
        """
        if self.mesh is None:
            return 0, False
        try:
            from trimesh.remesh import subdivide
            verts = self.mesh.vertices
            faces = self.mesh.faces
            for _ in range(iterations):
                verts, faces = subdivide(verts, faces)
            self.mesh = trimesh.Trimesh(vertices=verts, faces=faces)
            self._place_on_ground()
            return len(self.mesh.faces), True
        except Exception as e:
            print(f"Subdivide error: {e}")
            return 0, False
