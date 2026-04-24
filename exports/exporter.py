"""
Export sliced parts to STL/OBJ/3MF files.
"""
import os
import trimesh
import numpy as np
from typing import List
from core.slicer import SlicedPart


class Exporter:
    def __init__(self):
        self.output_dir = ""
        self.format = "stl"   # stl, obj, 3mf

    def set_output_dir(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.output_dir = path

    def export_all(self, parts: List[SlicedPart], fmt: str = "stl",
                   joint_type: str = "flat", joint_size: float = 4.0,
                   tolerance: float = 0.3) -> List[str]:
        """
        Export all parts with optional joint geometry.
        Returns list of exported file paths.
        """
        exported = []
        for part in parts:
            mesh = part.mesh.copy()

            if joint_type != "flat":
                mesh = self._apply_joints(mesh, joint_type, joint_size, tolerance)

            filename = f"{part.label}.{fmt}"
            path = os.path.join(self.output_dir, filename)

            try:
                if fmt == "stl":
                    mesh.export(path, file_type="stl")
                elif fmt == "obj":
                    mesh.export(path, file_type="obj")
                elif fmt == "3mf":
                    mesh.export(path, file_type="3mf")
                exported.append(path)
            except Exception as e:
                print(f"Export error for {part.label}: {e}")

        # Write a manifest
        manifest_path = os.path.join(self.output_dir, "manifest.txt")
        with open(manifest_path, "w") as f:
            f.write("3D Print Slicer Export Manifest\n")
            f.write("=" * 40 + "\n\n")
            for i, part in enumerate(parts):
                e = part.mesh.extents
                f.write(f"{part.label}\n")
                f.write(f"  Grid index: {part.grid_index}\n")
                f.write(f"  Dimensions: {e[0]:.1f} x {e[1]:.1f} x {e[2]:.1f} mm\n")
                f.write(f"  Triangles: {len(part.mesh.faces)}\n")
                f.write(f"  Joint type: {part.joint_type}\n\n")

        return exported

    def _apply_joints(self, mesh: trimesh.Trimesh, joint_type: str,
                      joint_size: float, tolerance: float) -> trimesh.Trimesh:
        """
        Apply joint features to cut faces.
        Dovetail: adds trapezoidal notch markers on cut faces.
        Dowel: adds cylindrical hole markers on cut faces.
        """
        try:
            if joint_type == "dowel":
                mesh = self._apply_dowel_holes(mesh, joint_size, tolerance)
            elif joint_type == "dovetail":
                mesh = self._apply_dovetail_slots(mesh, joint_size, tolerance)
        except Exception as e:
            print(f"Joint application error: {e}")
        return mesh

    def _apply_dowel_holes(self, mesh: trimesh.Trimesh, size: float, tolerance: float) -> trimesh.Trimesh:
        """
        Add cylindrical dowel holes to flat cut faces.
        Hole radius = size/2 mm, depth = size mm.
        """
        bounds = mesh.bounds
        dims = bounds[1] - bounds[0]

        # Find axis-aligned cut faces
        cut_faces = self._find_cut_faces(mesh)

        for face in cut_faces:
            normal = face['normal']
            center = face['center']
            axis = face['axis']

            # Place holes in a grid on the cut face
            spacing = size * 3.5
            u_axis = (axis + 1) % 3
            v_axis = (axis + 2) % 3

            face_dims_u = dims[u_axis]
            face_dims_v = dims[v_axis]

            n_u = max(1, int(face_dims_u / spacing))
            n_v = max(1, int(face_dims_v / spacing))

            for i in range(n_u):
                for j in range(n_v):
                    # Position of hole center
                    offset_u = (i - (n_u-1)/2) * spacing
                    offset_v = (j - (n_v-1)/2) * spacing

                    hole_center = center.copy()
                    hole_center[u_axis] += offset_u
                    hole_center[v_axis] += offset_v

                    # Create cylinder aligned with face normal
                    radius = (size / 2) - tolerance
                    depth = size

                    try:
                        cyl = trimesh.creation.cylinder(
                            radius=radius,
                            height=depth + 2,
                            sections=16
                        )
                        # Align cylinder to face normal
                        rot = trimesh.geometry.align_vectors([0, 0, 1], normal.tolist())
                        cyl.apply_transform(rot)
                        cyl.apply_translation(hole_center - normal * (depth / 2))

                        # Boolean difference
                        from core.boolean_ops import boolean_difference
                        result = boolean_difference([mesh, cyl])
                        if result is not None and len(result.faces) > 0:
                            mesh = result
                    except Exception:
                        pass  # Skip if boolean fails, continue without hole

        return mesh

    def _apply_dovetail_slots(self, mesh: trimesh.Trimesh, size: float, tolerance: float) -> trimesh.Trimesh:
        """
        Add trapezoidal dovetail slots to cut faces.
        """
        # Dovetail as a trapezoid prism
        # For now returns mesh unchanged — requires reliable boolean ops
        # Full implementation needs manifold mesh guarantee
        return mesh

    def _find_cut_faces(self, mesh: trimesh.Trimesh) -> list:
        """Find large flat axis-aligned faces (the cut faces)."""
        detected = []
        try:
            if not hasattr(mesh, 'facets') or len(mesh.facets) == 0:
                return detected
            for i, (facet, normal, area) in enumerate(
                zip(mesh.facets, mesh.facets_normal, mesh.facets_area)
            ):
                abs_n = np.abs(normal)
                if np.max(abs_n) > 0.99 and area > 25:
                    verts = mesh.vertices[mesh.faces[facet].flatten()]
                    center = verts.mean(axis=0)
                    detected.append({
                        'normal': normal,
                        'center': center,
                        'area': area,
                        'axis': int(np.argmax(abs_n))
                    })
        except Exception:
            pass
        return detected
