"""
PartTree: manages a hierarchy of mesh parts produced by sequential cuts.

Each Part:
  - Has a mesh
  - Has a display label and colour index
  - Can be split into two children by applying a CutDefinition
  - Leaf nodes are the printable parts (no children)

The tree supports full undo by keeping the cut history.
"""
import numpy as np
import trimesh
from typing import Optional, List, Tuple
from core.cut_definition import CutDefinition
import uuid


class Part:
    """A single piece of the model — either the root or a result of cutting."""

    # 50 colours using golden-angle hue spacing — matches viewport palette
    # Consecutive colours are always visually distinct
    @staticmethod
    def _generate_colors():
        import colorsys
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
    _color_counter = 0

    def __init__(self, mesh: trimesh.Trimesh, label: str,
                 parent: Optional['Part'] = None):
        self.mesh = mesh
        self.label = label
        self.parent: Optional[Part] = parent
        self.children: List[Part] = []
        self.cut_used: Optional[CutDefinition] = None  # cut that produced this part
        self.id = str(uuid.uuid4())[:8]
        self.color_idx = Part._color_counter % len(Part.COLORS)
        Part._color_counter += 1
        self.selected = False
        self.visible = True   # toggle show/hide in viewport

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def color(self) -> Tuple[float, float, float]:
        return Part.COLORS[self.color_idx % len(Part.COLORS)]

    def get_dimensions(self) -> Tuple[float, float, float]:
        e = self.mesh.extents
        return (float(e[0]), float(e[1]), float(e[2]))

    def get_bounds(self):
        return self.mesh.bounds.copy()

    def split(self, cut: CutDefinition) -> Tuple[
            Optional['Part'], Optional['Part']]:
        """
        Apply a cut to this part. Returns (child_A, child_B).
        Both children are added to self.children if successful.
        Returns (None, None) if the cut fails or misses.
        """
        mesh_a, mesh_b = cut.apply_to_mesh(self.mesh)

        if mesh_a is None and mesh_b is None:
            return None, None

        children = []
        for i, m in enumerate([mesh_a, mesh_b]):
            if m is not None and len(m.faces) > 0:
                suffix = 'A' if i == 0 else 'B'
                child = Part(m, f"{self.label}-{suffix}", parent=self)
                child.cut_used = cut
                children.append(child)

        if len(children) == 0:
            return None, None

        self.children = children
        return (children[0] if len(children) > 0 else None,
                children[1] if len(children) > 1 else None)

    def undo_split(self):
        """Remove all children — restores this part to a leaf."""
        self.children = []

    def all_leaves(self) -> List['Part']:
        """Return all leaf nodes in the subtree rooted here."""
        if self.is_leaf:
            return [self]
        result = []
        for child in self.children:
            result.extend(child.all_leaves())
        return result

    def depth(self) -> int:
        if self.parent is None:
            return 0
        return self.parent.depth() + 1

    def __repr__(self):
        return f"Part({self.label}, {len(self.mesh.faces)} faces)"


class PartTree:
    """
    Manages the full hierarchy of parts from initial import through all cuts.
    Provides undo by storing the full cut action history.
    """

    def __init__(self):
        self.root: Optional[Part] = None
        self._undo_stack: List[dict] = []   # list of undo records
        self.selected_part: Optional[Part] = None

    def load_mesh(self, mesh: trimesh.Trimesh, label: str = "Body"):
        """Initialise the tree with a single root part."""
        Part._color_counter = 0
        self.root = Part(mesh.copy(), label)
        self.selected_part = self.root
        self._undo_stack = []

    def push_mesh_snapshot(self, parts: list, label: str = "operation"):
        """
        Save mesh snapshots of the given parts before a destructive operation.
        Call this BEFORE smoothing, hollowing, repair, or any mesh modification.
        These are restored on undo().
        """
        snapshots = []
        for part in parts:
            snapshots.append({
                'part_id': part.id,
                'part_ref': part,       # direct reference so we can restore
                'mesh_copy': part.mesh.copy(),
                'label_copy': part.label,
            })
        self._undo_stack.append({'action': 'mesh_edit', 'snapshots': snapshots,
                                  'label': label})

    def apply_cut(self, part: Part, cut: CutDefinition) -> Tuple[
            Optional[Part], Optional[Part]]:
        """
        Apply a cut to a specific part. Pushes to undo stack.
        """
        if not part.is_leaf:
            return None, None

        child_a, child_b = part.split(cut)

        if child_a is None and child_b is None:
            return None, None

        self._undo_stack.append({'action': 'split', 'part': part})

        if child_a is not None:
            self.selected_part = child_a

        return child_a, child_b

    def undo(self) -> Optional[str]:
        """
        Undo the last operation.
        Returns description string of what was undone, or None if nothing to undo.
        """
        if not self._undo_stack:
            return None
        record = self._undo_stack.pop()
        action = record.get('action', '')

        if action == 'split':
            part = record['part']
            part.undo_split()
            self.selected_part = part
            return f"Undone cut on {part.label}"

        elif action == 'mesh_edit':
            label = record.get('label', 'operation')
            for snap in record.get('snapshots', []):
                part = snap['part_ref']
                part.mesh = snap['mesh_copy'].copy()
                part.label = snap['label_copy']
            n = len(record.get('snapshots', []))
            return f"Undone {label} ({n} part{'s' if n>1 else ''})"

        return None

    def can_undo(self) -> bool:
        return len(self._undo_stack) > 0

    def undo_description(self) -> str:
        """What will be undone next."""
        if not self._undo_stack:
            return ""
        record = self._undo_stack[-1]
        action = record.get('action', '')
        if action == 'split':
            return f"Undo cut on {record['part'].label}"
        elif action == 'mesh_edit':
            label = record.get('label', 'edit')
            n = len(record.get('snapshots', []))
            return f"Undo {label} ({n} part{'s' if n>1 else ''})"
        return "Undo"

    def get_all_leaves(self) -> List[Part]:
        """All current printable parts (leaf nodes)."""
        if self.root is None:
            return []
        return self.root.all_leaves()

    def get_all_parts(self) -> List[Part]:
        """All parts in the tree (BFS order)."""
        if self.root is None:
            return []
        result = []
        queue = [self.root]
        while queue:
            p = queue.pop(0)
            result.append(p)
            queue.extend(p.children)
        return result

    def select(self, part: Part):
        self.selected_part = part

    def set_visible(self, part: Part, visible: bool):
        """Toggle visibility of a part and all its descendants."""
        part.visible = visible
        for child in part.children:
            self.set_visible(child, visible)

    def show_all(self):
        for p in self.get_all_parts():
            p.visible = True

    def hide_all_except(self, keep_part: Part):
        """Hide all leaf parts except keep_part."""
        for p in self.get_all_leaves():
            p.visible = (p.id == keep_part.id)

    def get_visible_leaves(self) -> list:
        return [p for p in self.get_all_leaves() if p.visible]

    def find_by_id(self, part_id: str) -> Optional[Part]:
        for p in self.get_all_parts():
            if p.id == part_id:
                return p
        return None

    def apply_auto_cuts(self, cut_size: float, axis_order: str = 'xyz'):
        """
        Auto-generate a grid of full cuts on the root at default spacing.
        Correctly slices each leaf multiple times per axis until it fits cut_size.
        """
        if self.root is None:
            return

        # Reset to just the root
        self.root.children = []
        self._undo_stack = []
        Part._color_counter = 1

        for axis in axis_order:
            ax_idx = {'x': 0, 'y': 1, 'z': 2}[axis]

            # Keep slicing until all leaves fit within cut_size on this axis
            changed = True
            max_passes = 50  # safety limit
            passes = 0
            while changed and passes < max_passes:
                changed = False
                passes += 1
                for leaf in self.root.all_leaves():
                    b = leaf.mesh.bounds
                    lo_l = float(b[0][ax_idx])
                    hi_l = float(b[1][ax_idx])
                    span = hi_l - lo_l
                    if span <= cut_size * 1.05:  # fits — skip
                        continue
                    # Cut at lo + cut_size
                    cut_pos = lo_l + cut_size
                    cut = CutDefinition('full')
                    cut.axis = axis
                    pt = np.zeros(3)
                    pt[ax_idx] = cut_pos
                    cut.position = pt.copy()
                    child_a, child_b = leaf.split(cut)
                    if child_a is not None or child_b is not None:
                        changed = True

        self._undo_stack = []

        # Renumber leaves: Body-001, Body-002 instead of Body-A-B-A
        base = self.root.label
        for i, leaf in enumerate(self.root.all_leaves()):
            leaf.label = f"{base}-{i+1:03d}"
