"""
boolean_ops.py
Wrapper for mesh boolean operations with automatic fallback.

Tries engines in order:
  1. manifold (best, most reliable)
  2. auto (trimesh default — blender if available, else basic)

This prevents the app from crashing if manifold3d isn't installed.
"""
import trimesh

# Detect best available engine at import time
_ENGINE = 'auto'
try:
    import manifold3d
    _ENGINE = 'manifold'
except ImportError:
    pass


def boolean_difference(meshes, **kwargs):
    """
    Subtract mesh B from mesh A.
    meshes: [mesh_a, mesh_b]
    Returns the result mesh, or mesh_a if the operation fails.
    """
    try:
        result = trimesh.boolean.difference(meshes, engine=_ENGINE, **kwargs)
        if result is not None and len(result.faces) > 0:
            return result
    except Exception as e:
        # Try fallback engine
        if _ENGINE != 'auto':
            try:
                result = trimesh.boolean.difference(meshes, engine='auto', **kwargs)
                if result is not None and len(result.faces) > 0:
                    return result
            except Exception:
                pass
        print(f"Boolean difference failed: {e}")
    return meshes[0]  # return original if all else fails


def get_engine():
    """Return the current boolean engine name."""
    return _ENGINE
