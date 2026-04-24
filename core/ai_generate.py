"""
ai_generate.py
AI-powered 3D model generation from photos and text.

Supports four image-to-3D backends:
  1. TRELLIS 2 (Microsoft) — best quality, needs 14-16GB VRAM (FP16), MIT license
  2. SAM 3D Objects (Meta) — excellent quality, needs 16GB VRAM (FP16)
  3. PartCrafter (NeurIPS 2025) — generates pre-separated parts, needs 12GB VRAM
  4. TripoSR (Stability AI) — fastest, needs 6-8GB VRAM

Plus text-to-3D:
  5. OpenSCAD — parametric 3D from text description (CPU only, no GPU needed)

All run 100% locally. No cloud, no cost, no limits.
"""
import os
import sys
import subprocess
import json
import tempfile
import shutil
from typing import Optional, Tuple, List, Dict
from pathlib import Path


# Where AI backends are installed
AI_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ai_backends')


def check_gpu() -> Dict:
    """Check available GPU and VRAM."""
    result = {'has_gpu': False, 'gpu_name': '', 'vram_total_gb': 0, 'vram_free_gb': 0}
    try:
        import torch
        if torch.cuda.is_available():
            result['has_gpu'] = True
            result['gpu_name'] = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            result['vram_total_gb'] = round(total, 1)
            # Free VRAM
            free = (torch.cuda.get_device_properties(0).total_mem -
                    torch.cuda.memory_allocated(0)) / (1024**3)
            result['vram_free_gb'] = round(free, 1)
    except ImportError:
        # PyTorch not installed — try nvidia-smi
        try:
            out = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=name,memory.total,memory.free',
                 '--format=csv,noheader,nounits'],
                timeout=5).decode().strip()
            parts = out.split(',')
            if len(parts) >= 3:
                result['has_gpu'] = True
                result['gpu_name'] = parts[0].strip()
                result['vram_total_gb'] = round(float(parts[1].strip()) / 1024, 1)
                result['vram_free_gb'] = round(float(parts[2].strip()) / 1024, 1)
        except Exception:
            pass
    return result


def check_backends() -> Dict[str, bool]:
    """Check which AI backends are installed and ready."""
    status = {
        'trellis2': False,
        'sam3d': False,
        'partcrafter': False,
        'triposr': False,
        'openscad': False,
        'gpu_available': False,
        'vram_gb': 0,
        'recommendation': '',
    }

    gpu = check_gpu()
    status['gpu_available'] = gpu['has_gpu']
    status['vram_gb'] = gpu['vram_total_gb']

    # Check TRELLIS 2 (Microsoft)
    trellis_path = os.path.join(AI_DIR, 'TRELLIS.2')
    if os.path.isdir(trellis_path):
        status['trellis2'] = True

    # Check SAM 3D Objects
    sam3d_path = os.path.join(AI_DIR, 'sam-3d-objects')
    if os.path.isdir(sam3d_path):
        status['sam3d'] = True

    # Check PartCrafter
    partcrafter_path = os.path.join(AI_DIR, 'PartCrafter')
    if os.path.isdir(partcrafter_path):
        status['partcrafter'] = True

    # Check TripoSR
    triposr_path = os.path.join(AI_DIR, 'TripoSR')
    if os.path.isdir(triposr_path):
        status['triposr'] = True

    # Check OpenSCAD
    try:
        result = subprocess.run(['openscad', '--version'], capture_output=True, timeout=5)
        if result.returncode == 0:
            status['openscad'] = True
    except Exception:
        # Check common install paths on Windows
        for p in [r'C:\Program Files\OpenSCAD\openscad.exe',
                  r'C:\Program Files (x86)\OpenSCAD\openscad.exe']:
            if os.path.exists(p):
                status['openscad'] = True
                break

    # Recommendation — TRELLIS 2 is best quality, then SAM 3D, then TripoSR
    if not gpu['has_gpu']:
        status['recommendation'] = 'No GPU detected — AI generation requires an NVIDIA GPU'
    elif gpu['vram_total_gb'] >= 14 and status['trellis2']:
        status['recommendation'] = f"TRELLIS 2 (FP16) — best quality on your {gpu['gpu_name']}"
    elif gpu['vram_total_gb'] >= 16 and status['sam3d']:
        status['recommendation'] = f"SAM 3D Objects (FP16) — excellent quality on your {gpu['gpu_name']}"
    elif status['triposr']:
        status['recommendation'] = f"TripoSR — fast and efficient on your {gpu['gpu_name']}"
    elif gpu['vram_total_gb'] >= 8:
        status['recommendation'] = 'Install TripoSR for AI generation (run setup_ai_backends.bat)'
    else:
        status['recommendation'] = f"GPU has {gpu['vram_total_gb']}GB VRAM — minimum 8GB needed"

    return status


def generate_mesh_triposr(image_paths: List[str],
                           output_dir: str,
                           resolution: int = 256) -> Tuple[Optional[str], str]:
    """
    Generate a 3D mesh from image(s) using TripoSR.

    image_paths: list of 1-4 image file paths
    output_dir: where to save the output OBJ
    resolution: marching cubes resolution (128/256/512)

    Returns (obj_path, status_message) or (None, error_message)
    """
    triposr_path = os.path.join(AI_DIR, 'TripoSR')
    if not os.path.isdir(triposr_path):
        return None, "TripoSR not installed. Run setup_ai_backends.bat first."

    try:
        os.makedirs(output_dir, exist_ok=True)
        image = image_paths[0]  # TripoSR uses single image primarily
        output_name = os.path.splitext(os.path.basename(image))[0]
        output_obj = os.path.join(output_dir, f"{output_name}.obj")

        # Run TripoSR inference
        cmd = [
            sys.executable, os.path.join(triposr_path, 'run.py'),
            image,
            '--output-dir', output_dir,
            '--mc-resolution', str(resolution),
            '--format', 'obj',
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120,
                                cwd=triposr_path)

        if result.returncode != 0:
            return None, f"TripoSR error: {result.stderr[:300]}"

        # Find the output file
        for f in os.listdir(output_dir):
            if f.endswith('.obj'):
                return os.path.join(output_dir, f), "TripoSR generation complete"

        return None, "TripoSR finished but no OBJ file found"

    except subprocess.TimeoutExpired:
        return None, "TripoSR timed out (>120 seconds)"
    except Exception as e:
        return None, f"TripoSR error: {e}"


def generate_mesh_sam3d(image_path: str,
                         mask_path: Optional[str],
                         output_dir: str) -> Tuple[Optional[str], str]:
    """
    Generate a 3D mesh from an image using SAM 3D Objects (FP16).

    image_path: path to input photo
    mask_path: optional mask image (white = object, black = background)
               If None, the full image is used
    output_dir: where to save output

    Returns (obj_path, status_message) or (None, error_message)
    """
    sam3d_path = os.path.join(AI_DIR, 'sam-3d-objects')
    if not os.path.isdir(sam3d_path):
        return None, "SAM 3D Objects not installed. Run setup_ai_backends.bat first."

    try:
        os.makedirs(output_dir, exist_ok=True)

        # Build inference script that uses FP16
        script = f"""
import sys
sys.path.insert(0, {repr(sam3d_path.replace(os.sep, "/"))})
import torch
from sam3d.inference import Inference

# Load model in FP16 to fit in 16GB VRAM
model = Inference(device='cuda', dtype=torch.float16)
result = model.infer(
    image_path={repr(image_path.replace(os.sep, "/"))},
    {'mask_path=' + repr(mask_path.replace(os.sep, "/")) + ',' if mask_path else ''}
)
# Export as OBJ
output = {repr(os.path.join(output_dir, "sam3d_output.obj").replace(os.sep, "/"))}
result.export_mesh(output)
print(f'DONE:{{output}}')
"""
        script_path = os.path.join(output_dir, '_sam3d_run.py')
        with open(script_path, 'w') as f:
            f.write(script)

        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True, text=True, timeout=180,
            env={**os.environ, 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'})

        if result.returncode != 0:
            return None, f"SAM 3D error: {result.stderr[:300]}"

        # Find output
        for line in result.stdout.split('\n'):
            if line.startswith('DONE:'):
                obj_path = line[5:].strip()
                if os.path.exists(obj_path):
                    return obj_path, "SAM 3D Objects generation complete (FP16)"

        return None, "SAM 3D finished but no output found"

    except subprocess.TimeoutExpired:
        return None, "SAM 3D timed out (>180 seconds)"
    except Exception as e:
        return None, f"SAM 3D error: {e}"


def generate_mesh_trellis2(image_path: str,
                            output_dir: str,
                            resolution: int = 512) -> Tuple[Optional[str], str]:
    """
    Generate a 3D mesh from an image using TRELLIS 2 (Microsoft).
    Best quality available. Runs in FP16 to fit 16GB VRAM.

    image_path: path to input photo
    output_dir: where to save output
    resolution: mesh resolution (256/512/1024). 512 fits 16GB VRAM in FP16.

    Returns (obj_path, status_message) or (None, error_message)
    """
    trellis_path = os.path.join(AI_DIR, 'TRELLIS.2')
    if not os.path.isdir(trellis_path):
        return None, "TRELLIS 2 not installed. Run setup_ai_backends.bat first."

    try:
        os.makedirs(output_dir, exist_ok=True)
        output_name = os.path.splitext(os.path.basename(image_path))[0]

        # Build inference script for TRELLIS 2
        # Uses FP16 and low-vram mode to fit on 16GB cards
        script = f"""
import sys, os
sys.path.insert(0, {repr(trellis_path.replace(os.sep, "/"))})
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
torch.set_default_dtype(torch.float16)

from trellis.pipelines import TrellisImageTo3DPipeline

# Load pipeline in FP16
pipeline = TrellisImageTo3DPipeline.from_pretrained(
    "microsoft/TRELLIS-image-large",
    torch_dtype=torch.float16
)
pipeline = pipeline.to("cuda")

# Run inference
from PIL import Image
image = Image.open({repr(image_path.replace(os.sep, "/"))})
outputs = pipeline(
    image,
    seed=42,
    sparse_structure_sampler_params={{
        "steps": 12,
    }},
    slat_sampler_params={{
        "steps": 12,
    }},
)

# Export mesh as OBJ
output_path = {repr(os.path.join(output_dir, output_name + ".obj").replace(os.sep, "/"))}
mesh = outputs['mesh'][0]
mesh.export(output_path)
print(f'DONE:{{output_path}}')

# Free VRAM immediately
del pipeline, outputs
torch.cuda.empty_cache()
"""
        script_path = os.path.join(output_dir, '_trellis2_run.py')
        with open(script_path, 'w') as f:
            f.write(script)

        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True, text=True, timeout=300,
            env={**os.environ, 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'})

        if result.returncode != 0:
            return None, f"TRELLIS 2 error: {result.stderr[:500]}"

        # Find output
        for line in result.stdout.split('\n'):
            if line.startswith('DONE:'):
                obj_path = line[5:].strip()
                if os.path.exists(obj_path):
                    return obj_path, "TRELLIS 2 generation complete (FP16, best quality)"

        # Fallback: search output dir for any OBJ
        for f in os.listdir(output_dir):
            if f.endswith('.obj'):
                return os.path.join(output_dir, f), "TRELLIS 2 generation complete"

        return None, "TRELLIS 2 finished but no output found"

    except subprocess.TimeoutExpired:
        return None, "TRELLIS 2 timed out (>300 seconds)"
    except Exception as e:
        return None, f"TRELLIS 2 error: {e}"


def generate_mesh_partcrafter(image_path: str,
                               output_dir: str) -> Tuple[Optional[str], str]:
    """
    Generate 3D mesh with pre-separated parts using PartCrafter.
    Each part of the object (body, wheels, etc.) comes as a separate mesh.
    Needs ~12GB VRAM.

    Returns (obj_path, status_message) — exports a single combined OBJ
    """
    pc_path = os.path.join(AI_DIR, 'PartCrafter')
    if not os.path.isdir(pc_path):
        return None, "PartCrafter not installed. Run setup_ai_backends.bat first."

    try:
        os.makedirs(output_dir, exist_ok=True)
        output_name = os.path.splitext(os.path.basename(image_path))[0]

        script = f"""
import sys, os
sys.path.insert(0, {repr(pc_path.replace(os.sep, "/"))})
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
from inference import PartCrafterInference

model = PartCrafterInference(device='cuda', dtype=torch.float16)
results = model.generate({repr(image_path.replace(os.sep, "/"))})

# Export each part and combined mesh
output_dir = {repr(output_dir.replace(os.sep, "/"))}
combined_parts = []
for i, part_mesh in enumerate(results['meshes']):
    part_path = os.path.join(output_dir, {repr(output_name)} + f'_part_{{i:02d}}.obj')
    part_mesh.export(part_path)
    combined_parts.append(part_mesh)

# Also export combined
import trimesh
if combined_parts:
    combined = trimesh.util.concatenate(combined_parts)
    combined_path = os.path.join(output_dir, {repr(output_name + ".obj")})
    combined.export(combined_path)
    print(f'DONE:{{combined_path}}')
    print(f'PARTS:{{len(combined_parts)}}')

del model, results
torch.cuda.empty_cache()
"""
        script_path = os.path.join(output_dir, '_partcrafter_run.py')
        with open(script_path, 'w') as f:
            f.write(script)

        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True, text=True, timeout=300,
            env={**os.environ, 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'})

        if result.returncode != 0:
            return None, f"PartCrafter error: {result.stderr[:500]}"

        for line in result.stdout.split('\n'):
            if line.startswith('DONE:'):
                obj_path = line[5:].strip()
                parts_count = 1
                for l2 in result.stdout.split('\n'):
                    if l2.startswith('PARTS:'):
                        parts_count = int(l2[6:].strip())
                if os.path.exists(obj_path):
                    return obj_path, (f"PartCrafter: {parts_count} parts generated. "
                                     f"Individual part files also saved.")

        return None, "PartCrafter finished but no output found"

    except subprocess.TimeoutExpired:
        return None, "PartCrafter timed out (>300 seconds)"
    except Exception as e:
        return None, f"PartCrafter error: {e}"


def generate_from_text_openscad(description: str,
                                 output_dir: str) -> Tuple[Optional[str], str]:
    """
    Generate a 3D model from a text description using OpenSCAD.
    No GPU needed — runs on CPU.

    description: text like "a box 100mm x 80mm x 60mm with rounded corners"
    output_dir: where to save the output STL

    Returns (stl_path, status_message)
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        output_stl = os.path.join(output_dir, 'text_to_3d.stl')
        scad_file = os.path.join(output_dir, 'generated.scad')

        # Generate OpenSCAD code from the description
        # This is a simple parametric generator — for complex objects
        # it would need an LLM or the OpenSCAD MCP server
        scad_code = _description_to_openscad(description)

        with open(scad_file, 'w') as f:
            f.write(scad_code)

        # Find OpenSCAD executable
        openscad_exe = 'openscad'
        for p in [r'C:\Program Files\OpenSCAD\openscad.exe',
                  r'C:\Program Files (x86)\OpenSCAD\openscad.exe']:
            if os.path.exists(p):
                openscad_exe = p
                break

        # Render to STL
        result = subprocess.run(
            [openscad_exe, '-o', output_stl, scad_file],
            capture_output=True, text=True, timeout=60)

        if result.returncode != 0:
            return None, f"OpenSCAD error: {result.stderr[:300]}"

        if os.path.exists(output_stl):
            return output_stl, f"OpenSCAD generated: {output_stl}"

        return None, "OpenSCAD finished but no STL produced"

    except FileNotFoundError:
        return None, "OpenSCAD not installed. Download from openscad.org"
    except Exception as e:
        return None, f"OpenSCAD error: {e}"


def _description_to_openscad(description: str) -> str:
    """
    Convert a simple text description to OpenSCAD code.
    Handles basic shapes and dimensions. For complex objects,
    this would need an LLM backend.
    """
    desc = description.lower().strip()

    # Parse dimensions if present (e.g., "100x80x60" or "100mm x 80mm")
    import re
    dims = re.findall(r'(\d+(?:\.\d+)?)\s*(?:mm|cm|x)', desc)

    if 'cylinder' in desc or 'tube' in desc or 'pipe' in desc:
        r = float(dims[0]) / 2 if dims else 25
        h = float(dims[1]) if len(dims) > 1 else 50
        return f'$fn=64;\ncylinder(r={r}, h={h}, center=true);'

    elif 'sphere' in desc or 'ball' in desc:
        r = float(dims[0]) / 2 if dims else 25
        return f'$fn=64;\nsphere(r={r});'

    elif 'box' in desc or 'cube' in desc or 'block' in desc:
        x = float(dims[0]) if dims else 100
        y = float(dims[1]) if len(dims) > 1 else x
        z = float(dims[2]) if len(dims) > 2 else y
        if 'rounded' in desc or 'fillet' in desc:
            r = min(x, y, z) * 0.1
            return (f'$fn=32;\n'
                    f'minkowski() {{\n'
                    f'  cube([{x-2*r}, {y-2*r}, {z-2*r}], center=true);\n'
                    f'  sphere(r={r});\n'
                    f'}}')
        return f'cube([{x}, {y}, {z}], center=true);'

    else:
        # Default: generate a basic shape with given dimensions
        x = float(dims[0]) if dims else 100
        y = float(dims[1]) if len(dims) > 1 else 80
        z = float(dims[2]) if len(dims) > 2 else 60
        return (f'// Generated from: {description}\n'
                f'cube([{x}, {y}, {z}], center=true);')


def generate_mesh(image_paths: List[str],
                   output_dir: str,
                   backend: str = 'auto',
                   resolution: int = 256) -> Tuple[Optional[str], str]:
    """
    Generate a 3D mesh from photo(s) using the best available backend.

    backend: 'auto', 'trellis2', 'sam3d', 'partcrafter', 'triposr'
    Returns (obj_path, status_message)
    """
    if backend == 'auto':
        status = check_backends()
        # Priority: TRELLIS 2 > SAM 3D > PartCrafter > TripoSR
        if status['vram_gb'] >= 14 and status['trellis2']:
            backend = 'trellis2'
        elif status['vram_gb'] >= 16 and status['sam3d']:
            backend = 'sam3d'
        elif status['vram_gb'] >= 12 and status['partcrafter']:
            backend = 'partcrafter'
        elif status['triposr']:
            backend = 'triposr'
        else:
            return None, status['recommendation']

    if backend == 'trellis2':
        return generate_mesh_trellis2(image_paths[0], output_dir, resolution)
    elif backend == 'sam3d':
        return generate_mesh_sam3d(image_paths[0], None, output_dir)
    elif backend == 'partcrafter':
        return generate_mesh_partcrafter(image_paths[0], output_dir)
    elif backend == 'triposr':
        return generate_mesh_triposr(image_paths, output_dir, resolution)
    else:
        return None, f"Unknown backend: {backend}"
