#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ROCm GPU Profiling Module for AMD GPUs
======================================

This module provides NVIDIA Nsight Compute/Systems equivalent functionality
for AMD GPUs using ROCm profiling tools (rocprofv3).

Adapts KernelMem project to work with AMD GPUs by:
1. Using rocprofv3 instead of ncu for kernel metrics
2. Using rocprofv3 instead of nsys for kernel launch analysis
3. Mapping NVIDIA metrics to ROCm equivalents

Usage:
    from run_rocm_profiler import profile_bench, load_rocm_metrics, metrics_to_prompt
    
    kernel_names = extract_cuda_kernel_names(test_kernel)
    csv_path = profile_bench(kernel_names=kernel_names)
    df = load_rocm_metrics(csv_path)
    prompt_block = metrics_to_prompt(df)
"""

import os
import re
import sys
import shutil
import subprocess
import signal
import time
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Sequence, Union, Any, Dict, Tuple
import json
import math

# Try to import pandas/numpy, but make them optional
try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None
    np = None


__all__ = [
    "METRICS",
    "METRICS_MAP",
    "SECTIONS",
    "profile_bench",
    "load_rocm_metrics",
    "metrics_to_prompt",
    "is_rocm_available",
]

# ROCm metrics (amd gfx11xx series equivalents to NVIDIA metrics)
# Mapping from NVIDIA metrics to ROCm equivalents
METRICS_MAP = {
    # Compute throughput
    "sm__cycles_active.avg": "GRBM_COUNT",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "MeanOccupancyPerCU",
    "sm__inst_executed.sum": "VALUInsts",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "GPUBusy",
    
    # Memory
    "dram__bytes_read.sum": "READ_SIZE",
    "dram__bytes_write.sum": "WRITE_SIZE",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": "MemUnitBusy",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed": "MemUnitBusy",
    
    # L2 Cache
    "l2cache__bytes_read.sum": "GL2C_MC_RDREQ",
    "l2cache__bytes_write.sum": "GL2C_MC_WRREQ",
    "lts__t_sector_hit_rate.pct": "L2CacheHit",
    
    # Occupancy (approximations)
    "launch__registers_per_thread": "SQ_INSTS_VALU",
    "launch__occupancy_limit_registers": "OccupancyPercent",
    "launch__occupancy_limit_shared_mem": "OccupancyPercent",
    
    # Wavefronts
    "gpu__time_duration.avg": "SQ_WAVE_CYCLES",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active": "WAVE_DEP_WAIT",
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active": "WAVE_ISSUE_WAIT",
    
    # LDS
    "lds__bytes_read.sum": "SQ_INSTS_LDS",
    "lds__bytes_write.sum": "SQ_INSTS_LDS",
    "smsp__warp_issue_stalled_memory_dependency_per_warp_active": "LdsLatency",
}


# Core ROCm metrics to collect
METRICS = ",".join([
    "Wavefronts",
    "VALUInsts",
    "SALUInsts",
    "SFetchInsts",
    "SQ_INSTS_VALU",
    "SQ_INSTS_SALU",
    "SQ_INSTS_LDS",
    "SQ_INSTS_TEX_LOAD",
    "SQ_INSTS_TEX_STORE",
    "SQ_INSTS_GDS",
    "SQ_INSTS_FLAT",
    "L2CacheHit",
    "LDSBankConflict",
    "GPUBusy",
    "GPU_UTIL",
    "GRBM_COUNT",
    "MemUnitBusy",
    "MeanOccupancyPerCU",
    "OccupancyPercent",
    "READ_SIZE",
    "WRITE_SIZE",
    "SQ_WAVES",
    "TA_BUFFER_LOAD_WAVEFRONTS",
])


SECTIONS = [
    "pmc",  # Performance monitoring counters
]


def is_rocm_available() -> bool:
    """Check if ROCm is available on this system."""
    try:
        result = subprocess.run(
            ["rocminfo"],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0 and "AMD" in result.stdout
    except Exception:
        return False


def find_rocm_profiler() -> str:
    """Find ROCm profiler binary."""
    # Try rocprofv3 first (recommended)
    for bin_name in ["rocprofv3", "rocprof"]:
        for path in [f"/opt/rocm/bin/{bin_name}", f"/usr/bin/{bin_name}"]:
            if os.path.exists(path):
                return path
        # Check in PATH
        result = subprocess.run(["which", bin_name], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
    
    # Fallback to PATH
    return "rocprofv3"


def extract_kernel_names_from_file(kernel_file: Union[str, Path]) -> List[str]:
    """Extract CUDA/HIP kernel names from a Python file."""
    try:
        src = Path(kernel_file).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []

    # Match both __global__ void and template kernel patterns
    p1 = re.compile(r"__global__\s+void\s+([A-Za-z_]\w*)\s*\(", re.MULTILINE)
    p2 = re.compile(r"__global__\s+__launch_bounds__\s*\([^)]*\)\s*void\s+([A-Za-z_]\w*)\s*\(", re.MULTILINE)
    p3 = re.compile(r"__device__\s+void\s+([A-Za-z_]\w*)\s*\(", re.MULTILINE)
    
    names = p1.findall(src) + p2.findall(src) + p3.findall(src)
    seen, ordered = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            ordered.append(n)
    return ordered


def profile_bench(
    bench_py: str = "bench_ref_inputs.py",
    kernel_names: Optional[List[str]] = None,
    kernel_file: Optional[Union[str, Path]] = None,
    conda_bin: str = "/root/miniconda3/envs/robust_kbench/bin",
    out_csv: Union[str, Path] = "rocm_temp.csv",
    repeat: int = 10,
    device_idx: Optional[int] = None,
    timeout_override: Optional[int] = None,
) -> Path:
    """
    Profile a benchmark using ROCm profiler.
    
    Equivalent to NVIDIA's ncu profiling functionality.
    """
    rocm_bin = find_rocm_profiler()
    csv_path = Path(out_csv).resolve()
    
    env = os.environ.copy()
    env["PATH"] = f"{conda_bin}:{env.get('PATH', '')}"
    
    # Use HIP_VISIBLE_DEVICES for device selection
    if device_idx is not None:
        env["HIP_VISIBLE_DEVICES"] = str(device_idx)
    
    # Build the profiling command
    cmd = [
        rocm_bin,
        "--tool-version", "2",  # Use rocprofv2 for more metrics
        "-o", str(csv_path),
        "--",
        sys.executable,
        bench_py,
    ]
    
    if kernel_file:
        cmd.extend(["--test", str(kernel_file)])
    
    cmd.extend(["--repeat", str(repeat)])
    
    if device_idx is not None:
        cmd.extend(["--device-idx", str(device_idx)])
    
    print(f"[rocm] Running:", " ".join(cmd), flush=True)
    
    timeout = timeout_override if timeout_override else 600
    
    try:
        proc = subprocess.Popen(
            cmd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid
        )
        
        try:
            stdout, _ = proc.communicate(timeout=timeout)
            print(f"[rocm] Output:\n{stdout}", flush=True)
            
            if proc.returncode != 0:
                print(f"[rocm] Warning: profiler returned {proc.returncode}", flush=True)
                
        except subprocess.TimeoutExpired:
            print(f"[rocm] Timeout after {timeout}s, killing...", flush=True)
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait()
            raise TimeoutError(f"Profiling timed out after {timeout}s")
            
    except Exception as e:
        print(f"[rocm] Error: {e}", flush=True)
        raise
    
    return csv_path


def load_rocm_metrics(
    csv_path: Union[str, Path],
    extra_keep: Optional[Tuple[str, ...]] = None,
) -> Tuple[Any, Dict]:
    """
    Load and parse ROCm profiling metrics from CSV.
    
    Returns:
        Tuple of (DataFrame, sections_dict)
    """
    if not HAS_PANDAS:
        print("[rocm] Warning: pandas not available, returning empty data")
        # Return empty dict that can be used like an empty DataFrame
        return {}, {}
    
    csv_path = Path(csv_path)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    
    # Read the CSV file
    # ROCm output format may vary, try different parsing approaches
    try:
        # Try reading with pandas
        df = pd.read_csv(csv_path, skipinitialspace=True)
    except Exception as e:
        print(f"[rocm] Warning: Failed to parse CSV with pandas: {e}")
        # Return empty DataFrame
        return pd.DataFrame(), {}
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    sections_dict = {}
    
    return df, sections_dict


def metrics_to_prompt(
    df: Any,
    sections_dict: Optional[Dict] = None,
    kernel_name: Optional[str] = None,
) -> str:
    """
    Convert metrics DataFrame to a string for LLM prompt.
    
    Maps NVIDIA-style metric names to readable descriptions.
    """
    # Handle both DataFrame and dict-like objects
    is_empty = False
    if hasattr(df, 'empty'):
        is_empty = df.empty
    elif isinstance(df, dict):
        is_empty = len(df) == 0
    else:
        is_empty = not df
    
    if is_empty:
        return "## Profiling Results\n\nNo profiling data available (profiling may have failed or timed out)."
    
    # Metric name mapping for display
    DISPLAY_NAMES = {
        "Wavefronts": "Wavefronts Executed",
        "VALUInsts": "VALU Instructions",
        "SALUInsts": "SALU Instructions", 
        "SFetchInsts": "Fetch Instructions",
        "SQ_INSTS_VALU": "Vector ALU Instructions",
        "SQ_INSTS_SALU": "Scalar ALU Instructions",
        "SQ_INSTS_LDS": "LDS Instructions",
        "L2CacheHit": "L2 Cache Hit Rate (%)",
        "LDSBankConflict": "LDS Bank Conflicts",
        "GPUBusy": "GPU Utilization (%)",
        "GPU_UTIL": "GPU Usage (%)",
        "GRBM_COUNT": "GRBM Cycles",
        "MemUnitBusy": "Memory Unit Busy (%)",
        "MeanOccupancyPerCU": "Mean Occupancy per CU",
        "OccupancyPercent": "Occupancy (%)",
        "READ_SIZE": "Read Size (bytes)",
        "WRITE_SIZE": "Write Size (bytes)",
        "SQ_WAVES": "Total Waves",
    }
    
    lines = ["## Profiling Results\n"]
    
    # Add kernel name if provided
    if kernel_name:
        lines.append(f"**Kernel**: `{kernel_name}`\n")
    
    # Add key metrics
    lines.append("### Key Metrics\n")
    
    for col in df.columns:
        if col in ["Kernel Name", "kernel_name"]:
            continue
        display_name = DISPLAY_NAMES.get(col, col)
        
        # Get value(s)
        values = df[col].dropna()
        if len(values) > 0:
            # Try to get numeric value
            try:
                val = float(values.iloc[0])
                if val > 1000:
                    lines.append(f"- **{display_name}**: {val:.0f}")
                elif val > 1:
                    lines.append(f"- **{display_name}**: {val:.2f}")
                else:
                    lines.append(f"- **{display_name}**: {val*100:.1f}%")
            except (ValueError, TypeError):
                lines.append(f"- **{display_name}**: {values.iloc[0]}")
    
    # Add analysis
    lines.append("\n### Analysis\n")
    
    # Basic analysis based on metrics
    if "GPUBusy" in df.columns:
        try:
            gpu_busy = float(df["GPUBusy"].dropna().iloc[0])
            if gpu_busy < 50:
                lines.append("- ⚠️ **Low GPU utilization** - Consider increasing parallelism or batch size")
            elif gpu_busy > 90:
                lines.append("- ✅ **Good GPU utilization**")
        except:
            pass
    
    if "OccupancyPercent" in df.columns:
        try:
            occ = float(df["OccupancyPercent"].dropna().iloc[0])
            if occ < 30:
                lines.append("- ⚠️ **Low occupancy** - Consider reducing register usage or shared memory")
            elif occ > 80:
                lines.append("- ✅ **Good occupancy**")
        except:
            pass
    
    if "L2CacheHit" in df.columns:
        try:
            l2_hit = float(df["L2CacheHit"].dropna().iloc[0])
            if l2_hit < 50:
                lines.append("- ⚠️ **Low L2 cache hit rate** - Consider improving data locality")
        except:
            pass
    
    return "\n".join(lines)


def profile_nsys_like(
    bench_py: str = "bench_ref_inputs.py",
    kernel_names: Optional[List[str]] = None,
    kernel_file: Optional[Union[str, Path]] = None,
    out_rep: Union[str, Path] = "rocm_temp.nsys-rep",
    device_idx: Optional[int] = None,
    timeout: int = 300,
) -> Path:
    """
    Profile to get kernel launch counts (like nsys).
    
    Uses ROCm tools to get kernel execution information.
    """
    rocm_bin = find_rocm_profiler()
    rep_path = Path(out_rep).resolve()
    
    env = os.environ.copy()
    
    if device_idx is not None:
        env["HIP_VISIBLE_DEVICES"] = str(device_idx)
    
    # Use rocprof to get basic stats
    # Note: ROCm doesn't have direct nsys equivalent, but we can get similar info
    cmd = [
        rocm_bin,
        "--tool-version", "2",
        "-o", str(rep_path),
        "--",
        sys.executable,
        bench_py,
    ]
    
    if kernel_file:
        cmd.extend(["--test", str(kernel_file)])
    
    print(f"[rocm] Running nsyst-like profile:", " ".join(cmd), flush=True)
    
    try:
        proc = subprocess.Popen(
            cmd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid
        )
        
        try:
            stdout, _ = proc.communicate(timeout=timeout)
            print(f"[rocm] Output:\n{stdout}", flush=True)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait()
            raise TimeoutError(f"Profiling timed out after {timeout}s")
            
    except Exception as e:
        print(f"[rocm] Error: {e}", flush=True)
    
    return rep_path


def load_nsys_stats(
    rep_path: Union[str, Path],
    kernel_names: Optional[List[str]] = None,
) -> Dict:
    """
    Load kernel statistics from profiling result.
    
    Returns dict with kernel names and launch counts.
    """
    rep_path = Path(rep_path)
    
    # For ROCm, we parse the CSV output differently
    # This is a placeholder - actual implementation depends on ROCm output format
    stats = {}
    
    csv_files = list(rep_path.parent.glob("*.csv"))
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            # Try to find kernel name and count columns
            for col in df.columns:
                if "kernel" in col.lower() or "name" in col.lower():
                    for _, row in df.iterrows():
                        name = str(row.get(col, "unknown"))
                        if name not in stats:
                            stats[name] = {"count": 1}
        except Exception as e:
            print(f"[rocm] Warning: Failed to parse {csv_file}: {e}")
    
    return stats


# Backward compatibility aliases
def load_rocm_profiler_stats(rep_path, kernel_names=None):
    """Alias for load_nsys_stats."""
    return load_nsys_stats(rep_path, kernel_names)