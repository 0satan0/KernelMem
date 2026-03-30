#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GPU Platform Detection and Tool Selection
==========================================

Detects whether running on NVIDIA (CUDA) or AMD (ROCm/HIP) platform
and provides appropriate tools for profiling.

Usage:
    from gpu_platform import get_profiler_module, is_amd_gpu, get_gpu_name
    
    if is_amd_gpu():
        profiler = get_profiler_module()
        # Use AMD profiling tools
    else:
        profiler = get_profiler_module()
        # Use NVIDIA profiling tools
"""

import os
import subprocess
import torch
from typing import Optional, Tuple


def is_amd_gpu() -> bool:
    """Check if running on AMD GPU (ROCm)."""
    # Check PyTorch version for HIP support
    if hasattr(torch.version, 'hip') and torch.version.hip is not None:
        return True
    
    # Check for HIP runtime
    if os.environ.get('HIP_VISIBLE_DEVICES') is not None:
        return True
    
    # Try rocminfo
    try:
        result = subprocess.run(
            ["rocminfo"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0 and "AMD" in result.stdout:
            return True
    except Exception:
        pass
    
    return False


def is_nvidia_gpu() -> bool:
    """Check if running on NVIDIA GPU (CUDA)."""
    if hasattr(torch.version, 'cuda') and torch.version.cuda is not None:
        return True
    
    # Try nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            return True
    except Exception:
        pass
    
    return False


def get_gpu_name() -> str:
    """Get the GPU device name."""
    if is_amd_gpu():
        try:
            result = subprocess.run(
                ["rocm-smi", "--showdevice"],
                capture_output=True,
                text=True,
                timeout=10
            )
            for line in result.stdout.split('\n'):
                if 'Device Name' in line:
                    return line.split(':')[-1].strip()
        except Exception:
            pass
        return "AMD GPU (ROCm)"
    
    elif is_nvidia_gpu():
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return "NVIDIA GPU (CUDA)"
    
    return "CPU"


def get_gpu_info() -> dict:
    """Get comprehensive GPU information."""
    info = {
        "platform": "cpu",
        "gpu_name": "CPU",
        "device_count": 0,
        "cuda_available": False,
        "rocn_available": False,
    }
    
    if is_nvidia_gpu():
        info["platform"] = "nvidia"
        info["cuda_available"] = True
        try:
            info["device_count"] = torch.cuda.device_count()
            if info["device_count"] > 0:
                info["gpu_name"] = torch.cuda.get_device_name(0)
        except Exception:
            pass
    
    elif is_amd_gpu():
        info["platform"] = "amd"
        info["rocn_available"] = True
        try:
            # Try to get AMD device count
            result = subprocess.run(
                ["rocm-smi", "--listdevices"],
                capture_output=True,
                text=True,
                timeout=10
            )
            # Count GPU lines
            gpu_count = result.stdout.count("GPU")
            info["device_count"] = gpu_count if gpu_count > 0 else 1
        except Exception:
            info["device_count"] = 1
        
        info["gpu_name"] = get_gpu_name()
    
    return info


def get_profiler_module():
    """
    Get the appropriate profiler module based on GPU platform.
    
    Returns:
        Module with profile_bench, load_metrics, metrics_to_prompt functions
    """
    if is_amd_gpu():
        from . import run_rocm_profiler as profiler
        return profiler
    else:
        from . import run_ncu_memory as profiler
        return profiler


def get_profile_command_prefix() -> list:
    """
    Get platform-specific profiler command prefix.
    
    Returns:
        List of command prefix parts (e.g., ['ncu', '--config-file-path=...'])
    """
    if is_amd_gpu():
        return ["rocprofv3", "--tool-version", "2"]
    else:
        return ["ncu"]


def should_use_hip() -> bool:
    """
    Check if HIP (ROCm) compilation should be used.
    
    Returns:
        True if should use HIP, False for CUDA
    """
    return is_amd_gpu()


# Convenience function for checking availability
def check_profiling_tools() -> dict:
    """Check which profiling tools are available."""
    tools = {
        "ncu": False,
        "nsys": False,
        "rocprof": False,
        "rocprofv3": False,
    }
    
    # Check NVIDIA tools
    for tool in ["ncu", "nsys"]:
        try:
            result = subprocess.run(
                ["which", tool],
                capture_output=True,
                text=True,
                timeout=5
            )
            tools[tool] = result.returncode == 0
        except Exception:
            pass
    
    # Check ROCm tools
    for tool in ["rocprof", "rocprofv3"]:
        try:
            result = subprocess.run(
                ["which", tool],
                capture_output=True,
                text=True,
                timeout=5
            )
            tools[tool] = result.returncode == 0
        except Exception:
            pass
    
    return tools


if __name__ == "__main__":
    print("GPU Platform Detection")
    print("=" * 40)
    
    info = get_gpu_info()
    for k, v in info.items():
        print(f"  {k}: {v}")
    
    print("\nProfiling Tools:")
    tools = check_profiling_tools()
    for k, v in tools.items():
        print(f"  {k}: {'✓' if v else '✗'}")