#!/usr/bin/env python3
"""
Test script to verify KernelMem works on AMD GPU (ROCm)
"""

import sys
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_rocm_environment():
    """Test ROCm environment is properly configured."""
    print("=" * 50)
    print("Testing ROCm Environment")
    print("=" * 50)
    
    # Check PyTorch ROCm support
    print(f"\n1. PyTorch version: {torch.__version__}")
    print(f"   ROCm version: {torch.version.hip}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    
    # Check device
    if torch.cuda.is_available():
        print(f"\n2. GPU Device:")
        print(f"   Name: {torch.cuda.get_device_name(0)}")
        print(f"   Device count: {torch.cuda.device_count()}")
        
        # Test basic tensor operation
        x = torch.randn(1000, 1000).cuda()
        y = x @ x.T
        print(f"   ✓ Matrix multiplication works")
        
        # Test synchronization
        torch.cuda.synchronize()
        print(f"   ✓ CUDA synchronization works")
    
    return True


def test_gpu_platform_detection():
    """Test GPU platform detection."""
    print("\n" + "=" * 50)
    print("Testing GPU Platform Detection")
    print("=" * 50)
    
    try:
        from gpu_platform import is_amd_gpu, is_nvidia_gpu, get_gpu_info
        
        print(f"\nis_amd_gpu(): {is_amd_gpu()}")
        print(f"is_nvidia_gpu(): {is_nvidia_gpu()}")
        
        info = get_gpu_info()
        print(f"\nGPU Info:")
        for k, v in info.items():
            print(f"   {k}: {v}")
        
        return True
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False


def test_rocm_profiler():
    """Test ROCm profiler module."""
    print("\n" + "=" * 50)
    print("Testing ROCm Profiler Module")
    print("=" * 50)
    
    try:
        from run_rocm_profiler import is_rocm_available, find_rocm_profiler, METRICS
        
        print(f"\nis_rocm_available(): {is_rocm_available()}")
        print(f"ROCm profiler: {find_rocm_profiler()}")
        print(f"\nAvailable metrics (sample):")
        for m in METRICS.split(',')[:10]:
            print(f"   - {m}")
        
        return True
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_task_loading():
    """Test loading a simple KernelBench task."""
    print("\n" + "=" * 50)
    print("Testing KernelBench Task Loading")
    print("=" * 50)
    
    try:
        task_dir = project_root / "KernelBench" / "level1"
        tasks = list(task_dir.glob("*.py"))
        print(f"\nFound {len(tasks)} tasks in level1")
        
        if tasks:
            # Try loading first task
            task_path = tasks[0]
            print(f"\nTrying to load: {task_path.name}")
            
            spec = __import__(f"KernelBench.level1.{task_path.stem}", fromlist=["Model"])
            model = spec.Model()
            print(f"   ✓ Loaded: {type(model).__name__}")
            
            # Test forward pass
            import torch
            x = torch.randn(1, 64)
            y = model(x)
            print(f"   ✓ Forward pass works, output shape: {y.shape}")
        
        return True
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print(" KernelMem AMD GPU (ROCm) Compatibility Test")
    print("=" * 60)
    
    results = []
    
    results.append(("ROCm Environment", test_rocm_environment()))
    results.append(("GPU Platform Detection", test_gpu_platform_detection()))
    results.append(("ROCm Profiler", test_rocm_profiler()))
    results.append(("Task Loading", test_task_loading()))
    
    print("\n" + "=" * 60)
    print(" Test Summary")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n✓ All tests passed! KernelMem should work on AMD GPU.")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())