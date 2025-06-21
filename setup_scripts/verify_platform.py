#!/usr/bin/env python3
"""
Platform verification script for PyTorch benchmarks.
Checks CUDA availability and PyTorch version compatibility for torchdynamo and torchinductor.
"""

import sys
import subprocess
import platform
from typing import Dict, List, Tuple, Optional
import warnings

def get_system_info() -> Dict[str, str]:
    """Get basic system information."""
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "architecture": platform.architecture()[0],
        "processor": platform.processor()
    }

def check_pytorch_installation() -> Tuple[bool, Dict[str, str]]:
    """Check PyTorch installation and version."""
    try:
        import torch
        version_info = {
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
            "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else "N/A"
        }
        return True, version_info
    except ImportError:
        return False, {"error": "PyTorch not installed"}

def check_cuda_devices() -> Tuple[bool, Dict[str, any]]:
    """Check CUDA devices and their properties."""
    try:
        import torch
        
        if not torch.cuda.is_available():
            return False, {"error": "CUDA not available"}
        
        device_count = torch.cuda.device_count()
        devices_info = []
        
        for i in range(device_count):
            device_props = torch.cuda.get_device_properties(i)
            device_info = {
                "index": i,
                "name": device_props.name,
                "compute_capability": f"{device_props.major}.{device_props.minor}",
                "total_memory_gb": device_props.total_memory / (1024**3),
                "multi_processor_count": device_props.multi_processor_count
            }
            devices_info.append(device_info)
        
        return True, {
            "device_count": device_count,
            "devices": devices_info,
            "current_device": torch.cuda.current_device() if device_count > 0 else None
        }
    except Exception as e:
        return False, {"error": f"Error checking CUDA devices: {str(e)}"}

def check_torchdynamo_support() -> Tuple[bool, Dict[str, str]]:
    """Check if torchdynamo is available and working."""
    try:
        import torch
        
        # Check if torch.compile is available (PyTorch 2.0+)
        if not hasattr(torch, 'compile'):
            return False, {"error": "torch.compile not available - requires PyTorch 2.0+"}
        
        # Test basic torch.compile functionality
        def test_function(x):
            return x * 2 + 1
        
        x = torch.randn(10)
        
        try:
            compiled_fn = torch.compile(test_function)
            result = compiled_fn(x)
            return True, {
                "status": "Available and working",
                "backend": "default"
            }
        except Exception as e:
            return False, {"error": f"torch.compile failed: {str(e)}"}
            
    except Exception as e:
        return False, {"error": f"Error checking torchdynamo: {str(e)}"}

def check_torchinductor_support() -> Tuple[bool, Dict[str, str]]:
    """Check if torchinductor is available and working."""
    try:
        import torch
        
        # Check if torch.compile is available
        if not hasattr(torch, 'compile'):
            return False, {"error": "torch.compile not available"}
        
        # Check available backends
        try:
            import torch._dynamo
            available_backends = torch._dynamo.list_backends()
            if 'inductor' not in available_backends:
                return False, {"error": f"inductor backend not available. Available: {available_backends}"}
        except Exception:
            return False, {"error": "Could not check available backends"}
        
        # Test basic inductor functionality
        def test_function(x):
            return x * 2 + 1
        
        x = torch.randn(10)
        
        try:
            compiled_fn = torch.compile(test_function, backend="inductor")
            result = compiled_fn(x)
            return True, {
                "status": "Available and working",
                "backend": "inductor"
            }
        except Exception as e:
            # If inductor fails due to compilation issues, check if it's a system dependency issue
            if "Python.h" in str(e) or "C++ compile error" in str(e):
                return False, {
                    "error": "inductor backend available but compilation failed (missing Python dev headers)",
                    "details": "This is a system dependency issue, not a PyTorch issue"
                }
            else:
                return False, {"error": f"torchinductor compilation failed: {str(e)}"}
            
    except Exception as e:
        return False, {"error": f"Error checking torchinductor: {str(e)}"}

def check_pytorch_version_compatibility() -> Tuple[bool, Dict[str, str]]:
    """Check if PyTorch version supports required features."""
    try:
        import torch
        import packaging.version
        
        pytorch_version = packaging.version.parse(torch.__version__)
        min_version = packaging.version.parse("2.0.0")
        
        if pytorch_version < min_version:
            return False, {
                "error": f"PyTorch version {torch.__version__} is too old. Requires 2.0.0+ for torchdynamo/inductor"
            }
        
        return True, {
            "version": torch.__version__,
            "status": "Compatible"
        }
    except Exception as e:
        return False, {"error": f"Error checking version compatibility: {str(e)}"}

def run_simple_benchmark() -> Tuple[bool, Dict[str, any]]:
    """Run a simple benchmark to test basic functionality."""
    try:
        import torch
        import time
        
        if not torch.cuda.is_available():
            return False, {"error": "CUDA not available for benchmark"}
        
        # Move to GPU
        device = torch.device("cuda")
        
        # Create test tensors
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        
        # Warm up
        for _ in range(10):
            _ = torch.mm(x, y)
        
        torch.cuda.synchronize()
        
        # Benchmark
        start_time = time.time()
        for _ in range(100):
            _ = torch.mm(x, y)
        torch.cuda.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        
        return True, {
            "avg_time_ms": avg_time * 1000,
            "operations_per_second": 100 / (end_time - start_time)
        }
    except Exception as e:
        return False, {"error": f"Benchmark failed: {str(e)}"}

def print_results(results: Dict[str, any]) -> None:
    """Print verification results in a formatted way."""
    print("=" * 60)
    print("PYTORCH PLATFORM VERIFICATION RESULTS")
    print("=" * 60)
    
    # System info
    print("\n📋 SYSTEM INFORMATION:")
    for key, value in results["system_info"].items():
        print(f"  {key}: {value}")
    
    # PyTorch installation
    print("\n🔥 PYTORCH INSTALLATION:")
    if results["pytorch_ok"]:
        for key, value in results["pytorch_info"].items():
            print(f"  {key}: {value}")
    else:
        print(f"  ❌ {results['pytorch_info']['error']}")
    
    # Version compatibility
    print("\n📦 VERSION COMPATIBILITY:")
    if results["version_ok"]:
        print(f"  ✅ {results['version_info']['status']} (PyTorch {results['version_info']['version']})")
    else:
        print(f"  ❌ {results['version_info']['error']}")
    
    # CUDA devices
    print("\n🚀 CUDA DEVICES:")
    if results["cuda_ok"]:
        print(f"  ✅ Found {results['cuda_info']['device_count']} CUDA device(s)")
        for device in results["cuda_info"]["devices"]:
            print(f"    Device {device['index']}: {device['name']}")
            print(f"      Compute Capability: {device['compute_capability']}")
            print(f"      Memory: {device['total_memory_gb']:.1f} GB")
            print(f"      Multiprocessors: {device['multi_processor_count']}")
    else:
        print(f"  ❌ {results['cuda_info']['error']}")
    
    # TorchDynamo
    print("\n⚡ TORCHDYNAMO:")
    if results["dynamo_ok"]:
        print(f"  ✅ {results['dynamo_info']['status']}")
    else:
        print(f"  ❌ {results['dynamo_info']['error']}")
    
    # TorchInductor
    print("\n🔧 TORCHINDUCTOR:")
    if results["inductor_ok"]:
        print(f"  ✅ {results['inductor_info']['status']}")
    else:
        print(f"  ❌ {results['inductor_info']['error']}")
    
    # Benchmark
    print("\n🏃 BENCHMARK TEST:")
    if results["benchmark_ok"]:
        print(f"  ✅ Matrix multiplication: {results['benchmark_info']['avg_time_ms']:.2f} ms avg")
        print(f"     Throughput: {results['benchmark_info']['operations_per_second']:.1f} ops/sec")
    else:
        print(f"  ❌ {results['benchmark_info']['error']}")
    
    # Overall status
    print("\n" + "=" * 60)
    if results["all_checks_passed"]:
        print("✅ ALL CHECKS PASSED - Platform ready for PyTorch benchmarks!")
    else:
        print("❌ SOME CHECKS FAILED - Please address the issues above")
    print("=" * 60)

def main():
    """Main verification function."""
    print("🔍 Starting PyTorch platform verification...")
    
    results = {}
    
    # System information
    results["system_info"] = get_system_info()
    
    # PyTorch installation
    results["pytorch_ok"], results["pytorch_info"] = check_pytorch_installation()
    
    # Version compatibility
    results["version_ok"], results["version_info"] = check_pytorch_version_compatibility()
    
    # CUDA devices
    results["cuda_ok"], results["cuda_info"] = check_cuda_devices()
    
    # TorchDynamo
    results["dynamo_ok"], results["dynamo_info"] = check_torchdynamo_support()
    
    # TorchInductor
    results["inductor_ok"], results["inductor_info"] = check_torchinductor_support()
    
    # Simple benchmark
    results["benchmark_ok"], results["benchmark_info"] = run_simple_benchmark()
    
    # Overall status
    results["all_checks_passed"] = all([
        results["pytorch_ok"],
        results["version_ok"],
        results["cuda_ok"],
        results["dynamo_ok"],
        results["inductor_ok"],
        results["benchmark_ok"]
    ])
    
    # Print results
    print_results(results)
    
    # Return appropriate exit code
    return 0 if results["all_checks_passed"] else 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠️  Verification interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error during verification: {str(e)}")
        sys.exit(1)
