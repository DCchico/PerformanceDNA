import torch
import pandas as pd
from datetime import datetime
import os
import warnings
import sys
from ast import literal_eval

warnings.filterwarnings("ignore")

# Try to locate and import the quantized models
quantized_repo_path = os.path.abspath('../quantized.pytorch')
if quantized_repo_path:
    sys.path.insert(0, quantized_repo_path)

try:
    import models  # From the quantized.pytorch repository
    QUANTIZED_MODELS_AVAILABLE = True
    QUANTIZED_REPO_PATH = quantized_repo_path
except ImportError as e:
    QUANTIZED_MODELS_AVAILABLE = False
    QUANTIZED_REPO_PATH = None
    print(f"Import error: {e}")

class QuantizedModelBenchmark:
    def __init__(self):
        self.results = []
        os.makedirs("quantized_results", exist_ok=True)
    
    def create_model_name(self, model_name, model_config):
        """Create a unified model name that includes configuration info"""
        if not model_config or model_config == {}:
            return model_name
        
        # Extract key config parameters and create a descriptive name
        config_parts = []
        
        if 'depth' in model_config:
            config_parts.append(f"depth{model_config['depth']}")
        
        if 'dataset' in model_config and model_config['dataset'] != 'imagenet':
            config_parts.append(f"{model_config['dataset']}")
        
        # Add other important config parameters
        for key, value in model_config.items():
            if key not in ['depth', 'dataset'] and value is not None:
                if isinstance(value, bool):
                    if value:
                        config_parts.append(key)
                elif isinstance(value, (int, float)):
                    config_parts.append(f"{key}{value}")
                elif isinstance(value, str) and len(value) < 10:
                    config_parts.append(f"{key}_{value}")
        
        if config_parts:
            return f"{model_name}_{'_'.join(config_parts)}"
        else:
            return model_name
    
    def measure_latency(self, model, device, batch_size=1, warmup_iterations=100, timed_iterations=100):
        """Measures average forward latency (ms) using CUDA event timing."""
        dummy_input = torch.randn(batch_size, 3, 224, 224, device=device)
        
        # Warm-up
        print(f"    Warming up with {warmup_iterations} iterations...")
        for _ in range(warmup_iterations):
            with torch.no_grad():
                _ = model(dummy_input)
        
        # Use CUDA events for GPU timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        total_time_ms = 0.0
        
        print(f"    Timing {timed_iterations} iterations...")
        for _ in range(timed_iterations):
            start_event.record()
            with torch.no_grad():
                _ = model(dummy_input)
            end_event.record()
            torch.cuda.synchronize()
            total_time_ms += start_event.elapsed_time(end_event)
        
        avg_latency_ms = total_time_ms / timed_iterations
        return avg_latency_ms
    
    def load_quantized_model(self, model_name, model_config):
        """Load a quantized model using the same pattern as main.py"""
        try:
            print(f"  Loading {model_name} with config: {model_config}")
            
            # Get the model constructor from models module
            if model_name not in models.__dict__:
                available_models = [name for name in models.__dict__ 
                                  if not name.startswith('__') and callable(models.__dict__[name])]
                print(f"  Available models: {available_models}")
                return None
            
            model_constructor = models.__dict__[model_name]
            
            # Create the model with configuration
            model = model_constructor(**model_config)
            
            print(f"  ✅ Successfully loaded {model_name}")
            return model
        except Exception as e:
            print(f"  ❌ Failed to load {model_name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def benchmark_model(self, model_name, model_config, batch_size=1):
        """Benchmark a single quantized model: Eager vs Compiled"""
        unified_model_name = self.create_model_name(model_name, model_config)
        print(f"\n--- Benchmarking {unified_model_name} (batch_size={batch_size}) ---")
        
        if not torch.cuda.is_available():
            print("  ❌ CUDA not available")
            return None
            
        device = torch.device("cuda")
        
        # Load quantized model
        base_model = self.load_quantized_model(model_name, model_config)
        if base_model is None:
            return None
        
        try:
            base_model = base_model.to(device)
            base_model.eval()
            print(f"  ✅ Model moved to {device}")
        except Exception as e:
            print(f"  ❌ Failed to move model to device: {e}")
            return None
        
        scenarios = []
        
        # 1) Eager Mode
        print("  Testing Eager mode...")
        try:
            # Test a forward pass first
            test_input = torch.randn(1, 3, 224, 224, device=device)
            with torch.no_grad():
                _ = base_model(test_input)
            scenarios.append(("Eager", base_model))
            print("  ✅ Eager mode working")
        except Exception as e:
            print(f"  ❌ Eager mode failed: {e}")
            return None
        
        # 2) torch.compile
        print("  Testing Compiled mode...")
        try:
            compiled_model = torch.compile(base_model, mode='default')
            # Trigger compilation with first run
            dummy_input = torch.randn(batch_size, 3, 224, 224, device=device)
            print("    Compiling model (first run)...")
            with torch.no_grad():
                _ = compiled_model(dummy_input)
            scenarios.append(("Compiled", compiled_model))
            print("  ✅ Compilation successful")
        except Exception as e:
            print(f"  ❌ Failed to compile: {e}")
            import traceback
            traceback.print_exc()
        
        # Benchmark each scenario
        results = {}
        for name, model_instance in scenarios:
            print(f"  Benchmarking {name}...")
            try:
                latency_ms = self.measure_latency(model_instance, device, batch_size)
                results[name] = latency_ms
                print(f"    Average latency: {latency_ms:.2f} ms")
            except Exception as e:
                print(f"    ❌ Failed to benchmark {name}: {e}")
        
        # Calculate speedup
        if "Compiled" in results and "Eager" in results:
            speedup = results["Eager"] / results["Compiled"]
            status = "🚀" if speedup > 1.0 else "🐌"
        else:
            speedup = None
            status = "❌"
        
        # Store results with unified model name (no separate config column)
        result = {
            'model': unified_model_name,
            'batch_size': batch_size,
            'device': 'cuda',
            'eager_latency_ms': results.get("Eager"),
            'compiled_latency_ms': results.get("Compiled"),
            'speedup': speedup,
            'timestamp': datetime.now().isoformat()
        }
        
        self.results.append(result)
        return result
    
    def get_available_quantized_models(self):
        """Get list of available quantized models from the models module"""
        if not QUANTIZED_MODELS_AVAILABLE:
            return []
        
        # Look for models that might be quantized
        model_names = [name for name in models.__dict__ 
                      if name.islower() and not name.startswith("__") 
                      and callable(models.__dict__[name])]
        
        print(f"Available models in repository: {model_names}")
        
        # Filter for likely quantized models
        quantized_models = [name for name in model_names 
                          if 'quantized' in name.lower() or 'quant' in name.lower()]
        
        if quantized_models:
            print(f"Found quantized models: {quantized_models}")
            return quantized_models
        else:
            print("No explicitly quantized models found, trying first few models...")
            return model_names[:3]  # Limit to first 3 to avoid too many tests
    
    def run_quantized_benchmarks(self, batch_sizes=[1]):
        """Test quantized models"""
        print("🎯 QUANTIZED MODEL COMPILATION BENCHMARK")
        print(f"PyTorch version: {torch.__version__}")
        
        if not QUANTIZED_MODELS_AVAILABLE:
            print("❌ Quantized models not available. Exiting.")
            return
        
        if not torch.cuda.is_available():
            print("❌ CUDA not available. Exiting.")
            return
            
        print("GPU Information:")
        print(f"  Device: {torch.cuda.get_device_name()}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"  Compute capability: {torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}")   
        print("Methodology: 50 warmup + 50 timing iterations with CUDA events\n")
        
        # Get available quantized models
        quantized_models = self.get_available_quantized_models()
        
        if not quantized_models:
            print("❌ No models available to test")
            return
        
        # Define model configurations to test
        model_configs = []
        
        # Try standard quantized models first
        for model_name in quantized_models:
            if 'resnet' in model_name.lower():
                # ResNet quantized variants
                model_configs.extend([
                    (model_name, {"depth": 18, "dataset": "imagenet"}),
                    (model_name, {"depth": 34, "dataset": "imagenet"}),
                    (model_name, {"depth": 50, "dataset": "imagenet"}),
                    (model_name, {"depth": 101, "dataset": "imagenet"}),
                    (model_name, {"depth": 152, "dataset": "imagenet"}),
                ])
            elif 'mobilenet_quantized' in model_name.lower(): 
                model_configs.extend([
                    # Test width scaling
                    ("mobilenet_quantized", {"width": 0.25, "shallow": False, "num_classes": 1000}),
                    ("mobilenet_quantized", {"width": 0.5,  "shallow": False, "num_classes": 1000}),
                    ("mobilenet_quantized", {"width": 1.0,  "shallow": False, "num_classes": 1000}),
                    ("mobilenet_quantized", {"width": 1.4,  "shallow": False, "num_classes": 1000}),
                    
                    # Test depth scaling  
                    ("mobilenet_quantized", {"width": 1.0,  "shallow": True,  "num_classes": 1000}),
                    ("mobilenet_quantized", {"width": 1.0,  "shallow": False, "num_classes": 1000}),
                ])
            elif 'resnet_quantized_float_bn' in model_name.lower(): 
                model_configs.extend([
                    (model_name, {"depth": 18, "dataset": "imagenet"}),
                    (model_name, {"depth": 34, "dataset": "imagenet"}),
                    (model_name, {"depth": 50, "dataset": "imagenet"}),
                    (model_name, {"depth": 101, "dataset": "imagenet"}),
                    (model_name, {"depth": 152, "dataset": "imagenet"}),
                ])
            else:
                # Basic config for other models
                model_configs.append((model_name))
        
        print(f"Will test {len(model_configs)} model configurations")
        
        for batch_size in batch_sizes:
            print(f"\n{'='*60}")
            print(f"BATCH SIZE {batch_size}")
            print(f"{'='*60}")
            
            for model_name, config in model_configs:
                try:
                    self.benchmark_model(model_name, config, batch_size)
                except Exception as e:
                    print(f"❌ {model_name} with {config} failed: {e}")
                    import traceback
                    traceback.print_exc()
        
        self.save_results()
        self.print_summary()
    
    def save_results(self):
        """Save results to CSV"""
        if not self.results:
            print("No results to save")
            return None
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"quantized_results/quantized_benchmark_{timestamp}.csv"
        
        df = pd.DataFrame(self.results)
        df.to_csv(filename, index=False)
        print(f"\n✅ Results saved: {filename}")
        return filename
    
    def print_summary(self):
        """Simple comparison summary"""
        if not self.results:
            print("No results to summarize")
            return
        
        print(f"\n{'='*75}")
        print("QUANTIZED MODEL RESULTS SUMMARY")
        print(f"{'='*75}")
        
        # Simplified table without separate config column
        print(f"{'Model':<35} {'BS':<3} {'Eager(ms)':<10} {'Compiled(ms)':<13} {'Speedup':<8}")
        print("-" * 70)
        
        for result in self.results:
            model = result['model']
            batch_size = result['batch_size']
            eager = result['eager_latency_ms']
            compiled = result['compiled_latency_ms']
            speedup = result['speedup']
            
            # Truncate model name if too long
            model_short = model[:33] + ".." if len(model) > 35 else model
            
            if compiled and eager:
                status = "🚀" if speedup and speedup > 1.0 else "🐌"
                print(f"{model_short:<35} {batch_size:<3} {eager:<10.2f} {compiled:<13.2f} {speedup:<7.2f}{status}")
            elif eager:
                print(f"{model_short:<35} {batch_size:<3} {eager:<10.2f} {'COMP_FAIL':<13} {'N/A':<8}")
            else:
                print(f"{model_short:<35} {batch_size:<3} {'LOAD_FAIL':<10} {'LOAD_FAIL':<13} {'N/A':<8}")
        
        print(f"{'='*75}")
        
        # Calculate average speedup
        successful_speedups = [r['speedup'] for r in self.results if r['speedup'] is not None and r['speedup'] > 0]
        if successful_speedups:
            avg_speedup = sum(successful_speedups) / len(successful_speedups)
            print(f"Average speedup: {avg_speedup:.2f}x")
            print(f"Successful compilations: {len(successful_speedups)}/{len(self.results)}")
        else:
            print("No successful compilations to report")

def check_setup():
    """Check if the environment is set up correctly"""
    print("🔍 SETUP CHECK")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
    else:
        print("❌ CUDA not available - this benchmark requires GPU")
        return False
    
    if not QUANTIZED_MODELS_AVAILABLE:
        print("❌ Cannot import models from quantized.pytorch repository")
        if QUANTIZED_REPO_PATH:
            print(f"   Found repo at {QUANTIZED_REPO_PATH} but import failed")
        else:
            print("   Repository not found in common locations")
        
        return False
    
    print(f"✅ Setup looks good! Using quantized repo at: {QUANTIZED_REPO_PATH}")
    return True

if __name__ == "__main__":
    print("🎯 QUANTIZED MODEL COMPILATION BENCHMARK")
    print("=" * 60)
    
    if not check_setup():
        print("\n💡 QUICK SETUP:")
        print("git clone https://github.com/eladhoffer/quantized.pytorch.git")
        print("cd quantized.pytorch && git checkout e09c447 && pip install -e . && cd path/to/quantized.py")
        print("python quantized.py")
        sys.exit(1)
    
    # Test a simple PyTorch operation first
    print("Testing basic PyTorch functionality...")
    if torch.cuda.is_available():
        x = torch.randn(2, 3).cuda()
        print(f"✅ Basic PyTorch CUDA test passed: {x.shape}")
    
    benchmark = QuantizedModelBenchmark()
    
    benchmark.run_quantized_benchmarks(batch_sizes=[1, 4, 8, 16])