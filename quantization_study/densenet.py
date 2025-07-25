import torch
import torchvision.models as models
import pandas as pd
from datetime import datetime
import os
import warnings
warnings.filterwarnings("ignore")

class SimpleDenseNetBenchmark:
    def __init__(self):
        self.results = []
        os.makedirs("densenet_results", exist_ok=True)
    
    def measure_latency(self, model, device, batch_size=1, warmup_iterations=100, timed_iterations=100):
        """Measures average forward latency (ms) using CUDA event timing."""
        dummy_input = torch.randn(batch_size, 3, 224, 224, device=device)
        
        # Warm-up
        for _ in range(warmup_iterations):
            model(dummy_input)
        
        # Use CUDA events for GPU timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        total_time_ms = 0.0
        
        for _ in range(timed_iterations):
            start_event.record()
            _ = model(dummy_input)
            end_event.record()
            torch.cuda.synchronize()
            total_time_ms += start_event.elapsed_time(end_event)
        
        avg_latency_ms = total_time_ms / timed_iterations
        return avg_latency_ms
    
    def benchmark_model(self, model_name, batch_size=1):
        """Benchmark a single DenseNet variant: Eager vs Compiled"""
        print(f"\n--- Benchmarking {model_name} (batch_size={batch_size}) ---")
        
        device = torch.device("cuda")  # GPU only
        
        # Load model
        model_func = getattr(models, model_name)
        weights = 'DEFAULT'
        base_model = model_func(weights=weights).to(device)
        base_model.eval()
        
        scenarios = []
        
        # 1) Eager Mode
        scenarios.append(("Eager", base_model))
        
        # 2) torch.compile
        try:
            compiled_model = torch.compile(base_model)
            # Trigger compilation with first run
            dummy_input = torch.randn(batch_size, 3, 224, 224, device=device)
            print("  Compiling model (first run)...")
            _ = compiled_model(dummy_input)
            scenarios.append(("Compiled", compiled_model))
        except Exception as e:
            print(f"  Failed to compile: {e}")
        
        # Benchmark each scenario
        results = {}
        for name, model_instance in scenarios:
            print(f"  Testing {name}...")
            latency_ms = self.measure_latency(model_instance, device, batch_size)
            results[name] = latency_ms
            print(f"    Average latency: {latency_ms:.2f} ms")
        
        # Calculate speedup
        if "Compiled" in results:
            speedup = results["Eager"] / results["Compiled"]
            status = "🚀" if speedup > 1.0 else "🐌"
        else:
            speedup = None
        
        # Store results
        result = {
            'model': model_name,
            'batch_size': batch_size,
            'device': 'cuda',
            'eager_latency_ms': results.get("Eager"),
            'compiled_latency_ms': results.get("Compiled"),
            'speedup': speedup,
            'timestamp': datetime.now().isoformat()
        }
        
        self.results.append(result)
        return result
    
    def run_all_densenets(self, batch_sizes=[1]):
        """Test all DenseNet variants"""
        print("🎯 DENSENET COMPILATION BENCHMARK")
        print(f"PyTorch version: {torch.__version__}")
        print("GPU Information:")
        print(f"  Device: {torch.cuda.get_device_name()}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"  Compute capability: {torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}")   
        print("Methodology: 100 warmup + 100 timing iterations with CUDA events\n")
        
        # All DenseNet variants
        densenet_models = [
            'densenet121',
            'densenet161', 
            'densenet169',
            'densenet201'
        ]
        
        for batch_size in batch_sizes:
            print(f"\n=== BATCH SIZE {batch_size} ===")
            for model_name in densenet_models:
                try:
                    self.benchmark_model(model_name, batch_size)
                except Exception as e:
                    print(f"❌ {model_name} failed: {e}")
        
        self.save_results()
        self.print_summary()
    
    def save_results(self):
        """Save results to CSV"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"densenet_results/densenet_benchmark_{timestamp}.csv"
        
        df = pd.DataFrame(self.results)
        df.to_csv(filename, index=False)
        print(f"\n✅ Results saved: {filename}")
        return filename
    
    def print_summary(self):
        """Simple comparison summary"""
        if not self.results:
            return
        
        print(f"\n{'='*75}")
        print("RESULTS SUMMARY")
        print(f"{'='*75}")
        
        # Simple table with batch size
        print(f"{'Model':<12} {'BS':<3} {'Eager(ms)':<10} {'Compiled(ms)':<13} {'Speedup':<8}")
        print("-" * 55)
        
        for result in self.results:
            model = result['model']
            batch_size = result['batch_size']
            eager = result['eager_latency_ms']
            compiled = result['compiled_latency_ms']
            speedup = result['speedup']
            
            if compiled:
                status = "🚀" if speedup > 1.0 else "🐌"
                print(f"{model:<12} {batch_size:<3} {eager:<10.2f} {compiled:<13.2f} {speedup:<7.2f}{status}")
            else:
                print(f"{model:<12} {batch_size:<3} {eager:<10.2f} {'FAILED':<13} {'N/A':<8}")
        
        print(f"{'='*75}")

if __name__ == "__main__":
    benchmark = SimpleDenseNetBenchmark()
    
    # Test all DenseNet variants with different batch sizes
    benchmark.run_all_densenets(batch_sizes=[1, 4, 8, 16])
