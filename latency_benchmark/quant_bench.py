import torch
import torch.nn as nn
import time
import argparse
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights
from transformers import BertModel, ViTModel
import csv
from datetime import datetime
import os
import warnings
import sys

# Suppress minor warnings
warnings.filterwarnings("ignore")

# Note: QAT models from external repositories are not suitable for inference benchmarking
# as they use fake quantization and are designed for training, not inference.
# We focus on converting standard torchvision models to true INT8 for real performance benefits.
QUANTIZED_MODELS_AVAILABLE = False
QUANTIZED_REPO_PATH = None

class QuantizedModelBenchmark:
    def __init__(self):
        self.results = []
        os.makedirs("../results", exist_ok=True)
        
    def convert_standard_model_to_int8(self, model, device="cuda"):
        """Convert standard PyTorch model to INT8 using post-training quantization"""
        try:
            print(f"    Converting standard model to INT8 using post-training quantization...")
            
            # Move model to CPU for quantization
            model = model.cpu()
            model.eval()
            
            # Set up quantization configuration
            qconfig = torch.quantization.get_default_qconfig('fbgemm')
            
            # Prepare the model (fuse modules, insert observers)
            prepared_model = torch.quantization.prepare(model, qconfig)
            
            # Calibrate with sample data
            print(f"    Calibrating with sample data...")
            with torch.no_grad():
                for i in range(100):  # calibration iterations
                    dummy_input = torch.randn(1, 3, 224, 224)
                    prepared_model(dummy_input)
                    if (i + 1) % 20 == 0:
                        print(f"      Calibration progress: {i+1}/100")
            
            # Convert to INT8
            print(f"    Converting to INT8...")
            int8_model = torch.quantization.convert(prepared_model)
            
            # Move back to original device
            int8_model = int8_model.to(device)
            
            print(f"    ✅ Successfully converted to true INT8 model")
            return int8_model
            
        except Exception as e:
            print(f"    ❌ Failed to convert to INT8: {e}")
            print(f"    Falling back to original model")
            return model.to(device)
    
    def convert_standard_model_to_int8(self, model, device="cuda"):
        """Convert standard PyTorch model to INT8 using post-training quantization"""
        try:
            print(f"    Converting standard model to INT8 using post-training quantization...")
            
            # Move model to CPU for quantization
            model = model.cpu()
            model.eval()
            
            # Set up quantization configuration
            qconfig = torch.quantization.get_default_qconfig('fbgemm')
            
            # Prepare the model (fuse modules, insert observers)
            prepared_model = torch.quantization.prepare(model, qconfig)
            
            # Calibrate with sample data
            print(f"    Calibrating with sample data...")
            with torch.no_grad():
                for i in range(100):  # calibration iterations
                    dummy_input = torch.randn(1, 3, 224, 224)
                    prepared_model(dummy_input)
                    if (i + 1) % 20 == 0:
                        print(f"      Calibration progress: {i+1}/100")
            
            # Convert to INT8
            print(f"    Converting to INT8...")
            int8_model = torch.quantization.convert(prepared_model)
            
            # Move back to original device
            int8_model = int8_model.to(device)
            
            print(f"    ✅ Successfully converted to true INT8 model")
            return int8_model
            
        except Exception as e:
            print(f"    ❌ Failed to convert to INT8: {e}")
            print(f"    Falling back to original model")
            return model.to(device)
        
    def measure_latency(self, model, dummy_inputs, timed_iterations=100, warmup_iterations=50):
        """
        Measures average inference latency using CUDA events.
        Ensures model and inputs stay on GPU throughout.
        """
        device = next(model.parameters()).device
        assert device.type == 'cuda', f"Model must be on GPU, found on {device}"
        
        # Warm-up phase
        print(f"    Warming up with {warmup_iterations} iterations...")
        for _ in range(warmup_iterations):
            with torch.no_grad():
                _ = model(*dummy_inputs)
        
        # Synchronize to ensure warm-up is complete
        torch.cuda.synchronize()
        
        # Timing using CUDA events
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(timed_iterations):
            with torch.no_grad():
                _ = model(*dummy_inputs)
        end_event.record()
        
        # Wait for all kernels to complete
        torch.cuda.synchronize()
        
        # Calculate average latency
        total_time_ms = start_event.elapsed_time(end_event)
        avg_latency_ms = total_time_ms / timed_iterations
        return avg_latency_ms
    

    
    def get_model_and_input(self, model_name, batch_size, device="cuda"):
        """Load pretrained models and create dummy inputs."""
        # Standard models
        if model_name == "resnet50":
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval()
            # Convert to INT8 if requested
            if hasattr(self, 'use_true_int8') and self.use_true_int8:
                model = self.convert_standard_model_to_int8(model, device)
            dummy_input = torch.randn(batch_size, 3, 224, 224, device=device)
            return model, (dummy_input,)
            
        elif model_name == "resnet18":
            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval()
            # Convert to INT8 if requested
            if hasattr(self, 'use_true_int8') and self.use_true_int8:
                model = self.convert_standard_model_to_int8(model, device)
            dummy_input = torch.randn(batch_size, 3, 224, 224, device=device)
            return model, (dummy_input,)
            
        elif model_name == "bert-base-uncased":
            model = BertModel.from_pretrained("bert-base-uncased").eval()
            sequence_length = 512
            dummy_input = torch.randint(0, 30000, (batch_size, sequence_length), dtype=torch.long, device=device)
            return model, (dummy_input,)
            
        elif model_name == "vit":
            model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k").eval()
            dummy_input = torch.randn(batch_size, 3, 224, 224, device=device)
            return model, (dummy_input,)
        

            
        else:
            raise ValueError(f"Model '{model_name}' not supported.")
    
    def apply_fp16_precision(self, model, device):
        """Apply FP16 (half precision) to the model."""
        print("    Applying FP16 precision...")
        
        # Convert model to FP16
        model_fp16 = model.half()
        
        # Verify it's on GPU and FP16
        assert next(model_fp16.parameters()).device.type == 'cuda'
        assert next(model_fp16.parameters()).dtype == torch.float16
        
        print("    ✅ FP16 precision applied")
        return model_fp16
    
    def get_available_quantized_models(self):
        """Get list of available quantized models"""
        # Note: We focus on converting standard torchvision models to INT8
        # rather than using external QAT models that are not suitable for inference
        return []
    
    def benchmark_model(self, model_name, batch_size, device="cuda"):
        """Benchmark a model with different quantization methods."""
        print(f"\n{'='*60}")
        print(f"BENCHMARKING {model_name.upper()} (batch_size={batch_size})")
        print(f"{'='*60}")
        
        results_data = []
        gpu_name = torch.cuda.get_device_name(0).replace(" ", "_")
        
        # Check if this is a quantized model (using true INT8 conversion)
        is_quantized = hasattr(self, 'use_true_int8') and self.use_true_int8
        
        # Test scenarios - create separate model instances for each precision
        scenarios = []
        
        if is_quantized:
            # For INT8 models, only test the INT8 version
            try:
                model, inputs = self.get_model_and_input(model_name, batch_size, device)
                model.to(device)
                model.eval()
                scenarios.append(("TRUE_INT8_Eager", model, "True INT8 quantized model", inputs))
            except Exception as e:
                print(f"    ❌ INT8 model loading failed: {e}")
        else:
            # Check if this is a standard model being converted to INT8
            if hasattr(self, 'use_true_int8') and self.use_true_int8:
                # INT8 model (converted from standard model)
                try:
                    int8_model, int8_inputs = self.get_model_and_input(model_name, batch_size, device)
                    int8_model.to(device)
                    int8_model.eval()
                    scenarios.append(("TRUE_INT8_Eager", int8_model, "True INT8 quantized model", int8_inputs))
                except Exception as e:
                    print(f"    ❌ INT8 model loading failed: {e}")
            else:
                # FP32 baseline
                try:
                    base_model, dummy_inputs = self.get_model_and_input(model_name, batch_size, device)
                    base_model.to(device)
                    scenarios.append(("FP32_Eager", base_model, "Baseline FP32 model", dummy_inputs))
                except Exception as e:
                    print(f"    ❌ FP32 model loading failed: {e}")
                
                # FP16 model
                try:
                    fp16_model, fp16_inputs = self.get_model_and_input(model_name, batch_size, device)
                    fp16_model.to(device)
                    fp16_model = self.apply_fp16_precision(fp16_model, device)
                    # Create FP16 inputs to match the model
                    fp16_inputs = tuple(t.half() for t in fp16_inputs)
                    scenarios.append(("FP16_Eager", fp16_model, "FP16 precision", fp16_inputs))
                except Exception as e:
                    print(f"    ❌ FP16 conversion failed: {e}")
        
        # Test torch.compile on each scenario
        compiled_scenarios = []
        for name, model, description, inputs in scenarios:
            try:
                compiled_model = torch.compile(model, backend="inductor")
                compiled_scenarios.append((f"{name}_Compiled", compiled_model, f"{description} + torch.compile", inputs))
            except Exception as e:
                print(f"    ❌ Compilation failed for {name}: {e}")
        
        scenarios.extend(compiled_scenarios)
        
        # Benchmark each scenario
        for scenario_name, model, description, test_inputs in scenarios:
            print(f"\n  Testing {scenario_name}: {description}")
            
            try:
                # Test if model actually runs on GPU
                test_inputs = tuple(t.clone() for t in test_inputs)
                with torch.no_grad():
                    _ = model(*test_inputs)
                
                # Measure latency
                latency = self.measure_latency(model, test_inputs)
                
                print(f"    ✅ Average latency: {latency:.3f} ms")
                
                # Determine quantization type
                if hasattr(self, 'use_true_int8') and self.use_true_int8 and ("TRUE_INT8" in scenario_name or is_quantized):
                    quant_type = "TRUE_INT8"
                elif "QAT" in scenario_name:
                    quant_type = "QAT_INT8"
                elif "FP16" in scenario_name:
                    quant_type = "FP16"
                else:
                    quant_type = "FP32"
                
                # Determine compilation type
                if "Compiled" in scenario_name:
                    compilation = "torch.compile"
                else:
                    compilation = "Eager"
                
                results_data.append([
                    model_name, batch_size, gpu_name, quant_type, compilation, latency
                ])
                
            except Exception as e:
                print(f"    ❌ Failed: {e}")
                
                # Determine quantization type for failed cases
                if hasattr(self, 'use_true_int8') and self.use_true_int8 and is_quantized:
                    quant_type = "TRUE_INT8"
                elif "QAT" in scenario_name or "TRUE_INT8" in scenario_name:
                    quant_type = "QAT_INT8"
                elif "FP16" in scenario_name:
                    quant_type = "FP16"
                else:
                    quant_type = "FP32"
                
                results_data.append([
                    model_name, batch_size, gpu_name, 
                    quant_type,
                    "Eager" if "Compiled" not in scenario_name else "torch.compile", 
                    float('inf')
                ])
        
        return results_data

    def run_benchmarks(self, models, batch_sizes, device="cuda"):
        """Run benchmarks for all models and batch sizes."""
        print("🎯 QUANTIZATION BENCHMARK")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Device memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        print("ℹ️  Focus: Converting standard torchvision models to true INT8 for real performance benefits")
        
        print()
        
        all_results = []
        
        for model_name in models:
            for batch_size in batch_sizes:
                try:
                    results = self.benchmark_model(model_name, batch_size, device)
                    all_results.extend(results)
                except Exception as e:
                    print(f"❌ Failed to benchmark {model_name} with batch_size={batch_size}: {e}")
                    continue
        
        return all_results

    def print_summary(self, results):
        """Print a summary table of results."""
        print("\n" + "="*80)
        print("QUANTIZATION BENCHMARK SUMMARY")
        print("="*80)
        print(f"{'Model':<12} {'BS':<3} {'Quant':<8} {'Compile':<12} {'Latency(ms)':<12} {'Status':<8}")
        print("-" * 80)
        
        for result in results:
            model_name, batch_size, gpu_name, quant_type, compilation, latency = result
            
            if latency == float('inf'):
                status = "❌ FAIL"
                latency_str = "inf"
            else:
                status = "✅ PASS"
                latency_str = f"{latency:.3f}"
            
            print(f"{model_name:<12} {batch_size:<3} {quant_type:<8} {compilation:<12} {latency_str:<12} {status:<8}")
        
        print("="*80)
        
        # Save results to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"../results/quantization_benchmark_{timestamp}.csv"
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Model', 'Batch_Size', 'GPU', 'Quantization', 'Compilation', 'Latency_ms'])
            writer.writerows(results)
        
        print(f"\nResults saved to: {filename}")

def main():
    parser = argparse.ArgumentParser(description="Quantization Benchmark")
    parser.add_argument("--models", nargs="+", 
                       default=["resnet18", "resnet50"],
                       help="Models to benchmark")
    parser.add_argument("--batch-sizes", nargs="+", type=int,
                       default=[1, 4, 8, 16],
                       help="Batch sizes to test")
    parser.add_argument("--device", default="cuda",
                       help="Device to run on (cuda/cpu)")
    parser.add_argument("--list-quantized", action="store_true",
                       help="List available quantized models and exit")

    parser.add_argument("--true-int8", action="store_true",
                       help="Convert QAT models to true INT8 models for real quantization performance")
    
    args = parser.parse_args()
    
    benchmark = QuantizedModelBenchmark()
    if args.true_int8:
        benchmark.use_true_int8 = True
    
    if args.list_quantized:
        if QUANTIZED_MODELS_AVAILABLE:
            quantized_models = benchmark.get_available_quantized_models()
            print("Available quantized models:")
            for model in quantized_models:
                print(f"  - {model}")
        else:
            print("No quantized models available")
        return
    
    results = benchmark.run_benchmarks(args.models, args.batch_sizes, args.device)
    benchmark.print_summary(results)

if __name__ == "__main__":
    main()

'''
# Test with default models and batch sizes
python quant_bench.py

# Test specific models
python quant_bench.py --models resnet18 resnet50 --batch_sizes 1 4 8

# Test all models
python quant_bench.py --models resnet18 resnet50 bert-base-uncased vit --batch_sizes 1 2 4 8

# List available quantized models
python quant_bench.py --list-quantized

# Test standard models (FP32/FP16)
python quant_bench.py --models resnet18 resnet50

# Test standard models (FP32/FP16)
python quant_bench.py --models resnet18 resnet50

# Test true INT8 models (converted from standard models)
python quant_bench.py --models resnet18 resnet50 --true-int8

# Test mixed models
python quant_bench.py --models resnet18 resnet50 --batch-sizes 1 4 8
'''