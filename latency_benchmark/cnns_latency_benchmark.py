import torch
import torch.nn as nn
import time
import argparse
import csv
import os
from torchvision.models import (
    resnet50, ResNet50_Weights,
    densenet121, DenseNet121_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
    mobilenet_v2, MobileNet_V2_Weights
)
import warnings
warnings.filterwarnings("ignore")

CNN_MODELS = {
    'resnet50': (resnet50, ResNet50_Weights.IMAGENET1K_V2, (3, 224, 224)),
    'densenet121': (densenet121, DenseNet121_Weights.IMAGENET1K_V1, (3, 224, 224)),
    'efficientnet_b0': (efficientnet_b0, EfficientNet_B0_Weights.IMAGENET1K_V1, (3, 224, 224)),
    'mobilenet_v2': (mobilenet_v2, MobileNet_V2_Weights.IMAGENET1K_V1, (3, 224, 224)),
}

def benchmark(model, dummy_inputs, timed_iterations):
    for _ in range(10):
        _ = model(*dummy_inputs)
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(timed_iterations):
        _ = model(*dummy_inputs)
    end_event.record()
    torch.cuda.synchronize()
    total_time_ms = start_event.elapsed_time(end_event)
    avg_latency_ms = total_time_ms / timed_iterations
    return avg_latency_ms

def get_model_and_input(model_name, batch_size, device="cuda"):
    if model_name not in CNN_MODELS:
        raise ValueError(f"Model '{model_name}' not supported.")
    model_fn, weights, input_shape = CNN_MODELS[model_name]
    model = model_fn(weights=weights).eval().to(device)
    dummy_input = torch.randn(batch_size, *input_shape, device=device)
    return model, (dummy_input,)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("CUDA not available. Benchmarking on CPU is not meaningful for latency comparison.")
        return
    print(f"Using device: {device}")
    results_dir = "../results"
    os.makedirs(results_dir, exist_ok=True)
    results_filepath = os.path.join(results_dir, "cnn_benchmark_results.csv")
    write_header = not os.path.exists(results_filepath)
    print(f"Appending results to: {results_filepath}")
    gpu_name = torch.cuda.get_device_name(0).replace(" ", "_")
    try:
        import torch_tensorrt
        TRT_AVAILABLE = True
    except ImportError:
        TRT_AVAILABLE = False
    for batch_size in args.batch_sizes:
        print(f"\n{'='*20} RUNNING BENCHMARK FOR BATCH SIZE: {batch_size} {'='*20}")
        try:
            base_model, dummy_inputs = get_model_and_input(args.model, batch_size, device)
            results_data = []
            if hasattr(args, 'mode') and args.mode:
                modes_to_run = [args.mode]
            else:
                modes_to_run = ['eager', 'torchscript_trace', 'torchscript_script', 'inductor']
            # Eager
            if 'eager' in modes_to_run:
                print("\n[1] Benchmarking Eager Mode...")
                eager_latency = benchmark(base_model, dummy_inputs, args.iterations)
                print(f"    -> Average Latency: {eager_latency:.3f} ms")
                results_data.append([args.model, batch_size, gpu_name, "Eager", "N/A", eager_latency])
            # TorchScript Trace
            if 'torchscript_trace' in modes_to_run:
                print("\n[2] Benchmarking TorchScript Trace...")
                try:
                    traced_model = torch.jit.trace(base_model, dummy_inputs)
                    ts_latency = benchmark(traced_model, dummy_inputs, args.iterations)
                    print(f"    -> Average Latency: {ts_latency:.3f} ms")
                except Exception as e:
                    print(f"    -> Failed to benchmark TorchScript Trace: {e}")
                    ts_latency = float('inf')
                results_data.append([args.model, batch_size, gpu_name, "TorchScript", "Trace", ts_latency])
            # TorchScript Script
            if 'torchscript_script' in modes_to_run:
                print("\n[3] Benchmarking TorchScript Script...")
                try:
                    scripted_model = torch.jit.script(base_model)
                    ts_script_latency = benchmark(scripted_model, dummy_inputs, args.iterations)
                    print(f"    -> Average Latency: {ts_script_latency:.3f} ms")
                except Exception as e:
                    print(f"    -> Failed to benchmark TorchScript Script: {e}")
                    ts_script_latency = float('inf')
                results_data.append([args.model, batch_size, gpu_name, "TorchScript", "Script", ts_script_latency])
            # TorchInductor
            if 'inductor' in modes_to_run:
                print("\n[4] Benchmarking TorchInductor (torch.compile)...")
                try:
                    compiled_model = torch.compile(base_model, backend="inductor")
                    inductor_latency = benchmark(compiled_model, dummy_inputs, args.iterations)
                    print(f"    -> Average Latency: {inductor_latency:.3f} ms")
                except Exception as e:
                    print(f"    -> Failed to benchmark TorchInductor: {e}")
                    inductor_latency = float('inf')
                results_data.append([args.model, batch_size, gpu_name, "Dynamo", "Inductor", inductor_latency])
            # TensorRT
            if 'tensorrt' in modes_to_run:
                print("\n[5] Benchmarking TensorRT (torch_tensorrt.compile)...")
                if not TRT_AVAILABLE:
                    print("    -> Torch-TensorRT is not installed. Skipping TensorRT mode.")
                    trt_latency = float('inf')
                else:
                    try:
                        trt_model = torch_tensorrt.compile(
                            base_model,
                            inputs=[torch_tensorrt.Input(dummy_inputs[0].shape, dtype=dummy_inputs[0].dtype, device={'device_type': 'cuda', 'device_index': 0})],
                            enabled_precisions={torch.float32},  # Change to {torch.float16} for FP16 if desired
                            workspace_size=1 << 22
                        )
                        trt_latency = benchmark(trt_model, dummy_inputs, args.iterations)
                        print(f"    -> Average Latency: {trt_latency:.3f} ms")
                    except Exception as e:
                        print(f"    -> Failed to benchmark TensorRT: {e}")
                        trt_latency = float('inf')
                results_data.append([args.model, batch_size, gpu_name, "TensorRT", "TensorRT", trt_latency])
            # Write results
            with open(results_filepath, 'a', newline='') as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(["model", "batch_size", "hardware", "graph_compiler", "backend", "latency_ms"])
                    write_header = False
                writer.writerows(results_data)
        except torch.cuda.OutOfMemoryError:
            print(f"\n!!! OUT OF MEMORY on batch size {batch_size}. Stopping benchmark for this size. !!!")
            results_data = [
                [args.model, batch_size, gpu_name, "Eager", "N/A", "OOM"],
                [args.model, batch_size, gpu_name, "TorchScript", "Trace", "OOM"],
                [args.model, batch_size, gpu_name, "TorchScript", "Script", "OOM"],
                [args.model, batch_size, gpu_name, "Dynamo", "Inductor", "OOM"],
            ]
            with open(results_filepath, 'a', newline='') as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(["model", "batch_size", "hardware", "graph_compiler", "backend", "latency_ms"])
                    write_header = False
                writer.writerows(results_data)
        except Exception as e:
            print(f"\n!!! An unexpected error occurred on batch size {batch_size}: {e}. Skipping. !!!")
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CNN Inference Latency Benchmark")
    parser.add_argument("--model", type=str, default="resnet50", choices=list(CNN_MODELS.keys()), help="CNN model to benchmark")
    parser.add_argument("--iterations", type=int, default=100, help="Number of timed iterations for benchmark")
    parser.add_argument("--batch_sizes", type=int, nargs='+', default=[1, 2, 4, 8, 16, 32], help="List of batch sizes to sweep through")
    parser.add_argument("--mode", type=str, choices=['eager', 'torchscript_trace', 'torchscript_script', 'inductor', 'tensorrt'], help="Run only specific benchmark mode (default: run all modes)")
    args = parser.parse_args()
    main(args)
