import torch
import torch.nn as nn
import time
import argparse
from torchvision.models import resnet50, ResNet50_Weights
from transformers import BertModel, ViTModel, AlignModel
import csv
from datetime import datetime
import os

# Suppress minor warnings
import warnings
warnings.filterwarnings("ignore")

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=False # More common for performance benchmarks
        )

    def forward(self, x):
        # LSTM returns output, (hidden_state, cell_state)
        # We only care about the output tensor for the benchmark
        output, _ = self.lstm(x)
        return output

def benchmark(model, dummy_inputs, timed_iterations, model_name, batch_size, device, input_sequence=None):
    """
    Measures the average inference latency of a model.
    For dynamic models, uses pre-generated input sequence for fair comparison.
    """
    # Warm-up phase to let CUDA initialize and optimize
    for _ in range(10):
        _ = model(*dummy_inputs)

    # Synchronize to ensure warm-up is complete before timing
    torch.cuda.synchronize()

    # Timing using CUDA events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for i in range(timed_iterations):
        # For dynamic models, use pre-generated inputs
        if model_name in ["bert-base-uncased", "lstm", "align"] and input_sequence is not None:
            _ = model(*input_sequence[i])
        else:
            # For fixed-shape models, use same inputs
            _ = model(*dummy_inputs)
    end_event.record()

    # Wait for all kernels to complete
    torch.cuda.synchronize()

    # Calculate the total time and average latency
    total_time_ms = start_event.elapsed_time(end_event)
    avg_latency_ms = total_time_ms / timed_iterations
    return avg_latency_ms

def generate_input_sequence(model_name, batch_size, device, num_iterations):
    """
    Pre-generate a fixed sequence of inputs for fair comparison across all methods.
    """
    input_sequence = []
    
    for i in range(num_iterations):
        if model_name == "bert-base-uncased":
            # Vary sequence length between 128 and 512
            seq_length = 128 + (i % 10) * 40  # 128, 168, 208, ..., 488, 128, ...
            input_sequence.append((torch.randint(0, 30000, (batch_size, seq_length), dtype=torch.long, device=device),))
        
        elif model_name == "lstm":
            # Vary sequence length between 256 and 768
            seq_length = 256 + (i % 8) * 64  # 256, 320, 384, ..., 704, 256, ...
            input_size = 256
            input_sequence.append((torch.randn(seq_length, batch_size, input_size, device=device),))
        
        elif model_name == "align":
            # Vary image size and text length
            image_size = 224 + (i % 4) * 32  # 224, 256, 288, 320, 224, ...
            text_length = 64 + (i % 3) * 8   # 64, 72, 80, 64, ...
            
            image_input = torch.randn(batch_size, 3, image_size, image_size, device=device)
            text_input = torch.randint(0, 30000, (batch_size, text_length), dtype=torch.long, device=device)
            input_sequence.append((text_input, image_input))
        
        else:
            raise ValueError(f"Model '{model_name}' does not support dynamic inputs.")
    
    return input_sequence

def get_model_and_input(model_name, batch_size, device="cuda"):
    """
    Loads a pretrained model or creates a custom one, along with dummy inputs.
    For dynamic models, returns a representative input shape.
    """
    if model_name == "resnet50":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval()
        dummy_input = torch.randn(batch_size, 3, 224, 224)
        return model, (dummy_input,) # Return as a tuple
        
    elif model_name == "bert-base-uncased":
        model = BertModel.from_pretrained("bert-base-uncased").eval()
        # Use a representative sequence length for initial setup
        sequence_length = 256
        dummy_input = torch.randint(0, 30000, (batch_size, sequence_length), dtype=torch.long)
        return model, (dummy_input,) # Return as a tuple

    elif model_name == "vit":
        model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k").eval()
        dummy_input = torch.randn(batch_size, 3, 224, 224)
        return model, (dummy_input,) # Return as a tuple
    
    elif model_name == "align":
        model = AlignModel.from_pretrained("kakaobrain/align-base").eval()
        # Use representative sizes for initial setup
        image_input = torch.randn(batch_size, 3, 256, 256)
        text_input = torch.randint(0, 30000, (batch_size, 72), dtype=torch.long)
        return model, (text_input, image_input) # Return a tuple of two inputs

    elif model_name == "lstm":
        # Define our custom LSTM architecture
        input_size = 256
        hidden_size = 256
        num_layers = 10
        # Use a representative sequence length for initial setup
        seq_length = 384
        
        model = SimpleLSTM(input_size, hidden_size, num_layers).eval()
        dummy_input = torch.randn(seq_length, batch_size, input_size)
        return model, (dummy_input,)

    else:
        raise ValueError(f"Model '{model_name}' not supported.")

def main(args):
    """
    Main function to run the benchmark scenarios across different batch sizes.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("CUDA not available. Benchmarking on CPU is not meaningful for latency comparison.")
        return

    print(f"Using device: {device}")
    
    # --- Setup Results File ---
    results_dir = "../results"
    os.makedirs(results_dir, exist_ok=True)
    results_filepath = os.path.join(results_dir, "benchmark_results_dynamic.csv")
    write_header = not os.path.exists(results_filepath)
    print(f"Appending results to: {results_filepath}")
    
    gpu_name = torch.cuda.get_device_name(0).replace(" ", "_")
    
    # --- Main Batch Size Loop ---
    for batch_size in args.batch_sizes:
        print(f"\n{'='*20} RUNNING BENCHMARK FOR BATCH SIZE: {batch_size} {'='*20}")
        
        try:
            # Get the model and a tuple of its input tensors
            base_model, dummy_inputs = get_model_and_input(args.model, batch_size, device)
            base_model.to(device)
            dummy_inputs = tuple(t.to(device) for t in dummy_inputs)

            # Generate input sequence for dynamic models (same sequence for all methods)
            input_sequence = None
            if args.model in ["bert-base-uncased", "lstm", "align"]:
                print(f"    -> Generating input sequence for {args.iterations} iterations...")
                input_sequence = generate_input_sequence(args.model, batch_size, device, args.iterations)

            # --- Benchmark and Collect Data ---
            results_data = []

            # Scenario 1: Eager Mode
            print("\n[1] Benchmarking Eager Mode...")
            eager_latency = benchmark(base_model, dummy_inputs, args.iterations, args.model, batch_size, device, input_sequence)
            print(f"    -> Average Latency: {eager_latency:.3f} ms")
            results_data.append([args.model, batch_size, gpu_name, "Eager", "N/A", eager_latency])

            # Scenario 2: TorchScript Trace
            print("\n[2] Benchmarking TorchScript Trace...")
            try:
                traced_model = torch.jit.trace(base_model, dummy_inputs)
                ts_latency = benchmark(traced_model, dummy_inputs, args.iterations, args.model, batch_size, device, input_sequence)
                print(f"    -> Average Latency: {ts_latency:.3f} ms")
            except Exception as e:
                print(f"    -> Failed to benchmark TorchScript Trace: {e}")
                ts_latency = float('inf')
            results_data.append([args.model, batch_size, gpu_name, "TorchScript", "Trace", ts_latency])

            # Scenario 3: TorchScript Script
            print("\n[3] Benchmarking TorchScript Script...")
            try:
                scripted_model = torch.jit.script(base_model)
                ts_script_latency = benchmark(scripted_model, dummy_inputs, args.iterations, args.model, batch_size, device, input_sequence)
                print(f"    -> Average Latency: {ts_script_latency:.3f} ms")
            except Exception as e:
                print(f"    -> Failed to benchmark TorchScript Script: {e}")
                ts_script_latency = float('inf')
            results_data.append([args.model, batch_size, gpu_name, "TorchScript", "Script", ts_script_latency])

            # Scenario 4: TorchInductor
            print("\n[4] Benchmarking TorchInductor (torch.compile)...")
            try:
                compiled_model = torch.compile(base_model, backend="inductor")
                inductor_latency = benchmark(compiled_model, dummy_inputs, args.iterations, args.model, batch_size, device, input_sequence)
                print(f"    -> Average Latency: {inductor_latency:.3f} ms")
            except Exception as e:
                print(f"    -> Failed to benchmark TorchInductor: {e}")
                inductor_latency = float('inf')
            results_data.append([args.model, batch_size, gpu_name, "Dynamo", "Inductor", inductor_latency])

            # --- Write results to CSV ---
            with open(results_filepath, 'a', newline='') as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(["model", "batch_size", "hardware", "graph_compiler", "backend", "latency_ms"])
                    write_header = False # Ensure header is only written once per script run
                writer.writerows(results_data)
            
        except torch.cuda.OutOfMemoryError:
            print(f"\n!!! OUT OF MEMORY on batch size {batch_size}. Stopping benchmark for this size. !!!")
            # Optionally write OOM result to CSV
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
    parser = argparse.ArgumentParser(description="PyTorch Inference Latency Benchmark")
    parser.add_argument("--model", type=str, default="resnet50", help="Model to benchmark (e.g., resnet50, bert-base-uncased, vit, align, lstm)")
    parser.add_argument("--iterations", type=int, default=100, help="Number of timed iterations for benchmark")
    parser.add_argument("--batch_sizes", type=int, nargs='+', default=[1, 2, 4, 8, 16, 32], help="List of batch sizes to sweep through")
    args = parser.parse_args()
    main(args)