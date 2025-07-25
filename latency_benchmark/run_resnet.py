import torch
from torchvision.models import resnet50, ResNet50_Weights
import torch_tensorrt

import warnings
warnings.filterwarnings("ignore")  # suppress minor warnings

def measure_latency(model, device, batch_size=1, timed_iterations=1000):
    """Measures average forward latency (ms) using CUDA event timing."""
    dummy_input = torch.randn(batch_size, 3, 224, 224, device=device)

    # Warm-up
    warmup_iterations = 5
    for _ in range(warmup_iterations):
        model(dummy_input)

    # Timed runs
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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Base model (Eager)
    base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(device)
    base_model.eval()

    # We'll define scenarios. Some might fail if the environment doesn't support them.
    scenarios = []

    # 1) Plain Eager Mode
    scenarios.append(("Eager", base_model))

    # 2) TorchScript Trace
    try:
        # For tracing, we do a single pass with a sample input
        trace_input = torch.randn(1, 3, 224, 224, device=device)
        traced_model = torch.jit.trace(base_model, trace_input)
        traced_model.eval()
        scenarios.append(("TorchScript_Trace", traced_model))
    except Exception as e:
        print("Failed to create TorchScript trace:", e)

    # 3) TorchScript Script
    try:
        scripted_model = torch.jit.script(base_model)
        scripted_model.eval()
        scenarios.append(("TorchScript_Script", scripted_model))
    except Exception as e:
        print("Failed to create TorchScript script:", e)

    # 4) AOT Eager
    try:
        aot_eager_model = torch.compile(base_model, backend="aot_eager")
        scenarios.append(("Compile_aot_eager", aot_eager_model))
    except Exception as e:
        print("Failed to compile with aot_eager:", e)

    # 5) Tensorrt
    try:
        tensorrt_model = torch.compile(base_model, backend="torch_tensorrt")
        scenarios.append(("Compile_torch_tensorrt", tensorrt_model))
    except Exception as e:
        print("Failed to compile with torch_tensorrt:", e)

    # 6) CUDAgraphs
    try:
        cudagraphs_model = torch.compile(base_model, backend="cudagraphs")
        scenarios.append(("Compile_cudagraphs", cudagraphs_model))
    except Exception as e:
        print("Failed to compile with cudagraphs:", e)

    # Benchmark each scenario
    results = []
    batch_size = 1
    timed_iterations = 100

    for name, model_instance in scenarios:
        print(f"\n--- Benchmarking Scenario: {name} ---")
        latency_ms = measure_latency(model_instance, device, batch_size, timed_iterations)
        print(f"  Average latency: {latency_ms:.2f} ms")
        results.append((name, latency_ms))

    # Print summary
    print("\n=== Final Results ===")
    for name, lat in results:
        print(f"{name}: {lat:.2f} ms")

if __name__ == "__main__":
    main()
