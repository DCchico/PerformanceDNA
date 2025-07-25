import torch
from torchvision.models import resnet50, ResNet50_Weights

import warnings
warnings.filterwarnings("ignore")  # suppress minor warnings

def measure_latency(model, device, batch_size=1, timed_iterations=100, channels_last=False):
    """
    Measures average forward latency (ms) using CUDA event timing.
    If channels_last=True, we create the dummy input in channels-last format.
    """
    # Create dummy input in either default (NCHW) or channels-last format
    dummy_input = torch.randn(batch_size, 3, 224, 224, device=device)
    if channels_last:
        dummy_input = dummy_input.to(memory_format=torch.channels_last)

    # Warm-up
    warmup_iterations = 5
    for _ in range(warmup_iterations):
        _ = model(dummy_input)

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

    return total_time_ms / timed_iterations

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Base model
    base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(device)
    base_model.eval()

    # SCENARIOS
    scenarios = []

    # 1) Eager Mode (Default memory format)
    scenarios.append(("Eager_Default", base_model, False))

    # 2) Eager Mode (Channels-Last)
    #    - We clone/resnet50 again to avoid messing up the original model
    #    - Then we call .to(memory_format=torch.channels_last) on the model
    cl_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(device)
    cl_model.eval()
    cl_model.to(memory_format=torch.channels_last)
    scenarios.append(("Eager_ChannelsLast", cl_model, True))

    # Benchmark each scenario
    batch_size = 1
    timed_iterations = 1000
    results = []

    for name, model_instance, use_channels_last in scenarios:
        print(f"\n--- Benchmarking Scenario: {name} ---")
        latency_ms = measure_latency(
            model_instance, device,
            batch_size=batch_size,
            timed_iterations=timed_iterations,
            channels_last=use_channels_last
        )
        print(f"  Average latency: {latency_ms:.2f} ms")
        results.append((name, latency_ms))

    # Print summary
    print("\n=== Final Results ===")
    for name, lat in results:
        print(f"{name}: {lat:.2f} ms")

if __name__ == "__main__":
    main()
