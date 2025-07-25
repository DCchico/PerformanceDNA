import torch
import torchvision.models as models
import time
import matplotlib.pyplot as plt

def benchmark_model(model, input_data, config_name, settings):
    model.eval()
    with torch.no_grad():
        # Warm-up
        for _ in range(10):
            _ = model(input_data)
        
        # Benchmark
        start_time = time.time()
        for _ in range(100):  # Run the model multiple times for averaging
            _ = model(input_data)
        end_time = time.time()
    avg_time = (end_time - start_time) * 1000 / 100
    print(f"{config_name} ({settings}): {avg_time:.6f} ms per inference")
    return avg_time

def run_benchmarks(model_name, model, input_data):
    # Store results
    results = []

    # Baseline (Default Settings)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    results.append(benchmark_model(model, input_data, f"{model_name} Baseline", "Eager Mode, cuDNN enabled, cuDNN benchmark disabled, cuDNN deterministic disabled, Float32 precision"))

    # TorchScript (Graph Mode)
    scripted_model = torch.jit.script(model)
    results.append(benchmark_model(scripted_model, input_data, f"{model_name} TorchScript Mode", "Graph Mode, cuDNN enabled, cuDNN benchmark disabled, cuDNN deterministic disabled, Float32 precision"))

    # cuDNN disabled
    torch.backends.cudnn.enabled = False
    results.append(benchmark_model(model, input_data, f"{model_name} cuDNN Disabled", "Eager Mode, cuDNN disabled, cuDNN benchmark disabled, cuDNN deterministic disabled, Float32 precision"))

    # cuDNN benchmark
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    results.append(benchmark_model(model, input_data, f"{model_name} cuDNN Benchmark Enabled", "Eager Mode, cuDNN enabled, cuDNN benchmark enabled, cuDNN deterministic disabled, Float32 precision"))

    # cuDNN deterministic
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    results.append(benchmark_model(model, input_data, f"{model_name} cuDNN Deterministic", "Eager Mode, cuDNN enabled, cuDNN benchmark disabled, cuDNN deterministic enabled, Float32 precision"))

    # Mixed Precision
    model.half()
    if model_name == "ResNet50":
        input_data = input_data.half()
    elif model_name == "RetinaNet":
        input_data = [img.half() for img in input_data]
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    results.append(benchmark_model(model, input_data, f"{model_name} Mixed Precision", "Eager Mode, cuDNN enabled, cuDNN benchmark disabled, cuDNN deterministic disabled, Float16 precision"))

    # Reset to default
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False

    return results

def plot_results(results, model_name):
    configurations = [
        "Baseline",
        "TorchScript Mode",
        "cuDNN Disabled",
        "cuDNN Benchmark Enabled",
        "cuDNN Deterministic",
        "Mixed Precision"
    ]

    # Calculate percentage differences from the baseline
    baseline_time = results[0]
    percent_differences = [(baseline_time - time) / baseline_time * 100 for time in results]

    # Plotting
    plt.figure(figsize=(10, 6))
    bars = plt.bar(configurations, results, color='skyblue')
    plt.ylabel('Inference Time (ms)')
    plt.title(f'{model_name} Inference Time for Different PyTorch Configurations')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Annotate bars with percentage differences
    for bar, percent_diff in zip(bars, percent_differences):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{percent_diff:.2f}%', ha='center', va='bottom')

    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right', fontsize=10)

    # Save the figure to a file
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_inference_times.png', bbox_inches='tight')

    # Show the plot
    plt.show()

if __name__ == "__main__":
    # ResNet50 Benchmark
    resnet_input = torch.randn(1, 3, 224, 224).cuda()
    resnet_model = models.resnet50().cuda()
    resnet_results = run_benchmarks("ResNet50", resnet_model, resnet_input)
    plot_results(resnet_results, "ResNet50")

    # RetinaNet Benchmark
    # Load the RetinaNet model
    # retinanet_input = torch.randn(1, 3, 800, 800).cuda()
    # retinanet_model = torch.load('retinanet50_32x4d_fpn.pth').cuda()
    retinanet_input = [torch.randn(3, 800, 800).cuda()]
    retinanet_model = models.detection.retinanet_resnet50_fpn(weights=models.detection.RetinaNet_ResNet50_FPN_Weights.DEFAULT).cuda()

    retinanet_results = run_benchmarks("RetinaNet", retinanet_model, retinanet_input)
    plot_results(retinanet_results, "RetinaNet")
