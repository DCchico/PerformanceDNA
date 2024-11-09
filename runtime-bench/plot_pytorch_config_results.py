import matplotlib.pyplot as plt
import numpy as np

# Benchmark results
configurations = [
    "Baseline",
    "TorchScript Mode",
    "cuDNN Disabled",
    "cuDNN Benchmark Enabled",
    "cuDNN Deterministic",
    "Mixed Precision"
]

times = [
    2.242608,  # Baseline
    2.138135,  # TorchScript Mode
    2.070172,  # cuDNN Disabled
    2.138286,  # cuDNN Benchmark Enabled
    2.098467,  # cuDNN Deterministic
    1.673012   # Mixed Precision
]

# Calculate percentage differences from the baseline
baseline_time = times[0]
percent_differences = [(baseline_time - time) / baseline_time * 100 for time in times]

# Plotting
plt.figure(figsize=(10, 6))
bars = plt.bar(configurations, times, color='skyblue')
plt.ylabel('Inference Time (ms)')
plt.title('ResNet50 Inference Time for Different PyTorch Configurations')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Annotate bars with percentage differences
for bar, percent_diff in zip(bars, percent_differences):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{percent_diff:.2f}%', ha='center', va='bottom')

# Rotate x-axis labels
plt.xticks(rotation=45, ha='right', fontsize=10)

# Save the figure to a file
plt.savefig('resnet50_inference_times.png', bbox_inches='tight')

# Show the plot
plt.show()
