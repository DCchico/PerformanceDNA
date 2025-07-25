#!/usr/bin/env python3
"""
CNN Benchmark Results Visualization
Creates a 1x3 subplot figure showing latency comparison across models and backends
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

def load_and_process_data(csv_file):
    """Load and process the benchmark results data."""
    # Read CSV without headers and assign column names
    df = pd.read_csv(csv_file, header=None, names=[
        'model', 'batch_size', 'gpu', 'graph_compiler', 'backend', 'latency_ms'
    ])
    
    # Filter for batch sizes 1, 8, 32
    df_filtered = df[df['batch_size'].isin([1, 8, 32])].copy()
    
    # Create backend labels
    df_filtered['backend_label'] = df_filtered.apply(
        lambda row: f"{row['graph_compiler']} ({row['backend']})" 
        if row['backend'] != 'N/A' else row['graph_compiler'], 
        axis=1
    )
    
    return df_filtered

def create_plot(csv_file="../results/cnn_benchmark_results.csv"):
    """Create the main visualization."""
    
    # Load data
    df = load_and_process_data(csv_file)
    
    # Configuration
    models = ["resnet50", "densenet121", "efficientnet_b0", "mobilenet_v2"]
    batch_sizes = [1, 8, 32]
    backends = ["Eager", "TorchScript (Trace)", "TorchScript (Script)", "Dynamo (Inductor)", "TensorRT"]
    
    # Colors and styling
    colors = ["lightgray", "lightgreen", "gold", "plum", "skyblue"]
    hatches = [None] * len(backends)
    
    # Model display names
    model_names = {
        "resnet50": "ResNet50",
        "densenet121": "DenseNet121", 
        "efficientnet_b0": "EfficientNet-B0",
        "mobilenet_v2": "MobileNetV2"
    }
    
    # Create figure and layout - 3 rows, 1 column
    fig = plt.figure(figsize=(16, 15))
    gs = gridspec.GridSpec(3, 1, wspace=0.3, hspace=0.6)
    
    x = np.arange(len(models))
    width = 0.15  # Width of bars
    axes = []
    
    # Plot subfigures for each batch size
    for i, batch_size in enumerate(batch_sizes):
        ax = fig.add_subplot(gs[i, 0])
        axes.append(ax)
        
        # Get data for this batch size
        batch_data = df[df['batch_size'] == batch_size]
        
        # Prepare data for plotting
        for j, backend in enumerate(backends):
            if backend == "Eager":
                backend_data = batch_data[batch_data['graph_compiler'] == 'Eager']
            elif backend == "TorchScript (Trace)":
                backend_data = batch_data[
                    (batch_data['graph_compiler'] == 'TorchScript') & 
                    (batch_data['backend'] == 'Trace')
                ]
            elif backend == "TorchScript (Script)":
                backend_data = batch_data[
                    (batch_data['graph_compiler'] == 'TorchScript') & 
                    (batch_data['backend'] == 'Script')
                ]
            elif backend == "Dynamo (Inductor)":
                backend_data = batch_data[
                    (batch_data['graph_compiler'] == 'Dynamo') & 
                    (batch_data['backend'] == 'Inductor')
                ]
            elif backend == "TensorRT":
                backend_data = batch_data[
                    (batch_data['graph_compiler'] == 'TensorRT') & 
                    (batch_data['backend'] == 'TensorRT')
                ]
            
            # Extract latencies for each model
            latencies = []
            for model in models:
                model_data = backend_data[backend_data['model'] == model]
                if not model_data.empty:
                    latencies.append(model_data['latency_ms'].iloc[0])
                else:
                    latencies.append(0)  # Handle missing data
            
            # Plot bars
            ax.bar(x + j * width, latencies, width, 
                   color=colors[j], edgecolor='black', linewidth=0.5,
                   hatch=hatches[j] if hatches[j] else '',
                   label=backend if i == 0 else "",
                   alpha=0.8)
        
        # Customize subplot
        ax.set_xticks(x + width * 2)  # Center labels
        ax.set_xticklabels([model_names[m] for m in models], ha='right', fontsize=12)
        ax.set_ylabel("Latency (ms)", fontsize=14)
        ax.set_title(f"Batch Size {batch_size}", fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='both', which='major', labelsize=16)
        
        # Set y-axis limits based on data
        all_latencies = batch_data['latency_ms'].values
        y_max = max(all_latencies) * 1.1
        ax.set_ylim(0, y_max)
        
        # Add value labels on bars (optional, can be commented out for cleaner look)
        # for j, backend in enumerate(backends):
        #     for k, model in enumerate(models):
        #         # Get the latency value and add it as text on the bar
        #         pass
    
    # Add legend
    fig.legend(backends, loc='upper center', ncol=len(backends), 
               bbox_to_anchor=(0.5, 0.98), fontsize=16)
    
    # Adjust layout - give more space at the top for legend
    plt.tight_layout(rect=[0, 0.05, 1, 0.92])
    
    return fig



def main():
    """Main function to create and save the plot."""
    
    # Create the main plot
    fig = create_plot()
    fig.savefig("../results/cnn_benchmark_plot.png", dpi=300, bbox_inches='tight')
    fig.savefig("../results/cnn_benchmark_plot.pdf", bbox_inches='tight')
    
    print("Plot saved to ../results/cnn_benchmark_plot.png and .pdf")
    plt.show()

if __name__ == "__main__":
    main() 