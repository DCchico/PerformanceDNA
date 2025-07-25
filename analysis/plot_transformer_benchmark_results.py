#!/usr/bin/env python3
"""
Transformer Benchmark Results Visualization
Creates a 3x1 subplot figure showing latency comparison across models and backends
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

def load_and_process_data(csv_file):
    """Load and process the benchmark results data."""
    # Read CSV with headers
    df = pd.read_csv(csv_file)
    
    # Filter for batch sizes 1, 8, 32 and exclude GPT-2-Medium
    df_filtered = df[
        (df['Batch_Size'].isin([1, 8, 32])) & 
        (~df['Model'].isin(['gpt2-medium']))
    ].copy()
    
    return df_filtered

def create_plot(csv_file="../results/cache_managed_transformer_benchmark_20250703_144113.csv"):
    """Create the main visualization."""
    
    # Load data
    df = load_and_process_data(csv_file)
    
    # Configuration
    models = ["bert-base-uncased", "roberta-base", "distilbert-base-uncased", "gpt2", "vit", "electra-small", "gcn"]
    batch_sizes = [1, 8, 32]
    backends = ["Eager", "torch.compile"]
    
    # Colors and styling
    colors = ["lightgray", "lightgreen"]
    hatches = [None] * len(backends)
    
    # Model display names
    model_names = {
        "bert-base-uncased": "BERT-Base",
        "roberta-base": "RoBERTa-Base", 
        "distilbert-base-uncased": "DistilBERT",
        "gpt2": "GPT-2",
        "vit": "ViT",
        "electra-small": "ELECTRA-Small",
        "gcn": "GCN"
    }
    
    # Create figure and layout - 3 rows, 1 column
    fig = plt.figure(figsize=(16, 15))
    gs = gridspec.GridSpec(3, 1, wspace=0.3, hspace=0.6)
    
    x = np.arange(len(models))
    width = 0.35  # Width of bars (wider since only 2 backends)
    axes = []
    
    # Plot subfigures for each batch size
    for i, batch_size in enumerate(batch_sizes):
        ax = fig.add_subplot(gs[i, 0])
        axes.append(ax)
        
        # Get data for this batch size
        batch_data = df[df['Batch_Size'] == batch_size]
        
        # Prepare data for plotting
        for j, backend in enumerate(backends):
            backend_data = batch_data[batch_data['Compilation'] == backend]
            
            # Extract latencies for each model
            latencies = []
            for model in models:
                model_data = backend_data[backend_data['Model'] == model]
                if not model_data.empty:
                    latencies.append(model_data['Latency_ms'].iloc[0])
                else:
                    latencies.append(0)  # Handle missing data
            
            # Plot bars
            ax.bar(x + j * width, latencies, width, 
                   color=colors[j], edgecolor='black', linewidth=0.5,
                   hatch=hatches[j] if hatches[j] else '',
                   label=backend if i == 0 else "",
                   alpha=0.8)
        
        # Customize subplot
        ax.set_xticks(x + width * 0.5)  # Center labels
        ax.set_xticklabels([model_names[m] for m in models], rotation=45, ha='right', fontsize=12)
        ax.set_ylabel("Latency (ms)", fontsize=14)
        ax.set_title(f"Batch Size {batch_size}", fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='both', which='major', labelsize=16)
        
        # Set y-axis limits manually for better visibility
        if batch_size == 1:
            ax.set_ylim(0, 6)  # 0-25ms for batch size 1
        elif batch_size == 8:
            ax.set_ylim(0, 40)  # 0-60ms for batch size 8
        elif batch_size == 32:
            ax.set_ylim(0, 180)  # 0-150ms for batch size 32
    
    # Add legend
    fig.legend([backends[0], "Dynamo(Inductor)"], loc='upper center', ncol=len(backends), 
               bbox_to_anchor=(0.5, 0.98), fontsize=16)
    
    # Adjust layout - give more space at the top for legend
    plt.tight_layout(rect=[0, 0.05, 1, 0.92])
    
    return fig

def main():
    """Main function to create and save the plot."""
    
    # Create the main plot
    fig = create_plot()
    fig.savefig("../results/transformer_benchmark_plot.png", dpi=300, bbox_inches='tight')
    fig.savefig("../results/transformer_benchmark_plot.pdf", bbox_inches='tight')
    
    print("Plot saved to ../results/transformer_benchmark_plot.png and .pdf")
    plt.show()

if __name__ == "__main__":
    main() 