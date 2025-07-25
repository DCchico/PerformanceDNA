#!/usr/bin/env python3
"""
BERT Attention Block Kernel Performance Comparison
Compares kernel latencies between Eager and Dynamo(Inductor) modes
"""

import matplotlib.pyplot as plt
import numpy as np

# Example placeholder values (replace with your actual measurements)
# Each: [Eager, Dynamo(Inductor)]
latencies_bs1 = [59.04, 58.94, 59.26, 125.09, 59.20, 16.90, 156.77, 12.77, 165.54, 16.90]
latencies_bs1_inductor = [59.46, 59.49, 59.42, 144.96, 41.89, 25.10, 159.01, 12.67, 136.99, 24.92]

latencies_bs32 = [0.92, 0.92, 0.92, 1.87, 0.92, 0.25, 3.25, 0.51, 3.20, 0.26]
latencies_bs32_inductor = [0.92, 0.93, 0.93, 2.09, 0.84, 0.20, 3.25, 0.51, 3.25, 0.19]

def create_kernel_comparison_plot():
    """Create bar plot comparing kernel performance between Eager and Dynamo(Inductor)."""
    
    # Kernel names in BERT attention block
    kernels = [
        "SGEMM1", "SGEMM2", "SGEMM3", "FMHA",
        "SGEMM4", "Elementwise", "SGEMM5", "LN+Elementwise", "SGEMM6", "LN+Elementwise"
    ]
    
    
    x = np.arange(len(kernels))
    width = 0.35
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Batch size 1 (microseconds)
    axes[0].bar(x - width/2, latencies_bs1, width, label='Eager', 
                color='lightgray', edgecolor='black', alpha=0.8)
    axes[0].bar(x + width/2, latencies_bs1_inductor, width, label='Dynamo(Inductor)', 
                color='lightgreen', edgecolor='black', alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(kernels, rotation=45, ha='right', fontsize=10)
    axes[0].set_ylabel('Avg Kernel Latency (μs)', fontsize=12)
    axes[0].set_title('Batch Size 1', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, axis='y', alpha=0.3)
    axes[0].tick_params(axis='both', which='major', labelsize=10)
    
    # Batch size 32 (milliseconds)
    axes[1].bar(x - width/2, latencies_bs32, width, label='Eager', 
                color='lightgray', edgecolor='black', alpha=0.8)
    axes[1].bar(x + width/2, latencies_bs32_inductor, width, label='Dynamo(Inductor)', 
                color='lightgreen', edgecolor='black', alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(kernels, rotation=45, ha='right', fontsize=10)
    axes[1].set_ylabel('Avg Kernel Latency (ms)', fontsize=12)
    axes[1].set_title('Batch Size 32', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, axis='y', alpha=0.3)
    axes[1].tick_params(axis='both', which='major', labelsize=10)
    
    fig.suptitle('BERT Attention Block: Kernel Latency Comparison', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    return fig

def main():
    """Main function to create and save the kernel comparison plot."""
    
    # Calculate totals for comparison
    total_bs1_eager = sum(latencies_bs1)
    total_bs1_inductor = sum(latencies_bs1_inductor)
    total_bs32_eager = sum(latencies_bs32)
    total_bs32_inductor = sum(latencies_bs32_inductor)
    
    print("=== BERT Attention Block Total Latencies ===")
    print(f"Batch Size 1:")
    print(f"  Eager: {total_bs1_eager:.2f} ms")
    print(f"  Dynamo(Inductor): {total_bs1_inductor:.2f} ms")
    print(f"  Speedup: {total_bs1_eager/total_bs1_inductor:.2f}x")
    print(f"  Improvement: {((total_bs1_eager - total_bs1_inductor) / total_bs1_eager * 100):.1f}%")
    print()
    print(f"Batch Size 32:")
    print(f"  Eager: {total_bs32_eager:.2f} ms")
    print(f"  Dynamo(Inductor): {total_bs32_inductor:.2f} ms")
    print(f"  Speedup: {total_bs32_eager/total_bs32_inductor:.2f}x")
    print(f"  Improvement: {((total_bs32_eager - total_bs32_inductor) / total_bs32_eager * 100):.1f}%")
    print()
    
    # Create the plot
    fig = create_kernel_comparison_plot()
    
    # Save the plot
    fig.savefig("../results/kernel_comparison_plot.png", dpi=300, bbox_inches='tight')
    fig.savefig("../results/kernel_comparison_plot.pdf", bbox_inches='tight')
    
    print("Kernel comparison plot saved to ../results/")
    plt.show()

if __name__ == "__main__":
    main()
