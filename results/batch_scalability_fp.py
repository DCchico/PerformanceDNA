import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def load_and_process_data(csv_file):
    """Load CSV data and normalize latencies to batch size 1."""
    df = pd.read_csv(csv_file)
    
    # Calculate normalized latency (relative to batch size 1)
    normalized_data = []
    
    for model in df['Model'].unique():
        for quant in df['Quantization'].unique():
            for compile_mode in df['Compilation'].unique():
                # Get data for this combination
                mask = (df['Model'] == model) & (df['Quantization'] == quant) & (df['Compilation'] == compile_mode)
                subset = df[mask].copy()
                
                if len(subset) == 0:
                    continue
                
                # Get batch size 1 latency as baseline
                baseline = subset[subset['Batch_Size'] == 1]['Latency_ms'].iloc[0]
                
                # Calculate normalized latency
                subset['Normalized_Latency'] = subset['Latency_ms'] / baseline
                normalized_data.append(subset)
    
    return pd.concat(normalized_data, ignore_index=True)

def create_scalability_plots(df, output_file='batch_scalability.png'):
    """Create scalability plots for FP32 and FP16."""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Get unique batch sizes and models
    batch_sizes = sorted(df['Batch_Size'].unique())
    models = df['Model'].unique()
    
    # Colors for models
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Plot FP32 scalability
    ax1.set_title('FP32 Batch Scalability (Normalized to BS=1)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Batch Size', fontsize=12)
    ax1.set_ylabel('Normalized Latency', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    fp32_yvals = []
    for i, model in enumerate(models):
        # FP32 Eager
        fp32_eager = df[(df['Model'] == model) & (df['Quantization'] == 'FP32') & (df['Compilation'] == 'Eager')]
        if len(fp32_eager) > 0:
            ax1.plot(fp32_eager['Batch_Size'], fp32_eager['Normalized_Latency'], 
                    marker='o', linewidth=2, label=f'{model} Eager', 
                    color=colors[i], linestyle='-')
            fp32_yvals.extend(fp32_eager['Normalized_Latency'].values)
        
        # FP32 Compiled
        fp32_compiled = df[(df['Model'] == model) & (df['Quantization'] == 'FP32') & (df['Compilation'] == 'torch.compile')]
        if len(fp32_compiled) > 0:
            ax1.plot(fp32_compiled['Batch_Size'], fp32_compiled['Normalized_Latency'], 
                    marker='s', linewidth=2, label=f'{model} Compiled', 
                    color=colors[i], linestyle='--')
            fp32_yvals.extend(fp32_compiled['Normalized_Latency'].values)
    
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.set_xlim(0.5, max(batch_sizes) + 0.5)
    if fp32_yvals:
        ax1.set_ylim(min(fp32_yvals) * 0.95, max(fp32_yvals) * 1.05)
    
    # Plot FP16 scalability
    ax2.set_title('FP16 Batch Scalability (Normalized to BS=1)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Batch Size', fontsize=12)
    ax2.set_ylabel('Normalized Latency', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    fp16_yvals = []
    for i, model in enumerate(models):
        # FP16 Eager
        fp16_eager = df[(df['Model'] == model) & (df['Quantization'] == 'FP16') & (df['Compilation'] == 'Eager')]
        if len(fp16_eager) > 0:
            ax2.plot(fp16_eager['Batch_Size'], fp16_eager['Normalized_Latency'], 
                    marker='o', linewidth=2, label=f'{model} Eager', 
                    color=colors[i], linestyle='-')
            fp16_yvals.extend(fp16_eager['Normalized_Latency'].values)
        
        # FP16 Compiled
        fp16_compiled = df[(df['Model'] == model) & (df['Quantization'] == 'FP16') & (df['Compilation'] == 'torch.compile')]
        if len(fp16_compiled) > 0:
            ax2.plot(fp16_compiled['Batch_Size'], fp16_compiled['Normalized_Latency'], 
                    marker='s', linewidth=2, label=f'{model} Compiled', 
                    color=colors[i], linestyle='--')
            fp16_yvals.extend(fp16_compiled['Normalized_Latency'].values)
    
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.set_xlim(0.5, max(batch_sizes) + 0.5)
    if fp16_yvals:
        ax2.set_ylim(min(fp16_yvals) * 0.95, max(fp16_yvals) * 1.05)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_summary_table(df):
    """Create a summary table showing scaling efficiency."""
    print("\n" + "="*80)
    print("BATCH SCALING EFFICIENCY SUMMARY")
    print("="*80)
    
    summary_data = []
    
    for model in df['Model'].unique():
        for quant in df['Quantization'].unique():
            for compile_mode in df['Compilation'].unique():
                subset = df[(df['Model'] == model) & (df['Quantization'] == quant) & (df['Compilation'] == compile_mode)]
                
                if len(subset) < 2:  # Need at least 2 batch sizes
                    continue
                
                # Calculate scaling efficiency (how close to linear scaling)
                bs1_latency = subset[subset['Batch_Size'] == 1]['Latency_ms'].iloc[0]
                bs8_latency = subset[subset['Batch_Size'] == 8]['Latency_ms'].iloc[0]
                
                # Ideal scaling would be 8x latency for 8x batch size
                ideal_bs8_latency = bs1_latency * 8
                scaling_efficiency = (ideal_bs8_latency / bs8_latency) * 100
                
                summary_data.append({
                    'Model': model,
                    'Quantization': quant,
                    'Compilation': compile_mode,
                    'BS1_Latency': bs1_latency,
                    'BS8_Latency': bs8_latency,
                    'Scaling_Efficiency': scaling_efficiency
                })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Print formatted table
    print(f"{'Model':<10} {'Quant':<8} {'Compile':<12} {'BS1(ms)':<8} {'BS8(ms)':<8} {'Efficiency':<12}")
    print("-" * 80)
    
    for _, row in summary_df.iterrows():
        print(f"{row['Model']:<10} {row['Quantization']:<8} {row['Compilation']:<12} "
              f"{row['BS1_Latency']:<8.3f} {row['BS8_Latency']:<8.3f} {row['Scaling_Efficiency']:<12.1f}%")
    
    print("="*80)
    print("Note: Efficiency > 100% means better than linear scaling (sub-linear)")
    print("      Efficiency < 100% means worse than linear scaling (super-linear)")
    
    return summary_df

def main():
    # Load and process data
    csv_file = '/usr/scratch/difei/PerformanceDNA/results/quantization_benchmark_20250703_110536.csv'
    
    try:
        df = load_and_process_data(csv_file)
        print(f"Loaded data from {csv_file}")
        print(f"Found {len(df)} data points across {len(df['Model'].unique())} models")
        
        # Create plots
        fig = create_scalability_plots(df, 'batch_scalability_analysis.png')
        
        # Create summary table
        summary_df = create_summary_table(df)
        
        # Save summary to CSV
        summary_df.to_csv('batch_scaling_fp.csv', index=False)
        print(f"\nSummary saved to: batch_scaling_summary.csv")
        
    except FileNotFoundError:
        print(f"Error: Could not find {csv_file}")
        print("Please make sure the CSV file is in the current directory.")
    except Exception as e:
        print(f"Error processing data: {e}")

if __name__ == "__main__":
    main()
