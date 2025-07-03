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

def create_scalability_plots(df, output_file='int8_batch_scalability.png'):
    """Create scalability plots for INT8 models with separate subplots for CNNs and transformers."""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Get unique batch sizes and models
    batch_sizes = sorted(df['Batch_Size'].unique())
    models = df['Model'].unique()
    
    # Separate CNNs and transformers
    cnn_models = [model for model in models if 'resnet' in model.lower()]
    transformer_models = [model for model in models if model not in cnn_models]
    
    # Colors for models
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Plot CNNs scalability
    ax1.set_title('INT8 CNNs Batch Scalability (Normalized to BS=1)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Batch Size', fontsize=12)
    ax1.set_ylabel('Normalized Latency', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Add ideal scaling line (linear scaling)
    ax1.plot(batch_sizes, batch_sizes, 'k--', alpha=0.5, linewidth=2, label='Ideal Linear Scaling')
    
    cnn_yvals = []
    for i, model in enumerate(cnn_models):
        # INT8 Eager
        int8_eager = df[(df['Model'] == model) & (df['Quantization'] == 'TRUE_INT8') & (df['Compilation'] == 'Eager')]
        if len(int8_eager) > 0:
            ax1.plot(int8_eager['Batch_Size'], int8_eager['Normalized_Latency'], 
                    marker='o', linewidth=2, label=f'{model} Eager', 
                    color=colors[i], linestyle='-', markersize=8)
            cnn_yvals.extend(int8_eager['Normalized_Latency'].values)
        
        # INT8 Compiled
        int8_compiled = df[(df['Model'] == model) & (df['Quantization'] == 'TRUE_INT8') & (df['Compilation'] == 'torch.compile')]
        if len(int8_compiled) > 0:
            ax1.plot(int8_compiled['Batch_Size'], int8_compiled['Normalized_Latency'], 
                    marker='s', linewidth=2, label=f'{model} Compiled', 
                    color=colors[i], linestyle='--', markersize=8)
            cnn_yvals.extend(int8_compiled['Normalized_Latency'].values)
    
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax1.set_xlim(0.5, max(batch_sizes) + 0.5)
    if cnn_yvals:
        ax1.set_ylim(min(cnn_yvals) * 0.95, max(cnn_yvals) * 1.05)
    
    # Plot Transformers scalability
    ax2.set_title('INT8 Transformers Batch Scalability (Normalized to BS=1)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Batch Size', fontsize=12)
    ax2.set_ylabel('Normalized Latency', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Add ideal scaling line (linear scaling)
    ax2.plot(batch_sizes, batch_sizes, 'k--', alpha=0.5, linewidth=2, label='Ideal Linear Scaling')
    
    transformer_yvals = []
    for i, model in enumerate(transformer_models):
        # INT8 Eager
        int8_eager = df[(df['Model'] == model) & (df['Quantization'] == 'TRUE_INT8') & (df['Compilation'] == 'Eager')]
        if len(int8_eager) > 0:
            ax2.plot(int8_eager['Batch_Size'], int8_eager['Normalized_Latency'], 
                    marker='o', linewidth=2, label=f'{model} Eager', 
                    color=colors[i], linestyle='-', markersize=8)
            transformer_yvals.extend(int8_eager['Normalized_Latency'].values)
        
        # INT8 Compiled
        int8_compiled = df[(df['Model'] == model) & (df['Quantization'] == 'TRUE_INT8') & (df['Compilation'] == 'torch.compile')]
        if len(int8_compiled) > 0:
            ax2.plot(int8_compiled['Batch_Size'], int8_compiled['Normalized_Latency'], 
                    marker='s', linewidth=2, label=f'{model} Compiled', 
                    color=colors[i], linestyle='--', markersize=8)
            transformer_yvals.extend(int8_compiled['Normalized_Latency'].values)
    
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax2.set_xlim(0.5, max(batch_sizes) + 0.5)
    if transformer_yvals:
        ax2.set_ylim(min(transformer_yvals) * 0.95, max(transformer_yvals) * 1.05)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_summary_table(df):
    """Create a summary table showing scaling efficiency for INT8 models."""
    print("\n" + "="*80)
    print("INT8 BATCH SCALING EFFICIENCY SUMMARY")
    print("="*80)
    
    summary_data = []
    
    for model in df['Model'].unique():
        for compile_mode in df['Compilation'].unique():
            subset = df[(df['Model'] == model) & (df['Quantization'] == 'TRUE_INT8') & (df['Compilation'] == compile_mode)]
            
            if len(subset) < 2:  # Need at least 2 batch sizes
                continue
            
            # Calculate scaling efficiency (how close to linear scaling)
            bs1_latency = subset[subset['Batch_Size'] == 1]['Latency_ms'].iloc[0]
            bs8_latency = subset[subset['Batch_Size'] == 8]['Latency_ms'].iloc[0]
            bs16_latency = subset[subset['Batch_Size'] == 16]['Latency_ms'].iloc[0]
            
            # Ideal scaling would be 8x latency for 8x batch size
            ideal_bs8_latency = bs1_latency * 8
            ideal_bs16_latency = bs1_latency * 16
            scaling_efficiency_8 = (ideal_bs8_latency / bs8_latency) * 100
            scaling_efficiency_16 = (ideal_bs16_latency / bs16_latency) * 100
            
            summary_data.append({
                'Model': model,
                'Compilation': compile_mode,
                'BS1_Latency': bs1_latency,
                'BS8_Latency': bs8_latency,
                'BS16_Latency': bs16_latency,
                'Efficiency_BS8': scaling_efficiency_8,
                'Efficiency_BS16': scaling_efficiency_16
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Print formatted table
    print(f"{'Model':<15} {'Compile':<12} {'BS1(ms)':<8} {'BS8(ms)':<8} {'BS16(ms)':<9} {'Eff_BS8':<8} {'Eff_BS16':<9}")
    print("-" * 80)
    
    for _, row in summary_df.iterrows():
        print(f"{row['Model']:<15} {row['Compilation']:<12} "
              f"{row['BS1_Latency']:<8.3f} {row['BS8_Latency']:<8.3f} {row['BS16_Latency']:<9.3f} "
              f"{row['Efficiency_BS8']:<8.1f}% {row['Efficiency_BS16']:<9.1f}%")
    
    print("="*80)
    print("Note: Efficiency > 100% means better than linear scaling (sub-linear)")
    print("      Efficiency < 100% means worse than linear scaling (super-linear)")
    
    return summary_df

def main():
    # Load and process data
    csv_file = '/usr/scratch/difei/PerformanceDNA/results/quantization_benchmark_20250703_123303.csv'
    
    try:
        df = load_and_process_data(csv_file)
        print(f"Loaded data from {csv_file}")
        print(f"Found {len(df)} data points across {len(df['Model'].unique())} models")
        print(f"Quantization types: {df['Quantization'].unique()}")
        
        # Create plots
        fig = create_scalability_plots(df, 'int8_batch_scalability_analysis.png')
        
        # Create summary table
        summary_df = create_summary_table(df)
        
        # Save summary to CSV
        summary_df.to_csv('int8_batch_scaling_summary.csv', index=False)
        print(f"\nSummary saved to: int8_batch_scaling_summary.csv")
        
    except FileNotFoundError:
        print(f"Error: Could not find {csv_file}")
        print("Please make sure the CSV file is in the current directory.")
    except Exception as e:
        print(f"Error processing data: {e}")

if __name__ == "__main__":
    main()
