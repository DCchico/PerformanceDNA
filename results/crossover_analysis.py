import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def load_fp_data(csv_file):
    """Load FP32/FP16 benchmark data."""
    df = pd.read_csv(csv_file)
    
    # Clean up the data - handle inf and OOM values
    df['latency_ms'] = pd.to_numeric(df['latency_ms'], errors='coerce')
    df = df[df['latency_ms'].notna() & (df['latency_ms'] != np.inf)]
    
    return df

def load_int8_data(csv_file):
    """Load INT8 benchmark data."""
    df = pd.read_csv(csv_file)
    return df

def calculate_performance_ratios(df, precision_type="FP32"):
    """Calculate performance ratios between Eager and Compiled modes."""
    ratio_data = []
    
    for model in df['model'].unique() if 'model' in df.columns else df['Model'].unique():
        for batch_size in sorted(df['batch_size'].unique() if 'batch_size' in df.columns else df['Batch_Size'].unique()):
            
            if precision_type == "FP32":
                # FP32 data structure
                eager_data = df[(df['model'] == model) & 
                               (df['batch_size'] == batch_size) & 
                               (df['graph_compiler'] == 'Eager')]
                compiled_data = df[(df['model'] == model) & 
                                  (df['batch_size'] == batch_size) & 
                                  (df['graph_compiler'] == 'Dynamo')]
                
                if len(eager_data) > 0 and len(compiled_data) > 0:
                    eager_latency = eager_data['latency_ms'].iloc[0]
                    compiled_latency = compiled_data['latency_ms'].iloc[0]
                    
                    ratio = eager_latency / compiled_latency
                    ratio_data.append({
                        'Model': model,
                        'Batch_Size': batch_size,
                        'Eager_Latency': eager_latency,
                        'Compiled_Latency': compiled_latency,
                        'Performance_Ratio': ratio,
                        'Precision': precision_type
                    })
            else:
                # INT8 data structure
                eager_data = df[(df['Model'] == model) & 
                               (df['Batch_Size'] == batch_size) & 
                               (df['Quantization'] == 'TRUE_INT8') & 
                               (df['Compilation'] == 'Eager')]
                compiled_data = df[(df['Model'] == model) & 
                                  (df['Batch_Size'] == batch_size) & 
                                  (df['Quantization'] == 'TRUE_INT8') & 
                                  (df['Compilation'] == 'torch.compile')]
                
                if len(eager_data) > 0 and len(compiled_data) > 0:
                    eager_latency = eager_data['Latency_ms'].iloc[0]
                    compiled_latency = compiled_data['Latency_ms'].iloc[0]
                    
                    ratio = eager_latency / compiled_latency
                    ratio_data.append({
                        'Model': model,
                        'Batch_Size': batch_size,
                        'Eager_Latency': eager_latency,
                        'Compiled_Latency': compiled_latency,
                        'Performance_Ratio': ratio,
                        'Precision': precision_type
                    })
    
    return pd.DataFrame(ratio_data)

def create_crossover_plots(fp_df, int8_df, output_file='crossover_analysis.png'):
    """Create crossover analysis plots."""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Colors for models
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Plot 1: FP32 Performance Ratios
    ax1.set_title('FP32: Eager vs Compiled Performance Ratio', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Batch Size', fontsize=12)
    ax1.set_ylabel('Eager Latency / Compiled Latency', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Crossover Point')
    
    # Separate CNNs and transformers
    cnn_models = [model for model in fp_df['Model'].unique() if 'resnet' in model.lower()]
    transformer_models = [model for model in fp_df['Model'].unique() if model not in cnn_models]
    
    for i, model in enumerate(cnn_models + transformer_models):
        model_data = fp_df[fp_df['Model'] == model]
        if len(model_data) > 0:
            color = colors[i % len(colors)]
            linestyle = '-' if model in cnn_models else '--'
            marker = 'o' if model in cnn_models else 's'
            ax1.plot(model_data['Batch_Size'], model_data['Performance_Ratio'], 
                    marker=marker, linewidth=2, label=f'{model}', 
                    color=color, linestyle=linestyle, markersize=8)
    
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax1.set_ylim(0.5, 2.0)
    
    # Plot 2: INT8 Performance Ratios
    ax2.set_title('INT8: Eager vs Compiled Performance Ratio', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Batch Size', fontsize=12)
    ax2.set_ylabel('Eager Latency / Compiled Latency', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Crossover Point')
    
    for i, model in enumerate(int8_df['Model'].unique()):
        model_data = int8_df[int8_df['Model'] == model]
        if len(model_data) > 0:
            color = colors[i % len(colors)]
            linestyle = '-' if 'resnet' in model.lower() else '--'
            marker = 'o' if 'resnet' in model.lower() else 's'
            ax2.plot(model_data['Batch_Size'], model_data['Performance_Ratio'], 
                    marker=marker, linewidth=2, label=f'{model}', 
                    color=color, linestyle=linestyle, markersize=8)
    
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax2.set_ylim(0.5, 2.0)
    
    # Plot 3: Architecture Comparison - CNNs
    ax3.set_title('CNNs: Performance Ratio by Precision', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Batch Size', fontsize=12)
    ax3.set_ylabel('Eager Latency / Compiled Latency', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Crossover Point')
    
    # Plot ResNet50 for both precisions
    resnet50_fp = fp_df[fp_df['Model'] == 'resnet50']
    resnet50_int8 = int8_df[int8_df['Model'] == 'resnet50']
    
    if len(resnet50_fp) > 0:
        ax3.plot(resnet50_fp['Batch_Size'], resnet50_fp['Performance_Ratio'], 
                marker='o', linewidth=2, label='ResNet50 FP32', color='#1f77b4')
    if len(resnet50_int8) > 0:
        ax3.plot(resnet50_int8['Batch_Size'], resnet50_int8['Performance_Ratio'], 
                marker='s', linewidth=2, label='ResNet50 INT8', color='#ff7f0e')
    
    ax3.legend(fontsize=10)
    ax3.set_ylim(0.5, 2.0)
    
    # Plot 4: Architecture Comparison - Transformers
    ax4.set_title('Transformers: Performance Ratio by Precision', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Batch Size', fontsize=12)
    ax4.set_ylabel('Eager Latency / Compiled Latency', fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Crossover Point')
    
    # Plot ViT for both precisions
    vit_fp = fp_df[fp_df['Model'] == 'vit']
    vit_int8 = int8_df[int8_df['Model'] == 'vit']
    
    if len(vit_fp) > 0:
        ax4.plot(vit_fp['Batch_Size'], vit_fp['Performance_Ratio'], 
                marker='o', linewidth=2, label='ViT FP32', color='#2ca02c')
    if len(vit_int8) > 0:
        ax4.plot(vit_int8['Batch_Size'], vit_int8['Performance_Ratio'], 
                marker='s', linewidth=2, label='ViT INT8', color='#d62728')
    
    ax4.legend(fontsize=10)
    ax4.set_ylim(0.5, 2.0)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_summary_table(fp_df, int8_df):
    """Create a summary table showing crossover analysis."""
    print("\n" + "="*100)
    print("CROSSOVER ANALYSIS SUMMARY")
    print("="*100)
    
    print("\nFP32/FP16 Results:")
    print("-" * 50)
    for model in fp_df['Model'].unique():
        model_data = fp_df[fp_df['Model'] == model]
        if len(model_data) > 0:
            min_ratio = model_data['Performance_Ratio'].min()
            max_ratio = model_data['Performance_Ratio'].max()
            avg_ratio = model_data['Performance_Ratio'].mean()
            
            # Check if crossover occurs
            crossover_occurs = (min_ratio < 1) and (max_ratio > 1)
            crossover_info = "YES" if crossover_occurs else "NO"
            
            print(f"{model:<15} | Min: {min_ratio:<6.3f} | Max: {max_ratio:<6.3f} | Avg: {avg_ratio:<6.3f} | Crossover: {crossover_info}")
    
    print("\nINT8 Results:")
    print("-" * 50)
    for model in int8_df['Model'].unique():
        model_data = int8_df[int8_df['Model'] == model]
        if len(model_data) > 0:
            min_ratio = model_data['Performance_Ratio'].min()
            max_ratio = model_data['Performance_Ratio'].max()
            avg_ratio = model_data['Performance_Ratio'].mean()
            
            # Check if crossover occurs
            crossover_occurs = (min_ratio < 1) and (max_ratio > 1)
            crossover_info = "YES" if crossover_occurs else "NO"
            
            print(f"{model:<15} | Min: {min_ratio:<6.3f} | Max: {max_ratio:<6.3f} | Avg: {avg_ratio:<6.3f} | Crossover: {crossover_info}")
    
    print("\n" + "="*100)
    print("Key Insights:")
    print("• Ratio > 1: Eager is slower (Compiled wins)")
    print("• Ratio < 1: Eager is faster (Crossover occurred)")
    print("• Crossover typically occurs in transformers at larger batch sizes")
    print("• CNNs generally maintain compiled advantage across all batch sizes")

def main():
    # Load data
    fp_csv = '/usr/scratch/difei/PerformanceDNA/results/benchmark_results_torch27.csv'
    int8_csv = '/usr/scratch/difei/PerformanceDNA/results/quantization_benchmark_20250703_123303.csv'
    
    try:
        # Load and process data
        fp_raw = load_fp_data(fp_csv)
        int8_raw = load_int8_data(int8_csv)
        
        print(f"Loaded FP32/FP16 data: {len(fp_raw)} data points")
        print(f"Loaded INT8 data: {len(int8_raw)} data points")
        
        # Calculate performance ratios
        fp_ratios = calculate_performance_ratios(fp_raw, "FP32")
        int8_ratios = calculate_performance_ratios(int8_raw, "INT8")
        
        print(f"Calculated {len(fp_ratios)} FP32 performance ratios")
        print(f"Calculated {len(int8_ratios)} INT8 performance ratios")
        
        # Create plots
        fig = create_crossover_plots(fp_ratios, int8_ratios, 'crossover_analysis.png')
        
        # Create summary table
        create_summary_table(fp_ratios, int8_ratios)
        
        # Save ratio data
        fp_ratios.to_csv('fp32_performance_ratios.csv', index=False)
        int8_ratios.to_csv('int8_performance_ratios.csv', index=False)
        print(f"\nPerformance ratio data saved to CSV files")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find data file - {e}")
    except Exception as e:
        print(f"Error processing data: {e}")

if __name__ == "__main__":
    main() 