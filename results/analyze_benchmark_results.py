import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_clean_data(csv_path):
    """Load and clean the benchmark data"""
    df = pd.read_csv(csv_path)
    
    # Convert 'inf' and 'OOM' to NaN for analysis
    df['latency_ms'] = pd.to_numeric(df['latency_ms'], errors='coerce')
    
    # Create a combined compiler column for easier analysis
    # Fix: Handle Eager mode properly by checking for 'N/A' backend
    def create_compiler_backend(row):
        if row['graph_compiler'] == 'Eager':
            return 'Eager (N/A)'
        else:
            return row['graph_compiler'] + ' (' + row['backend'] + ')'
    
    df['compiler_backend'] = df.apply(create_compiler_backend, axis=1)
    
    return df

def analyze_model_performance_patterns(df):
    """Analyze performance patterns across different models and compilers"""
    
    print("=== PERFORMANCEDNA ANALYSIS ===\n")
    
    # 1. Model-specific compiler effectiveness
    print("1. MODEL-SPECIFIC COMPILER EFFECTIVENESS:")
    print("-" * 50)
    
    for model in df['model'].unique():
        model_data = df[df['model'] == model].copy()
        if model_data['latency_ms'].isna().all():
            continue
            
        print(f"\n{model.upper()}:")
        
        # Find best performing compiler for this model
        best_compiler = model_data.loc[model_data['latency_ms'].idxmin(), 'compiler_backend']
        best_latency = model_data['latency_ms'].min()
        
        # Compare with Eager mode
        eager_latency = model_data[model_data['graph_compiler'] == 'Eager']['latency_ms'].iloc[0]
        improvement = ((eager_latency - best_latency) / eager_latency) * 100
        
        print(f"  Best compiler: {best_compiler}")
        print(f"  Best latency: {best_latency:.3f} ms")
        print(f"  vs Eager: {improvement:+.1f}% improvement")
        
        # Check if any compilers failed - fix the bug by filtering properly
        failed_data = model_data[model_data['latency_ms'].isna()]
        if not failed_data.empty:
            failed_compilers = failed_data['compiler_backend'].dropna().tolist()
            if failed_compilers:
                print(f"  Failed compilers: {', '.join(failed_compilers)}")

def create_performance_heatmap(df):
    """Create a heatmap showing performance across models and compilers"""
    
    # Filter out failed runs (NaN values)
    df_clean = df.dropna(subset=['latency_ms'])
    
    # Create pivot table for heatmap
    pivot_data = df_clean.pivot_table(
        values='latency_ms', 
        index='model', 
        columns='compiler_backend',
        aggfunc='mean'
    )
    
    # Adjust figure size based on number of models
    num_models = len(pivot_data.index)
    fig_height = max(8, num_models * 1.5)  # Scale height with number of models
    
    # Create the heatmap
    plt.figure(figsize=(12, fig_height))
    sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='RdYlGn_r', 
                cbar_kws={'label': 'Latency (ms)'})
    plt.title('PerformanceDNA: Model vs Compiler Performance Heatmap\n(Lower is Better)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Compiler Backend', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('results/performance_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_batch_size_analysis(df):
    """Analyze how performance scales with batch size"""
    
    # Filter out failed runs
    df_clean = df.dropna(subset=['latency_ms'])
    
    # Create subplots for each model - handle all models
    models = df_clean['model'].unique()
    num_models = len(models)
    
    # Calculate subplot layout
    if num_models <= 4:
        rows, cols = 2, 2
    elif num_models <= 6:
        rows, cols = 2, 3
    else:
        rows, cols = 3, 3
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
    if num_models == 1:
        axes = [axes]
    else:
        axes = axes.ravel()
    
    # Define consistent colors for each compiler type
    colors = {
        'Eager (N/A)': 'black',
        'TorchScript (Trace)': 'blue',
        'TorchScript (Script)': 'cyan',
        'Dynamo (Inductor)': 'red'
    }
    
    # Keep track of legend handles and labels for common legend
    legend_handles = {}
    legend_labels = []
    
    for idx, model in enumerate(models):
        if idx >= len(axes):  # Safety check
            break
            
        model_data = df_clean[df_clean['model'] == model]
        
        # Get all unique batch sizes for this model and create categorical mapping
        all_batch_sizes = sorted(model_data['batch_size'].unique())
        batch_size_to_pos = {bs: i for i, bs in enumerate(all_batch_sizes)}
        
        # Get all available compilers for this model
        available_compilers = model_data['compiler_backend'].unique()
        
        # Ensure we plot Eager mode first as baseline, then other compilers
        compilers_to_plot = []
        if 'Eager (N/A)' in available_compilers:
            compilers_to_plot.append('Eager (N/A)')
        compilers_to_plot.extend([c for c in available_compilers if c != 'Eager (N/A)'])
        
        for compiler in compilers_to_plot:
            compiler_data = model_data[model_data['compiler_backend'] == compiler]
            if not compiler_data.empty:
                # Sort by batch_size to fix line connection issue
                compiler_data = compiler_data.sort_values('batch_size')
                
                # Convert batch sizes to categorical positions
                x_positions = [batch_size_to_pos[bs] for bs in compiler_data['batch_size']]
                
                color = colors.get(compiler, 'gray')
                linewidth = 3 if compiler == 'Eager (N/A)' else 2  # Make Eager mode more prominent
                linestyle = '-' if compiler == 'Eager (N/A)' else '--'  # Different style for Eager
                
                # Plot the line using categorical positions
                line, = axes[idx].plot(x_positions, compiler_data['latency_ms'], 
                                      marker='o', linewidth=linewidth, 
                                      color=color, linestyle=linestyle, markersize=6)
                
                # Store legend handle for each compiler (only once)
                if compiler not in legend_handles:
                    legend_handles[compiler] = line
                    legend_labels.append(compiler)
        
        axes[idx].set_title(f'{model.upper()}', fontweight='bold', fontsize=16)
        axes[idx].set_xlabel('Batch Size', fontsize=14)
        axes[idx].set_ylabel('Latency (ms)', fontsize=14)
        axes[idx].grid(True, alpha=0.3)
        
        # Use linear scale for both axes
        axes[idx].set_xscale('linear')
        axes[idx].set_yscale('linear')
        
        # Set x-ticks to categorical positions with batch size labels
        axes[idx].set_xticks(range(len(all_batch_sizes)))
        axes[idx].set_xticklabels([str(int(bs)) for bs in all_batch_sizes], fontsize=12)
        
        # Increase tick label sizes
        axes[idx].tick_params(axis='both', which='major', labelsize=12)
    
    # Hide unused subplots
    for idx in range(num_models, len(axes)):
        axes[idx].set_visible(False)
    
    # Add common legend outside the subplots with larger font
    legend_handles_list = [legend_handles[label] for label in legend_labels]
    fig.legend(legend_handles_list, legend_labels, loc='upper center', bbox_to_anchor=(0.5, 0.02), 
               ncol=len(legend_labels), frameon=True, fancybox=True, shadow=True, fontsize=14)
    
    plt.suptitle('PerformanceDNA: Batch Size Scaling Analysis', fontsize=20, fontweight='bold')
    plt.tight_layout()
    # Adjust layout to make room for the legend
    plt.subplots_adjust(bottom=0.15)
    plt.savefig('results/batch_size_scaling.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_compiler_failures(df):
    """Analyze which compilers fail for which models"""
    
    print("\n2. COMPILER FAILURE ANALYSIS:")
    print("-" * 50)
    
    # Count failures by model and compiler
    failure_matrix = df[df['latency_ms'].isna()].groupby(['model', 'compiler_backend']).size().unstack(fill_value=0)
    
    print("\nFailure Matrix (number of failed runs):")
    print(failure_matrix)
    
    # Identify patterns
    print("\nKey Observations:")
    
    # TorchScript failures
    torchscript_failures = df[(df['graph_compiler'] == 'TorchScript') & (df['latency_ms'].isna())]
    if not torchscript_failures.empty:
        failed_models = torchscript_failures['model'].unique()
        print(f"  - TorchScript fails for: {', '.join(failed_models)}")
        print("  - This suggests these models have dynamic control flow or unsupported operations")
    
    # OOM patterns
    oom_data = df[df['latency_ms'].isna()]
    if not oom_data.empty:
        oom_models = oom_data['model'].unique()
        print(f"  - OOM occurs for: {', '.join(oom_models)} at larger batch sizes")
        print("  - Memory-intensive models benefit from compiler optimizations")

def create_performance_dna_summary(df):
    """Create a summary of PerformanceDNA insights"""
    
    print("\n3. PERFORMANCEDNA INSIGHTS:")
    print("-" * 50)
    
    # Calculate average improvement for each model
    insights = []
    
    for model in df['model'].unique():
        model_data = df[df['model'] == model].copy()
        if model_data['latency_ms'].isna().all():
            continue
            
        eager_latency = model_data[model_data['graph_compiler'] == 'Eager']['latency_ms'].iloc[0]
        
        # Find best non-eager compiler
        non_eager_data = model_data[model_data['graph_compiler'] != 'Eager'].dropna()
        if not non_eager_data.empty:
            best_compiler = non_eager_data.loc[non_eager_data['latency_ms'].idxmin(), 'compiler_backend']
            best_latency = non_eager_data['latency_ms'].min()
            improvement = ((eager_latency - best_latency) / eager_latency) * 100
            
            insights.append({
                'model': model,
                'best_compiler': best_compiler,
                'improvement': improvement,
                'eager_latency': eager_latency,
                'best_latency': best_latency
            })
    
    # Sort by improvement
    insights.sort(key=lambda x: x['improvement'], reverse=True)
    
    print("\nModel PerformanceDNA Ranking (by compiler improvement):")
    for i, insight in enumerate(insights, 1):
        print(f"{i}. {insight['model']}: {insight['improvement']:+.1f}% improvement with {insight['best_compiler']}")
    
    # Create visualization
    models = [insight['model'] for insight in insights]
    improvements = [insight['improvement'] for insight in insights]
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, improvements, color=colors, alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title('PerformanceDNA: Compiler Effectiveness by Model', fontweight='bold')
    plt.xlabel('Model')
    plt.ylabel('Improvement vs Eager Mode (%)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, improvement in zip(bars, improvements):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + (0.5 if height > 0 else -1),
                f'{improvement:+.1f}%', ha='center', va='bottom' if height > 0 else 'top')
    
    plt.tight_layout()
    plt.savefig('results/compiler_effectiveness.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main analysis function"""
    
    # Load data
    csv_path = 'results/benchmark_results.csv'
    if not Path(csv_path).exists():
        print(f"Error: {csv_path} not found!")
        return
    
    df = load_and_clean_data(csv_path)
    
    print(f"Loaded {len(df)} benchmark results")
    print(f"Models: {', '.join(df['model'].unique())}")
    print(f"Compilers: {', '.join(df['graph_compiler'].unique())}")
    print(f"Batch sizes: {sorted(df['batch_size'].unique())}")
    
    # Run analyses
    analyze_model_performance_patterns(df)
    analyze_compiler_failures(df)
    create_performance_dna_summary(df)
    
    # Create visualizations
    create_performance_heatmap(df)
    create_batch_size_analysis(df)
    
    print("\n=== ANALYSIS COMPLETE ===")
    print("Generated visualizations:")
    print("- performance_heatmap.png")
    print("- batch_size_scaling.png") 
    print("- compiler_effectiveness.png")

if __name__ == "__main__":
    main() 