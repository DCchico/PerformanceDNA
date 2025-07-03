#!/usr/bin/env python3
"""
Transformer Quantization Analysis Script
Analyzes benchmark results to test hypothesis about eager vs compiled performance
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class TransformerQuantizationAnalyzer:
    def __init__(self, fp32_file, int8_file):
        self.fp32_data = pd.read_csv(fp32_file)
        self.int8_data = pd.read_csv(int8_file)
        self.combined_data = pd.concat([self.fp32_data, self.int8_data], ignore_index=True)
        
        # Create results directory
        Path("results/analysis").mkdir(parents=True, exist_ok=True)
        
    def prepare_data(self):
        """Prepare and clean the data for analysis."""
        # Filter for transformer models only
        transformer_models = ['bert-base-uncased', 'bert-large-uncased', 
                            'roberta-base', 'roberta-large', 'distilbert-base-uncased',
                            'gpt2', 'gpt2-medium', 'vit']
        
        self.fp32_transformer = self.fp32_data[
            self.fp32_data['Model'].isin(transformer_models)
        ].copy()
        
        self.int8_transformer = self.int8_data[
            self.int8_data['Model'].isin(transformer_models)
        ].copy()
        
        # Add model type classification
        def classify_model(model_name):
            if 'bert' in model_name.lower() or 'roberta' in model_name.lower() or 'distilbert' in model_name.lower():
                return 'BERT-family'
            elif 'gpt' in model_name.lower():
                return 'GPT-family'
            elif 'vit' in model_name.lower():
                return 'Vision-Transformer'
            else:
                return 'Other'
        
        self.fp32_transformer['Model_Type'] = self.fp32_transformer['Model'].apply(classify_model)
        self.int8_transformer['Model_Type'] = self.int8_transformer['Model'].apply(classify_model)
        
        # Calculate performance ratios (Eager/Compiled)
        self.calculate_performance_ratios()
        
    def calculate_performance_ratios(self):
        """Calculate performance ratios between Eager and Compiled modes."""
        # For FP32 data
        fp32_pivot = self.fp32_transformer.pivot_table(
            index=['Model', 'Batch_Size'], 
            columns='Compilation', 
            values='Latency_ms', 
            aggfunc='mean'
        ).reset_index()
        
        fp32_pivot['Performance_Ratio'] = fp32_pivot['Eager'] / fp32_pivot['torch.compile']
        fp32_pivot['Compilation_Benefit'] = (fp32_pivot['Eager'] - fp32_pivot['torch.compile']) / fp32_pivot['Eager'] * 100
        
        # For INT8 data
        int8_pivot = self.int8_transformer.pivot_table(
            index=['Model', 'Batch_Size'], 
            columns='Compilation', 
            values='Latency_ms', 
            aggfunc='mean'
        ).reset_index()
        
        int8_pivot['Performance_Ratio'] = int8_pivot['Eager'] / int8_pivot['torch.compile']
        int8_pivot['Compilation_Benefit'] = (int8_pivot['Eager'] - int8_pivot['torch.compile']) / int8_pivot['Eager'] * 100
        
        self.fp32_ratios = fp32_pivot
        self.int8_ratios = int8_pivot
        
    def plot_performance_comparison(self):
        """Plot 1: Overall performance comparison between FP32 and INT8."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Transformer Performance: FP32 vs INT8 Quantization', fontsize=16, fontweight='bold')
        
        # FP32 Performance
        fp32_models = self.fp32_transformer['Model'].unique()
        for model in fp32_models:
            model_data = self.fp32_transformer[self.fp32_transformer['Model'] == model]
            eager_data = model_data[model_data['Compilation'] == 'Eager']
            compiled_data = model_data[model_data['Compilation'] == 'torch.compile']
            
            axes[0, 0].plot(eager_data['Batch_Size'], eager_data['Latency_ms'], 
                           marker='o', label=f'{model} (Eager)', alpha=0.8)
            axes[0, 1].plot(compiled_data['Batch_Size'], compiled_data['Latency_ms'], 
                           marker='s', label=f'{model} (Compiled)', alpha=0.8)
        
        # INT8 Performance
        int8_models = self.int8_transformer['Model'].unique()
        for model in int8_models:
            model_data = self.int8_transformer[self.int8_transformer['Model'] == model]
            eager_data = model_data[model_data['Compilation'] == 'Eager']
            compiled_data = model_data[model_data['Compilation'] == 'torch.compile']
            
            axes[1, 0].plot(eager_data['Batch_Size'], eager_data['Latency_ms'], 
                           marker='o', label=f'{model} (Eager)', alpha=0.8)
            axes[1, 1].plot(compiled_data['Batch_Size'], compiled_data['Latency_ms'], 
                           marker='s', label=f'{model} (Compiled)', alpha=0.8)
        
        # Formatting
        for ax in axes.flat:
            ax.set_xlabel('Batch Size')
            ax.set_ylabel('Latency (ms)')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        axes[0, 0].set_title('FP32 - Eager Mode')
        axes[0, 1].set_title('FP32 - Compiled Mode')
        axes[1, 0].set_title('INT8 - Eager Mode')
        axes[1, 1].set_title('INT8 - Compiled Mode')
        
        plt.tight_layout()
        plt.savefig('results/analysis/performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_performance_ratios(self):
        """Plot 2: Performance ratios showing when eager is faster than compiled."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Performance Ratios: Eager/Compiled (Values > 1 indicate Eager is SLOWER)', 
                    fontsize=16, fontweight='bold')
        
        # FP32 Performance Ratios
        for model in self.fp32_ratios['Model'].unique():
            model_data = self.fp32_ratios[self.fp32_ratios['Model'] == model]
            axes[0, 0].plot(model_data['Batch_Size'], model_data['Performance_Ratio'], 
                           marker='o', label=model, alpha=0.8)
        
        # INT8 Performance Ratios
        for model in self.int8_ratios['Model'].unique():
            model_data = self.int8_ratios[self.int8_ratios['Model'] == model]
            axes[0, 1].plot(model_data['Batch_Size'], model_data['Performance_Ratio'], 
                           marker='o', label=model, alpha=0.8)
        
        # Compilation Benefit (% improvement)
        for model in self.fp32_ratios['Model'].unique():
            model_data = self.fp32_ratios[self.fp32_ratios['Model'] == model]
            axes[1, 0].plot(model_data['Batch_Size'], model_data['Compilation_Benefit'], 
                           marker='o', label=model, alpha=0.8)
        
        for model in self.int8_ratios['Model'].unique():
            model_data = self.int8_ratios[self.int8_ratios['Model'] == model]
            axes[1, 1].plot(model_data['Batch_Size'], model_data['Compilation_Benefit'], 
                           marker='o', label=model, alpha=0.8)
        
        # Add reference lines
        for ax in [axes[0, 0], axes[0, 1]]:
            ax.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Eager = Compiled (Equal)')
            ax.set_ylabel('Performance Ratio (Eager/Compiled)')
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        for ax in [axes[1, 0], axes[1, 1]]:
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='No Benefit')
            ax.set_ylabel('Compilation Benefit (%) - Positive = Compiled Faster')
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        for ax in axes.flat:
            ax.set_xlabel('Batch Size')
        
        axes[0, 0].set_title('FP32 Performance Ratios')
        axes[0, 1].set_title('INT8 Performance Ratios')
        axes[1, 0].set_title('FP32 Compilation Benefit')
        axes[1, 1].set_title('INT8 Compilation Benefit')
        
        plt.tight_layout()
        plt.savefig('results/analysis/performance_ratios.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_model_type_analysis(self):
        """Plot 3: Analysis by model type (BERT-family vs GPT-family)."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Performance Analysis by Model Architecture', fontsize=16, fontweight='bold')
        
        # Group by model type
        model_types = ['BERT-family', 'GPT-family', 'Vision-Transformer']
        colors = ['blue', 'green', 'orange']
        
        for i, model_type in enumerate(model_types):
            # FP32 data
            fp32_type_data = self.fp32_transformer[self.fp32_transformer['Model_Type'] == model_type]
            if not fp32_type_data.empty:
                fp32_pivot = fp32_type_data.pivot_table(
                    index='Batch_Size', columns='Compilation', values='Latency_ms', aggfunc='mean'
                )
                fp32_ratio = fp32_pivot['Eager'] / fp32_pivot['torch.compile']
                axes[0, 0].plot(fp32_ratio.index, fp32_ratio.values, 
                               marker='o', label=model_type, color=colors[i], alpha=0.8)
            
            # INT8 data
            int8_type_data = self.int8_transformer[self.int8_transformer['Model_Type'] == model_type]
            if not int8_type_data.empty:
                int8_pivot = int8_type_data.pivot_table(
                    index='Batch_Size', columns='Compilation', values='Latency_ms', aggfunc='mean'
                )
                int8_ratio = int8_pivot['Eager'] / int8_pivot['torch.compile']
                axes[0, 1].plot(int8_ratio.index, int8_ratio.values, 
                               marker='o', label=model_type, color=colors[i], alpha=0.8)
        
        # Batch scaling analysis
        for model_type in model_types:
            fp32_type_data = self.fp32_transformer[self.fp32_transformer['Model_Type'] == model_type]
            if not fp32_type_data.empty:
                eager_data = fp32_type_data[fp32_type_data['Compilation'] == 'Eager'].groupby('Batch_Size')['Latency_ms'].mean()
                compiled_data = fp32_type_data[fp32_type_data['Compilation'] == 'torch.compile'].groupby('Batch_Size')['Latency_ms'].mean()
                
                axes[1, 0].plot(eager_data.index, eager_data.values, 
                               marker='o', label=f'{model_type} (Eager)', alpha=0.8)
                axes[1, 0].plot(compiled_data.index, compiled_data.values, 
                               marker='s', label=f'{model_type} (Compiled)', alpha=0.8)
        
        # Formatting
        for ax in [axes[0, 0], axes[0, 1]]:
            ax.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Eager = Compiled (Equal)')
            ax.set_ylabel('Performance Ratio (Eager/Compiled)')
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        for ax in [axes[1, 0], axes[1, 1]]:
            ax.set_xlabel('Batch Size')
            ax.set_ylabel('Latency (ms)')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        axes[0, 0].set_title('FP32 Performance Ratios by Architecture')
        axes[0, 1].set_title('INT8 Performance Ratios by Architecture')
        axes[1, 0].set_title('FP32 Batch Scaling by Architecture')
        axes[1, 1].set_title('INT8 Batch Scaling by Architecture')
        
        plt.tight_layout()
        plt.savefig('results/analysis/model_type_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_crossover_analysis(self):
        """Plot 4: Detailed crossover analysis for specific models."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Crossover Analysis: When Eager Becomes Faster Than Compiled', 
                    fontsize=16, fontweight='bold')
        
        # Focus on models with interesting patterns
        interesting_models = ['gpt2', 'gpt2-medium', 'bert-base-uncased', 'roberta-base']
        
        for i, model in enumerate(interesting_models):
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            # FP32 data
            fp32_model_data = self.fp32_transformer[self.fp32_transformer['Model'] == model]
            if not fp32_model_data.empty:
                fp32_pivot = fp32_model_data.pivot_table(
                    index='Batch_Size', columns='Compilation', values='Latency_ms', aggfunc='mean'
                )
                fp32_ratio = fp32_pivot['Eager'] / fp32_pivot['torch.compile']
                ax.plot(fp32_ratio.index, fp32_ratio.values, 
                       marker='o', label='FP32', linewidth=2, markersize=8)
            
            # INT8 data
            int8_model_data = self.int8_transformer[self.int8_transformer['Model'] == model]
            if not int8_model_data.empty:
                int8_pivot = int8_model_data.pivot_table(
                    index='Batch_Size', columns='Compilation', values='Latency_ms', aggfunc='mean'
                )
                int8_ratio = int8_pivot['Eager'] / int8_pivot['torch.compile']
                ax.plot(int8_ratio.index, int8_ratio.values, 
                       marker='s', label='INT8', linewidth=2, markersize=8)
            
            ax.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Eager = Compiled (Equal)')
            ax.set_xlabel('Batch Size')
            ax.set_ylabel('Performance Ratio (Eager/Compiled)')
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_title(f'{model}')
            
            # Add annotations for crossover points
            if not fp32_model_data.empty:
                fp32_crossovers = fp32_ratio[fp32_ratio < 1]
                if not fp32_crossovers.empty:
                    ax.annotate(f'FP32: {len(fp32_crossovers)} crossover(s)', 
                               xy=(0.05, 0.95), xycoords='axes fraction',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
            
            if not int8_model_data.empty:
                int8_crossovers = int8_ratio[int8_ratio < 1]
                if not int8_crossovers.empty:
                    ax.annotate(f'INT8: {len(int8_crossovers)} crossover(s)', 
                               xy=(0.05, 0.85), xycoords='axes fraction',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        
        plt.tight_layout()
        plt.savefig('results/analysis/crossover_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def generate_summary_statistics(self):
        """Generate summary statistics and insights."""
        print("=" * 80)
        print("TRANSFORMER QUANTIZATION ANALYSIS SUMMARY")
        print("=" * 80)
        
        # Overall statistics
        print(f"\n📊 DATASET OVERVIEW:")
        print(f"   FP32 Models: {len(self.fp32_transformer['Model'].unique())}")
        print(f"   INT8 Models: {len(self.int8_transformer['Model'].unique())}")
        print(f"   Total FP32 Data Points: {len(self.fp32_transformer)}")
        print(f"   Total INT8 Data Points: {len(self.int8_transformer)}")
        
        # Crossover analysis
        print(f"\n🔄 CROSSOVER ANALYSIS (Eager < Compiled - Eager is faster):")
        
        # FP32 crossovers
        fp32_crossovers = self.fp32_ratios[self.fp32_ratios['Performance_Ratio'] < 1]
        if not fp32_crossovers.empty:
            print(f"   FP32 Crossovers: {len(fp32_crossovers)} instances")
            for _, row in fp32_crossovers.iterrows():
                print(f"     - {row['Model']} (batch {row['Batch_Size']}): {row['Performance_Ratio']:.3f}")
        else:
            print(f"   FP32 Crossovers: None found")
        
        # INT8 crossovers
        int8_crossovers = self.int8_ratios[self.int8_ratios['Performance_Ratio'] < 1]
        if not int8_crossovers.empty:
            print(f"   INT8 Crossovers: {len(int8_crossovers)} instances")
            for _, row in int8_crossovers.iterrows():
                print(f"     - {row['Model']} (batch {row['Batch_Size']}): {row['Performance_Ratio']:.3f}")
        else:
            print(f"   INT8 Crossovers: None found")
        
        # Average compilation benefits
        print(f"\n📈 AVERAGE COMPILATION BENEFITS:")
        fp32_avg_benefit = self.fp32_ratios['Compilation_Benefit'].mean()
        int8_avg_benefit = self.int8_ratios['Compilation_Benefit'].mean()
        print(f"   FP32: {fp32_avg_benefit:.1f}% benefit (positive = compiled faster)")
        print(f"   INT8: {int8_avg_benefit:.1f}% benefit (positive = compiled faster)")
        
        # Model-specific insights
        print(f"\n🎯 MODEL-SPECIFIC INSIGHTS:")
        for model in self.fp32_ratios['Model'].unique():
            model_fp32 = self.fp32_ratios[self.fp32_ratios['Model'] == model]
            model_int8 = self.int8_ratios[self.int8_ratios['Model'] == model]
            
            fp32_benefit = model_fp32['Compilation_Benefit'].mean()
            int8_benefit = model_int8['Compilation_Benefit'].mean() if not model_int8.empty else None
            
            print(f"   {model}:")
            print(f"     FP32: {fp32_benefit:.1f}% benefit (positive = compiled faster)")
            if int8_benefit is not None:
                print(f"     INT8: {int8_benefit:.1f}% benefit (positive = compiled faster)")
        
        # Hypothesis testing
        print(f"\n🔬 HYPOTHESIS TESTING:")
        print(f"   Hypothesis: 'Transformers, especially quantized ones, show counterintuitive results")
        print(f"   where eager mode can be faster than compiled mode at higher batch sizes'")
        
        total_crossovers = len(fp32_crossovers) + len(int8_crossovers)
        if total_crossovers > 0:
            print(f"   ✅ SUPPORTED: Found {total_crossovers} instances where eager < compiled (eager is faster)")
            print(f"   📊 Evidence: {len(int8_crossovers)} INT8 crossovers vs {len(fp32_crossovers)} FP32 crossovers")
        else:
            print(f"   ❌ NOT SUPPORTED: No crossovers found in current dataset")
            print(f"   💡 Note: Limited INT8 data (only GPT models) may affect results")
        
        print("=" * 80)

def main():
    # File paths
    fp32_file = "../results/cache_managed_transformer_benchmark_20250703_144113.csv"
    int8_file = "../results/expanded_int8_transformer_benchmark_20250703_145219.csv"
    
    # Initialize analyzer
    analyzer = TransformerQuantizationAnalyzer(fp32_file, int8_file)
    analyzer.prepare_data()
    
    # Generate plots
    print("Generating analysis plots...")
    analyzer.plot_performance_comparison()
    analyzer.plot_performance_ratios()
    analyzer.plot_model_type_analysis()
    analyzer.plot_crossover_analysis()
    
    # Generate summary
    analyzer.generate_summary_statistics()
    
    print(f"\n✅ Analysis complete! Check 'results/analysis/' for generated plots.")

if __name__ == "__main__":
    main() 

'''
Key Analysis Questions:
Do we see crossovers? (Eager > Compiled at higher batch sizes)
Are INT8 models more likely to show crossovers?
Do different transformer architectures behave differently?
What's the average compilation benefit for each precision?
Hypothesis Testing:
The script will specifically test our hypothesis: "Transformers, especially quantized ones, show counterintuitive results where eager mode can be faster than compiled mode at higher batch sizes before saturation."
'''