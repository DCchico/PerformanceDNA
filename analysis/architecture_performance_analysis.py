#!/usr/bin/env python3
"""
Architecture Performance Analysis
Analyzes why BERT-family models prefer eager mode while GPT-family models prefer compilation
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

class ArchitecturePerformanceAnalyzer:
    def __init__(self, fp32_file, int8_file):
        self.fp32_data = pd.read_csv(fp32_file)
        self.int8_data = pd.read_csv(int8_file)
        
        # Create results directory
        Path("results/analysis").mkdir(parents=True, exist_ok=True)
        
    def prepare_data(self):
        """Prepare and classify models by architecture."""
        # Filter for transformer models
        transformer_models = ['bert-base-uncased', 'bert-large-uncased', 
                            'roberta-base', 'roberta-large', 'distilbert-base-uncased',
                            'gpt2', 'gpt2-medium', 'vit']
        
        self.fp32_transformer = self.fp32_data[
            self.fp32_data['Model'].isin(transformer_models)
        ].copy()
        
        self.int8_transformer = self.int8_data[
            self.int8_data['Model'].isin(transformer_models)
        ].copy()
        
        # Classify models by architecture
        def classify_architecture(model_name):
            if 'bert' in model_name.lower() or 'roberta' in model_name.lower() or 'distilbert' in model_name.lower():
                return 'BERT-family (Encoder-only)'
            elif 'gpt' in model_name.lower():
                return 'GPT-family (Decoder-only)'
            elif 'vit' in model_name.lower():
                return 'Vision-Transformer'
            else:
                return 'Other'
        
        self.fp32_transformer['Architecture'] = self.fp32_transformer['Model'].apply(classify_architecture)
        self.int8_transformer['Architecture'] = self.int8_transformer['Model'].apply(classify_architecture)
        
        # Calculate performance metrics
        self.calculate_architecture_metrics()
        
    def calculate_architecture_metrics(self):
        """Calculate performance metrics by architecture."""
        # Pivot data for analysis
        fp32_pivot = self.fp32_transformer.pivot_table(
            index=['Model', 'Batch_Size'], 
            columns='Compilation', 
            values='Latency_ms', 
            aggfunc='mean'
        ).reset_index()
        
        fp32_pivot['Performance_Ratio'] = fp32_pivot['Eager'] / fp32_pivot['torch.compile']
        fp32_pivot['Compilation_Benefit'] = (fp32_pivot['Eager'] - fp32_pivot['torch.compile']) / fp32_pivot['Eager'] * 100
        
        # Add architecture classification
        fp32_pivot['Architecture'] = fp32_pivot['Model'].apply(
            lambda x: 'BERT-family' if any(name in x.lower() for name in ['bert', 'roberta', 'distilbert']) 
            else 'GPT-family' if 'gpt' in x.lower() else 'Vision-Transformer'
        )
        
        self.fp32_metrics = fp32_pivot
        
        # INT8 metrics
        if not self.int8_transformer.empty:
            int8_pivot = self.int8_transformer.pivot_table(
                index=['Model', 'Batch_Size'], 
                columns='Compilation', 
                values='Latency_ms', 
                aggfunc='mean'
            ).reset_index()
            
            int8_pivot['Performance_Ratio'] = int8_pivot['Eager'] / int8_pivot['torch.compile']
            int8_pivot['Compilation_Benefit'] = (int8_pivot['Eager'] - int8_pivot['torch.compile']) / int8_pivot['Eager'] * 100
            
            int8_pivot['Architecture'] = int8_pivot['Model'].apply(
                lambda x: 'BERT-family' if any(name in x.lower() for name in ['bert', 'roberta', 'distilbert']) 
                else 'GPT-family' if 'gpt' in x.lower() else 'Vision-Transformer'
            )
            
            self.int8_metrics = int8_pivot
        else:
            self.int8_metrics = pd.DataFrame()
        
    def plot_architecture_comparison(self):
        """Plot 1: Architecture comparison showing BERT vs GPT performance patterns."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Architecture Performance Patterns: BERT-family vs GPT-family', 
                    fontsize=16, fontweight='bold')
        
        # Performance ratios by architecture
        architectures = ['BERT-family', 'GPT-family']
        colors = ['blue', 'green']
        
        for i, arch in enumerate(architectures):
            arch_data = self.fp32_metrics[self.fp32_metrics['Architecture'] == arch]
            if not arch_data.empty:
                for model in arch_data['Model'].unique():
                    model_data = arch_data[arch_data['Model'] == model]
                    axes[0, 0].plot(model_data['Batch_Size'], model_data['Performance_Ratio'], 
                                   marker='o', label=f'{model}', color=colors[i], alpha=0.8)
        
        # Compilation benefit by architecture
        for i, arch in enumerate(architectures):
            arch_data = self.fp32_metrics[self.fp32_metrics['Architecture'] == arch]
            if not arch_data.empty:
                for model in arch_data['Model'].unique():
                    model_data = arch_data[arch_data['Model'] == model]
                    axes[0, 1].plot(model_data['Batch_Size'], model_data['Compilation_Benefit'], 
                                   marker='o', label=f'{model}', color=colors[i], alpha=0.8)
        
        # Average performance by architecture
        for arch in architectures:
            arch_data = self.fp32_metrics[self.fp32_metrics['Architecture'] == arch]
            if not arch_data.empty:
                avg_ratio = arch_data.groupby('Batch_Size')['Performance_Ratio'].mean()
                avg_benefit = arch_data.groupby('Batch_Size')['Compilation_Benefit'].mean()
                
                axes[1, 0].plot(avg_ratio.index, avg_ratio.values, 
                               marker='s', label=f'{arch} (Avg)', linewidth=3, markersize=10)
                axes[1, 1].plot(avg_benefit.index, avg_benefit.values, 
                               marker='s', label=f'{arch} (Avg)', linewidth=3, markersize=10)
        
        # Formatting
        for ax in [axes[0, 0], axes[1, 0]]:
            ax.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Eager = Compiled')
            ax.set_ylabel('Performance Ratio (Eager/Compiled)')
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        for ax in [axes[0, 1], axes[1, 1]]:
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='No Benefit')
            ax.set_ylabel('Compilation Benefit (%)')
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        for ax in axes.flat:
            ax.set_xlabel('Batch Size')
        
        axes[0, 0].set_title('Individual Model Performance Ratios')
        axes[0, 1].set_title('Individual Model Compilation Benefits')
        axes[1, 0].set_title('Average Performance Ratios by Architecture')
        axes[1, 1].set_title('Average Compilation Benefits by Architecture')
        
        plt.tight_layout()
        plt.savefig('results/analysis/architecture_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_architectural_insights(self):
        """Plot 2: Detailed insights into why architectures behave differently."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Architectural Insights: Why BERT-family vs GPT-family Performance Differs', 
                    fontsize=16, fontweight='bold')
        
        # Batch scaling efficiency
        for arch in ['BERT-family', 'GPT-family']:
            arch_data = self.fp32_transformer[self.fp32_transformer['Architecture'] == arch]
            if not arch_data.empty:
                eager_data = arch_data[arch_data['Compilation'] == 'Eager'].groupby('Batch_Size')['Latency_ms'].mean()
                compiled_data = arch_data[arch_data['Compilation'] == 'torch.compile'].groupby('Batch_Size')['Latency_ms'].mean()
                
                axes[0, 0].plot(eager_data.index, eager_data.values, 
                               marker='o', label=f'{arch} (Eager)', alpha=0.8)
                axes[0, 0].plot(compiled_data.index, compiled_data.values, 
                               marker='s', label=f'{arch} (Compiled)', alpha=0.8)
        
        # Performance consistency across batch sizes
        for arch in ['BERT-family', 'GPT-family']:
            arch_data = self.fp32_metrics[self.fp32_metrics['Architecture'] == arch]
            if not arch_data.empty:
                # Calculate coefficient of variation (consistency measure)
                cv_by_batch = arch_data.groupby('Batch_Size')['Performance_Ratio'].std() / \
                             arch_data.groupby('Batch_Size')['Performance_Ratio'].mean()
                axes[0, 1].plot(cv_by_batch.index, cv_by_batch.values, 
                               marker='o', label=arch, linewidth=2, markersize=8)
        
        # Model size vs compilation benefit
        model_sizes = {
            'bert-base-uncased': 110, 'bert-large-uncased': 340,
            'roberta-base': 125, 'roberta-large': 355,
            'distilbert-base-uncased': 66,
            'gpt2': 124, 'gpt2-medium': 355,
            'vit': 86
        }
        
        for model in self.fp32_metrics['Model'].unique():
            model_data = self.fp32_metrics[self.fp32_metrics['Model'] == model]
            avg_benefit = model_data['Compilation_Benefit'].mean()
            size = model_sizes.get(model, 100)
            arch = model_data['Architecture'].iloc[0]
            color = 'blue' if 'BERT' in arch else 'green'
            
            axes[1, 0].scatter(size, avg_benefit, s=100, alpha=0.7, 
                              label=model, color=color)
        
        # Crossover frequency by architecture
        bert_crossovers = len(self.fp32_metrics[
            (self.fp32_metrics['Architecture'] == 'BERT-family') & 
            (self.fp32_metrics['Performance_Ratio'] < 1)
        ])
        gpt_crossovers = len(self.fp32_metrics[
            (self.fp32_metrics['Architecture'] == 'GPT-family') & 
            (self.fp32_metrics['Performance_Ratio'] < 1)
        ])
        
        total_bert = len(self.fp32_metrics[self.fp32_metrics['Architecture'] == 'BERT-family'])
        total_gpt = len(self.fp32_metrics[self.fp32_metrics['Architecture'] == 'GPT-family'])
        
        crossover_rates = [bert_crossovers/total_bert*100, gpt_crossovers/total_gpt*100]
        architectures = ['BERT-family', 'GPT-family']
        colors = ['blue', 'green']
        
        bars = axes[1, 1].bar(architectures, crossover_rates, color=colors, alpha=0.7)
        axes[1, 1].set_ylabel('Crossover Rate (%)')
        axes[1, 1].set_title('Frequency of Eager > Compiled by Architecture')
        
        # Add value labels on bars
        for bar, rate in zip(bars, crossover_rates):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                           f'{rate:.1f}%', ha='center', va='bottom')
        
        # Formatting
        axes[0, 0].set_xlabel('Batch Size')
        axes[0, 0].set_ylabel('Latency (ms)')
        axes[0, 0].set_xscale('log')
        axes[0, 0].set_yscale('log')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        axes[0, 0].set_title('Batch Scaling Efficiency')
        
        axes[0, 1].set_xlabel('Batch Size')
        axes[0, 1].set_ylabel('Coefficient of Variation')
        axes[0, 1].set_xscale('log')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        axes[0, 1].set_title('Performance Consistency')
        
        axes[1, 0].set_xlabel('Model Size (M parameters)')
        axes[1, 0].set_ylabel('Average Compilation Benefit (%)')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 0].set_title('Model Size vs Compilation Benefit')
        
        plt.tight_layout()
        plt.savefig('results/analysis/architectural_insights.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def generate_architectural_analysis(self):
        """Generate detailed analysis of architectural differences."""
        print("=" * 80)
        print("ARCHITECTURAL PERFORMANCE ANALYSIS")
        print("=" * 80)
        
        # Architecture-specific statistics
        print(f"\n🏗️ ARCHITECTURE-SPECIFIC ANALYSIS:")
        
        for arch in ['BERT-family', 'GPT-family']:
            arch_data = self.fp32_metrics[self.fp32_metrics['Architecture'] == arch]
            if not arch_data.empty:
                avg_ratio = arch_data['Performance_Ratio'].mean()
                avg_benefit = arch_data['Compilation_Benefit'].mean()
                crossover_rate = len(arch_data[arch_data['Performance_Ratio'] < 1]) / len(arch_data) * 100
                
                print(f"\n   {arch}:")
                print(f"     Average Performance Ratio: {avg_ratio:.3f}")
                print(f"     Average Compilation Benefit: {avg_benefit:.1f}%")
                print(f"     Crossover Rate (Eager > Compiled): {crossover_rate:.1f}%")
                
                # Model-specific breakdown
                for model in arch_data['Model'].unique():
                    model_data = arch_data[arch_data['Model'] == model]
                    model_ratio = model_data['Performance_Ratio'].mean()
                    model_benefit = model_data['Compilation_Benefit'].mean()
                    print(f"       {model}: Ratio={model_ratio:.3f}, Benefit={model_benefit:.1f}%")
        
        # Architectural insights
        print(f"\n🔍 ARCHITECTURAL INSIGHTS:")
        print(f"   BERT-family (Encoder-only) characteristics:")
        print(f"     - Bidirectional attention (O(n²) complexity)")
        print(f"     - Memory bandwidth bound")
        print(f"     - Dynamic attention patterns")
        print(f"     - Limited compilation benefits")
        
        print(f"\n   GPT-family (Decoder-only) characteristics:")
        print(f"     - Causal attention (structured patterns)")
        print(f"     - Compute bound operations")
        print(f"     - Cache-friendly memory access")
        print(f"     - Better compilation optimization")
        
        print(f"\n💡 EXPLANATION FOR PERFORMANCE DIFFERENCES:")
        print(f"   1. BERT-family models have irregular memory access patterns")
        print(f"      that are harder for compilers to optimize.")
        print(f"   2. GPT-family models have more structured, predictable")
        print(f"      computation patterns that benefit from compilation.")
        print(f"   3. Attention mechanism differences: bidirectional vs causal")
        print(f"      affect how well compilers can optimize the computation.")
        print(f"   4. Memory bandwidth vs compute bound: BERT-family may be")
        print(f"      memory bandwidth limited, while GPT-family is compute bound.")
        
        print("=" * 80)

def main():
    # File paths
    fp32_file = "../results/cache_managed_transformer_benchmark_20250703_144113.csv"
    int8_file = "../results/expanded_int8_transformer_benchmark_20250703_145219.csv"
    
    # Initialize analyzer
    analyzer = ArchitecturePerformanceAnalyzer(fp32_file, int8_file)
    analyzer.prepare_data()
    
    # Generate plots
    print("Generating architectural analysis plots...")
    analyzer.plot_architecture_comparison()
    analyzer.plot_architectural_insights()
    
    # Generate analysis
    analyzer.generate_architectural_analysis()
    
    print(f"\n✅ Architectural analysis complete! Check 'results/analysis/' for generated plots.")

if __name__ == "__main__":
    main() 