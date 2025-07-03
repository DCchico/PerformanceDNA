#!/usr/bin/env python3
"""
Hypothesis Formalization: Transformer Architecture Performance Patterns
Formalizes the hypothesis about BERT-family vs GPT-family performance differences
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class HypothesisFormalizer:
    def __init__(self, fp32_file, int8_file):
        self.fp32_file = fp32_file
        self.int8_file = int8_file
        self.hypotheses = {}
        
    def define_hypotheses(self):
        """Define formal hypotheses based on architectural observations."""
        
        print("=" * 80)
        print("HYPOTHESIS FORMALIZATION")
        print("=" * 80)
        
        # Primary Hypothesis
        print("\n🎯 PRIMARY HYPOTHESIS:")
        print("H1: Transformer architecture type significantly affects the relative performance")
        print("    of eager mode vs compiled mode execution.")
        print("\n   Specifically:")
        print("   H1a: BERT-family (encoder-only) models show better performance in eager mode")
        print("        compared to compiled mode across most batch sizes.")
        print("   H1b: GPT-family (decoder-only) models show better performance in compiled mode")
        print("        compared to eager mode across most batch sizes.")
        
        # Secondary Hypotheses
        print("\n🔬 SECONDARY HYPOTHESES:")
        print("H2: The performance difference is more pronounced at larger batch sizes")
        print("    due to memory bandwidth vs compute bound characteristics.")
        
        print("H3: Crossover points (where eager > compiled) occur more frequently")
        print("    in BERT-family models than in GPT-family models.")
        
        print("H4: Model size within each architecture family affects the magnitude")
        print("    of compilation benefits, but not the direction.")
        
        # Null Hypothesis
        print("\n❌ NULL HYPOTHESIS:")
        print("H0: There is no significant difference in eager vs compiled performance")
        print("    between BERT-family and GPT-family transformer architectures.")
        
        # Alternative Hypothesis
        print("\n✅ ALTERNATIVE HYPOTHESIS:")
        print("Ha: BERT-family models have significantly higher performance ratios")
        print("    (Eager/Compiled) compared to GPT-family models.")
        
        # Store hypotheses for testing
        self.hypotheses = {
            'primary': {
                'H1': 'Architecture type affects eager vs compiled performance',
                'H1a': 'BERT-family performs better in eager mode',
                'H1b': 'GPT-family performs better in compiled mode'
            },
            'secondary': {
                'H2': 'Performance difference increases with batch size',
                'H3': 'BERT-family has more crossover points',
                'H4': 'Model size affects magnitude but not direction'
            },
            'null': {
                'H0': 'No significant difference between architectures'
            },
            'alternative': {
                'Ha': 'BERT-family has higher performance ratios than GPT-family'
            }
        }
        
        print("\n📊 TESTABLE PREDICTIONS:")
        print("1. Mean performance ratio (Eager/Compiled) for BERT-family > 1.0")
        print("2. Mean performance ratio (Eager/Compiled) for GPT-family < 1.0")
        print("3. Statistical significance: p < 0.05 in t-test between architectures")
        print("4. Effect size: Cohen's d > 0.5 (medium effect)")
        print("5. Crossover rate: BERT-family > 50%, GPT-family < 50%")
        
    def load_and_prepare_data(self):
        """Load and prepare data for hypothesis testing."""
        print("\n📈 DATA PREPARATION:")
        
        # Load data
        self.fp32_data = pd.read_csv(self.fp32_file)
        self.int8_data = pd.read_csv(self.int8_file)
        
        # Filter transformer models
        transformer_models = ['bert-base-uncased', 'bert-large-uncased', 
                            'roberta-base', 'roberta-large', 'distilbert-base-uncased',
                            'gpt2', 'gpt2-medium', 'vit']
        
        self.fp32_transformer = self.fp32_data[
            self.fp32_data['Model'].isin(transformer_models)
        ].copy()
        
        # Classify architectures
        def classify_architecture(model_name):
            if any(name in model_name.lower() for name in ['bert', 'roberta', 'distilbert']):
                return 'BERT-family'
            elif 'gpt' in model_name.lower():
                return 'GPT-family'
            elif 'vit' in model_name.lower():
                return 'Vision-Transformer'
            else:
                return 'Other'
        
        self.fp32_transformer['Architecture'] = self.fp32_transformer['Model'].apply(classify_architecture)
        
        # Calculate performance metrics
        self.calculate_performance_metrics()
        
        print(f"   Loaded {len(self.fp32_transformer)} transformer measurements")
        print(f"   BERT-family models: {len(self.fp32_transformer[self.fp32_transformer['Architecture'] == 'BERT-family'])}")
        print(f"   GPT-family models: {len(self.fp32_transformer[self.fp32_transformer['Architecture'] == 'GPT-family'])}")
        
    def calculate_performance_metrics(self):
        """Calculate performance ratios and benefits."""
        # Pivot data
        pivot_data = self.fp32_transformer.pivot_table(
            index=['Model', 'Batch_Size'], 
            columns='Compilation', 
            values='Latency_ms', 
            aggfunc='mean'
        ).reset_index()
        
        pivot_data['Performance_Ratio'] = pivot_data['Eager'] / pivot_data['torch.compile']
        pivot_data['Compilation_Benefit'] = (pivot_data['Eager'] - pivot_data['torch.compile']) / pivot_data['Eager'] * 100
        
        # Add architecture classification
        pivot_data['Architecture'] = pivot_data['Model'].apply(
            lambda x: 'BERT-family' if any(name in x.lower() for name in ['bert', 'roberta', 'distilbert']) 
            else 'GPT-family' if 'gpt' in x.lower() else 'Vision-Transformer'
        )
        
        self.performance_data = pivot_data
        
    def test_hypotheses(self):
        """Perform statistical tests on the hypotheses."""
        print("\n🧪 STATISTICAL TESTING:")
        
        # Separate data by architecture
        bert_data = self.performance_data[self.performance_data['Architecture'] == 'BERT-family']
        gpt_data = self.performance_data[self.performance_data['Architecture'] == 'GPT-family']
        
        if bert_data.empty or gpt_data.empty:
            print("   ⚠️  Insufficient data for statistical testing")
            return
        
        # Test H1: Architecture difference in performance ratios
        print("\n   Testing H1: Architecture affects eager vs compiled performance")
        
        bert_ratios = bert_data['Performance_Ratio'].dropna()
        gpt_ratios = gpt_data['Performance_Ratio'].dropna()
        
        # T-test
        t_stat, p_value = stats.ttest_ind(bert_ratios, gpt_ratios)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(bert_ratios) - 1) * bert_ratios.var() + 
                             (len(gpt_ratios) - 1) * gpt_ratios.var()) / 
                            (len(bert_ratios) + len(gpt_ratios) - 2))
        cohens_d = (bert_ratios.mean() - gpt_ratios.mean()) / pooled_std
        
        print(f"     T-test: t = {t_stat:.3f}, p = {p_value:.6f}")
        print(f"     Effect size (Cohen's d): {cohens_d:.3f}")
        print(f"     BERT-family mean ratio: {bert_ratios.mean():.3f}")
        print(f"     GPT-family mean ratio: {gpt_ratios.mean():.3f}")
        
        # Test H1a: BERT-family > 1.0
        bert_t_stat, bert_p_value = stats.ttest_1samp(bert_ratios, 1.0)
        print(f"     BERT-family vs 1.0: t = {bert_t_stat:.3f}, p = {bert_p_value:.6f}")
        
        # Test H1b: GPT-family < 1.0
        gpt_t_stat, gpt_p_value = stats.ttest_1samp(gpt_ratios, 1.0)
        print(f"     GPT-family vs 1.0: t = {gpt_t_stat:.3f}, p = {gpt_p_value:.6f}")
        
        # Test H3: Crossover rates
        bert_crossovers = len(bert_data[bert_data['Performance_Ratio'] < 1])
        gpt_crossovers = len(gpt_data[gpt_data['Performance_Ratio'] < 1])
        
        bert_crossover_rate = bert_crossovers / len(bert_data) * 100
        gpt_crossover_rate = gpt_crossovers / len(gpt_data) * 100
        
        print(f"\n   Testing H3: Crossover frequency")
        print(f"     BERT-family crossover rate: {bert_crossover_rate:.1f}% ({bert_crossovers}/{len(bert_data)})")
        print(f"     GPT-family crossover rate: {gpt_crossover_rate:.1f}% ({gpt_crossovers}/{len(gpt_data)})")
        
        # Chi-square test for crossover rates
        contingency_table = [[bert_crossovers, len(bert_data) - bert_crossovers],
                           [gpt_crossovers, len(gpt_data) - gpt_crossovers]]
        chi2_stat, chi2_p_value = stats.chi2_contingency(contingency_table)[:2]
        print(f"     Chi-square test: χ² = {chi2_stat:.3f}, p = {chi2_p_value:.6f}")
        
        # Store results
        self.test_results = {
            'H1': {'t_stat': t_stat, 'p_value': p_value, 'cohens_d': cohens_d},
            'H1a': {'t_stat': bert_t_stat, 'p_value': bert_p_value},
            'H1b': {'t_stat': gpt_t_stat, 'p_value': gpt_p_value},
            'H3': {'chi2_stat': chi2_stat, 'chi2_p_value': chi2_p_value,
                   'bert_crossover_rate': bert_crossover_rate,
                   'gpt_crossover_rate': gpt_crossover_rate}
        }
        
    def interpret_results(self):
        """Interpret the statistical test results."""
        print("\n📊 HYPOTHESIS EVALUATION:")
        
        if not hasattr(self, 'test_results'):
            print("   No test results available")
            return
        
        # Evaluate H1
        print("\n   H1: Architecture affects eager vs compiled performance")
        if self.test_results['H1']['p_value'] < 0.05:
            print(f"     ✅ REJECT NULL HYPOTHESIS (p = {self.test_results['H1']['p_value']:.6f})")
            print(f"     ✅ SUPPORT PRIMARY HYPOTHESIS")
            
            effect_size = self.test_results['H1']['cohens_d']
            if abs(effect_size) > 0.8:
                effect_magnitude = "large"
            elif abs(effect_size) > 0.5:
                effect_magnitude = "medium"
            else:
                effect_magnitude = "small"
            print(f"     Effect size: {effect_magnitude} (Cohen's d = {effect_size:.3f})")
        else:
            print(f"     ❌ FAIL TO REJECT NULL HYPOTHESIS (p = {self.test_results['H1']['p_value']:.6f})")
        
        # Evaluate H1a
        print("\n   H1a: BERT-family performs better in eager mode")
        if self.test_results['H1a']['p_value'] < 0.05 and self.test_results['H1a']['t_stat'] > 0:
            print(f"     ✅ SUPPORTED (p = {self.test_results['H1a']['p_value']:.6f})")
        else:
            print(f"     ❌ NOT SUPPORTED (p = {self.test_results['H1a']['p_value']:.6f})")
        
        # Evaluate H1b
        print("\n   H1b: GPT-family performs better in compiled mode")
        if self.test_results['H1b']['p_value'] < 0.05 and self.test_results['H1b']['t_stat'] < 0:
            print(f"     ✅ SUPPORTED (p = {self.test_results['H1b']['p_value']:.6f})")
        else:
            print(f"     ❌ NOT SUPPORTED (p = {self.test_results['H1b']['p_value']:.6f})")
        
        # Evaluate H3
        print("\n   H3: BERT-family has more crossover points")
        if self.test_results['H3']['chi2_p_value'] < 0.05:
            print(f"     ✅ SUPPORTED (p = {self.test_results['H3']['chi2_p_value']:.6f})")
            print(f"     BERT-family: {self.test_results['H3']['bert_crossover_rate']:.1f}% crossovers")
            print(f"     GPT-family: {self.test_results['H3']['gpt_crossover_rate']:.1f}% crossovers")
        else:
            print(f"     ❌ NOT SUPPORTED (p = {self.test_results['H3']['chi2_p_value']:.6f})")
        
        print("\n🎯 CONCLUSION:")
        if (self.test_results['H1']['p_value'] < 0.05 and 
            self.test_results['H1a']['p_value'] < 0.05 and 
            self.test_results['H1b']['p_value'] < 0.05):
            print("   ✅ PRIMARY HYPOTHESIS FULLY SUPPORTED")
            print("   ✅ BERT-family models prefer eager mode")
            print("   ✅ GPT-family models prefer compiled mode")
        elif self.test_results['H1']['p_value'] < 0.05:
            print("   ⚠️  PARTIAL SUPPORT for primary hypothesis")
            print("   📊 Further investigation needed for sub-hypotheses")
        else:
            print("   ❌ PRIMARY HYPOTHESIS NOT SUPPORTED")
            print("   📊 No significant difference between architectures")
        
    def create_hypothesis_summary(self):
        """Create a summary document of the hypothesis and results."""
        summary = f"""
HYPOTHESIS SUMMARY REPORT
=========================

PRIMARY HYPOTHESIS:
H1: Transformer architecture type significantly affects the relative performance of eager mode vs compiled mode execution.

Sub-hypotheses:
H1a: BERT-family (encoder-only) models show better performance in eager mode compared to compiled mode.
H1b: GPT-family (decoder-only) models show better performance in compiled mode compared to eager mode.

THEORETICAL BASIS:
1. BERT-family: Bidirectional attention, memory bandwidth bound, irregular memory access
2. GPT-family: Causal attention, compute bound, structured patterns, cache-friendly

TESTABLE PREDICTIONS:
1. Mean performance ratio (Eager/Compiled) for BERT-family > 1.0
2. Mean performance ratio (Eager/Compiled) for GPT-family < 1.0
3. Statistical significance: p < 0.05 in t-test between architectures
4. Effect size: Cohen's d > 0.5 (medium effect)
5. Crossover rate: BERT-family > 50%, GPT-family < 50%

STATISTICAL TESTS:
- Independent t-test for architecture comparison
- One-sample t-test for each architecture vs 1.0
- Chi-square test for crossover frequency
- Effect size calculation (Cohen's d)

SIGNIFICANCE LEVEL: α = 0.05
"""
        
        # Save summary
        Path("results/analysis").mkdir(parents=True, exist_ok=True)
        with open("results/analysis/hypothesis_summary.txt", "w") as f:
            f.write(summary)
        
        print(f"\n📄 Hypothesis summary saved to: results/analysis/hypothesis_summary.txt")

def main():
    # File paths
    fp32_file = "../results/cache_managed_transformer_benchmark_20250703_144113.csv"
    int8_file = "../results/expanded_int8_transformer_benchmark_20250703_145219.csv"
    
    # Initialize formalizer
    formalizer = HypothesisFormalizer(fp32_file, int8_file)
    
    # Define hypotheses
    formalizer.define_hypotheses()
    
    # Load and prepare data
    formalizer.load_and_prepare_data()
    
    # Test hypotheses
    formalizer.test_hypotheses()
    
    # Interpret results
    formalizer.interpret_results()
    
    # Create summary
    formalizer.create_hypothesis_summary()
    
    print(f"\n✅ Hypothesis formalization complete!")

if __name__ == "__main__":
    main() 