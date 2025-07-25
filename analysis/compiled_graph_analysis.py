#!/usr/bin/env python3
"""
Compiled Graph Analysis for BERT
Analyzes how TorchInductor compilation affects BERT's memory layout and performance
"""

import torch
import torch.nn as nn
from transformers import BertModel
import time
import gc
from typing import Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

class BERTCompilationAnalyzer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eager_model = None
        self.compiled_model = None
        
    def load_models(self):
        """Load BERT models in both eager and compiled modes."""
        print("Loading BERT models...")
        
        # Load eager model
        self.eager_model = BertModel.from_pretrained("bert-base-uncased").eval().to(self.device)
        
        # Load and compile model
        self.compiled_model = BertModel.from_pretrained("bert-base-uncased").eval().to(self.device)
        self.compiled_model = torch.compile(self.compiled_model, backend="inductor")
        
        print("✓ Models loaded successfully")
        
    def analyze_tensor_properties(self, batch_size: int = 1) -> Dict[str, Any]:
        """Analyze tensor properties and memory layouts for both models."""
        print(f"\n=== Tensor Properties Analysis (batch_size={batch_size}) ===")
        
        # Create dummy input
        sequence_length = 512
        dummy_input = torch.randint(0, 30000, (batch_size, sequence_length), dtype=torch.long).to(self.device)
        
        results = {
            'eager': {},
            'compiled': {}
        }
        
        # Analyze eager model tensors
        print("\n--- Eager Model Analysis ---")
        with torch.no_grad():
            # Hook to capture intermediate tensors
            eager_tensors = {}
            
            def eager_hook(name):
                def hook(module, input, output):
                    if isinstance(output, torch.Tensor):
                        eager_tensors[name] = {
                            'shape': output.shape,
                            'dtype': output.dtype,
                            'device': output.device,
                            'stride': output.stride(),
                            'is_contiguous': output.is_contiguous(),
                            'storage_offset': output.storage_offset(),
                            'element_size': output.element_size(),
                            'numel': output.numel(),
                            'storage_size': output.storage().size(),
                            'is_channels_last': output.is_contiguous(memory_format=torch.channels_last) if output.dim() >= 4 else False,
                            'is_channels_last_3d': output.is_contiguous(memory_format=torch.channels_last_3d) if output.dim() >= 5 else False,
                        }
                return hook
            
            # Register hooks for key layers
            hooks = []
            for name, module in self.eager_model.named_modules():
                if 'attention' in name or 'intermediate' in name or 'output' in name:
                    hook = module.register_forward_hook(eager_hook(name))
                    hooks.append(hook)
            
            # Forward pass
            _ = self.eager_model(dummy_input)
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
            
            results['eager'] = eager_tensors
            
        # Analyze compiled model tensors
        print("\n--- Compiled Model Analysis ---")
        with torch.no_grad():
            # Hook to capture intermediate tensors
            compiled_tensors = {}
            
            def compiled_hook(name):
                def hook(module, input, output):
                    if isinstance(output, torch.Tensor):
                        compiled_tensors[name] = {
                            'shape': output.shape,
                            'dtype': output.dtype,
                            'device': output.device,
                            'stride': output.stride(),
                            'is_contiguous': output.is_contiguous(),
                            'storage_offset': output.storage_offset(),
                            'element_size': output.element_size(),
                            'numel': output.numel(),
                            'storage_size': output.storage().size(),
                            'is_channels_last': output.is_contiguous(memory_format=torch.channels_last) if output.dim() >= 4 else False,
                            'is_channels_last_3d': output.is_contiguous(memory_format=torch.channels_last_3d) if output.dim() >= 5 else False,
                        }
                return hook
            
            # Register hooks for key layers
            hooks = []
            for name, module in self.compiled_model.named_modules():
                if 'attention' in name or 'intermediate' in name or 'output' in name:
                    hook = module.register_forward_hook(compiled_hook(name))
                    hooks.append(hook)
            
            # Forward pass
            _ = self.compiled_model(dummy_input)
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
            
            results['compiled'] = compiled_tensors
            
        return results
    
    def compare_tensor_layouts(self, tensor_analysis: Dict[str, Any]):
        """Compare tensor layouts between eager and compiled models."""
        print("\n=== Tensor Layout Comparison ===")
        
        eager_tensors = tensor_analysis['eager']
        compiled_tensors = tensor_analysis['compiled']
        
        differences = []
        
        for name in eager_tensors.keys():
            if name in compiled_tensors:
                eager = eager_tensors[name]
                compiled = compiled_tensors[name]
                
                # Compare key properties
                layout_diff = {
                    'name': name,
                    'channels_last_diff': eager['is_channels_last'] != compiled['is_channels_last'],
                    'channels_last_3d_diff': eager['is_channels_last_3d'] != compiled['is_channels_last_3d'],
                    'stride_diff': eager['stride'] != compiled['stride'],
                    'contiguous_diff': eager['is_contiguous'] != compiled['is_contiguous'],
                    'storage_size_diff': eager['storage_size'] != compiled['storage_size'],
                }
                
                if any(layout_diff.values()):
                    differences.append(layout_diff)
                    print(f"\n🔍 Differences found in {name}:")
                    print(f"  Channels last: {eager['is_channels_last']} vs {compiled['is_channels_last']}")
                    print(f"  Channels last 3D: {eager['is_channels_last_3d']} vs {compiled['is_channels_last_3d']}")
                    print(f"  Stride: {eager['stride']} vs {compiled['stride']}")
                    print(f"  Contiguous: {eager['is_contiguous']} vs {compiled['is_contiguous']}")
                    print(f"  Storage size: {eager['storage_size']} vs {compiled['storage_size']}")
        
        if not differences:
            print("✓ No significant tensor layout differences found")
            
        return differences
    
    def analyze_memory_allocation(self, batch_sizes: list = [1, 8, 32]):
        """Analyze memory allocation patterns across different batch sizes."""
        print(f"\n=== Memory Allocation Analysis ===")
        
        results = {}
        
        for batch_size in batch_sizes:
            print(f"\n--- Batch Size: {batch_size} ---")
            
            # Clear cache
            torch.cuda.empty_cache()
            gc.collect()
            
            # Measure memory before
            torch.cuda.synchronize()
            memory_before = torch.cuda.memory_allocated()
            
            # Create input
            sequence_length = 512
            dummy_input = torch.randint(0, 30000, (batch_size, sequence_length), dtype=torch.long).to(self.device)
            
            # Eager forward pass
            with torch.no_grad():
                _ = self.eager_model(dummy_input)
            
            torch.cuda.synchronize()
            memory_after_eager = torch.cuda.memory_allocated()
            memory_eager = memory_after_eager - memory_before
            
            # Clear and measure compiled
            torch.cuda.empty_cache()
            gc.collect()
            
            torch.cuda.synchronize()
            memory_before = torch.cuda.memory_allocated()
            
            # Compiled forward pass
            with torch.no_grad():
                _ = self.compiled_model(dummy_input)
            
            torch.cuda.synchronize()
            memory_after_compiled = torch.cuda.memory_allocated()
            memory_compiled = memory_after_compiled - memory_before
            
            results[batch_size] = {
                'eager_memory': memory_eager,
                'compiled_memory': memory_compiled,
                'memory_diff': memory_compiled - memory_eager,
                'memory_ratio': memory_compiled / memory_eager if memory_eager > 0 else float('inf')
            }
            
            print(f"  Eager memory: {memory_eager / 1024**2:.2f} MB")
            print(f"  Compiled memory: {memory_compiled / 1024**2:.2f} MB")
            print(f"  Difference: {(memory_compiled - memory_eager) / 1024**2:.2f} MB")
            print(f"  Ratio: {memory_compiled / memory_eager:.3f}x")
        
        return results
    
    def analyze_graph_structure(self):
        """Analyze the computational graph structure differences."""
        print("\n=== Graph Structure Analysis ===")
        
        # Create dummy input for graph analysis
        dummy_input = torch.randint(0, 30000, (1, 512), dtype=torch.long).to(self.device)
        
        # Analyze eager graph (using torch.fx)
        print("\n--- Eager Graph Structure ---")
        try:
            from torch.fx import symbolic_trace
            eager_graph = symbolic_trace(self.eager_model)
            print(f"  Number of nodes: {len(eager_graph.graph.nodes)}")
            
            # Count operation types
            op_counts = {}
            for node in eager_graph.graph.nodes:
                op = node.op
                op_counts[op] = op_counts.get(op, 0) + 1
            
            print("  Operation distribution:")
            for op, count in op_counts.items():
                print(f"    {op}: {count}")
                
        except Exception as e:
            print(f"  Could not analyze eager graph: {e}")
        
        # Analyze compiled graph
        print("\n--- Compiled Graph Structure ---")
        try:
            # Get the compiled graph info
            compiled_info = self.compiled_model._orig_mod._compiled_graph_info
            print(f"  Compiled graph info available: {compiled_info is not None}")
            
            # Try to access graph structure
            if hasattr(self.compiled_model, '_orig_mod'):
                print(f"  Original module preserved: ✓")
                
        except Exception as e:
            print(f"  Could not analyze compiled graph: {e}")
    
    def performance_analysis_by_batch_size(self, batch_sizes: list = [1, 2, 4, 8, 16, 32]):
        """Analyze performance differences across batch sizes."""
        print(f"\n=== Performance Analysis by Batch Size ===")
        
        results = {}
        
        for batch_size in batch_sizes:
            print(f"\n--- Batch Size: {batch_size} ---")
            
            # Create input
            sequence_length = 512
            dummy_input = torch.randint(0, 30000, (batch_size, sequence_length), dtype=torch.long).to(self.device)
            
            # Warm up
            for _ in range(5):
                with torch.no_grad():
                    _ = self.eager_model(dummy_input)
                    _ = self.compiled_model(dummy_input)
            
            torch.cuda.synchronize()
            
            # Benchmark eager
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            for _ in range(10):
                with torch.no_grad():
                    _ = self.eager_model(dummy_input)
            end.record()
            
            torch.cuda.synchronize()
            eager_time = start.elapsed_time(end) / 10
            
            # Benchmark compiled
            start.record()
            for _ in range(10):
                with torch.no_grad():
                    _ = self.compiled_model(dummy_input)
            end.record()
            
            torch.cuda.synchronize()
            compiled_time = start.elapsed_time(end) / 10
            
            results[batch_size] = {
                'eager_time': eager_time,
                'compiled_time': compiled_time,
                'speedup': eager_time / compiled_time,
                'slowdown': compiled_time / eager_time
            }
            
            print(f"  Eager: {eager_time:.3f} ms")
            print(f"  Compiled: {compiled_time:.3f} ms")
            print(f"  Speedup: {eager_time / compiled_time:.3f}x")
            print(f"  Slowdown: {compiled_time / eager_time:.3f}x")
        
        return results
    
    def generate_insights(self, tensor_analysis: Dict, memory_analysis: Dict, performance_analysis: Dict):
        """Generate insights about compilation effects."""
        print("\n=== Insights and Analysis ===")
        
        print("\n🔍 Key Findings:")
        
        # Memory layout insights
        layout_diffs = self.compare_tensor_layouts(tensor_analysis)
        if layout_diffs:
            print("1. Memory Layout Changes:")
            print("   - Compilation changes tensor memory layouts")
            print("   - This affects cache efficiency and memory bandwidth")
            print("   - Explains the L1 hit rate and memory throughput differences")
        
        # Memory allocation insights
        memory_trends = []
        for batch_size, data in memory_analysis.items():
            memory_trends.append((batch_size, data['memory_ratio']))
        
        print("\n2. Memory Allocation Patterns:")
        for batch_size, ratio in memory_trends:
            print(f"   - Batch {batch_size}: {ratio:.3f}x memory usage")
        
        # Performance insights
        performance_trends = []
        for batch_size, data in performance_analysis.items():
            performance_trends.append((batch_size, data['slowdown']))
        
        print("\n3. Performance Trends:")
        for batch_size, slowdown in performance_trends:
            print(f"   - Batch {batch_size}: {slowdown:.3f}x slowdown")
        
        # Why compilation changes memory layout
        print("\n🤔 Why Does Compilation Change Memory Layout?")
        print("1. Fusion Preparation:")
        print("   - Compiler may change layouts to enable future kernel fusion")
        print("   - Different memory formats can enable more efficient fused kernels")
        
        print("\n2. Optimization Trade-offs:")
        print("   - Small batch: Memory layout changes may enable fusion benefits")
        print("   - Large batch: Memory bandwidth becomes bottleneck, layout changes hurt")
        
        print("\n3. Compiler Heuristics:")
        print("   - Inductor optimizes for typical use cases (smaller batches)")
        print("   - May not account for memory-bound scenarios at large batch sizes")
        
        print("\n💡 Recommendations:")
        print("1. Profile at target batch sizes")
        print("2. Consider memory layout hints for large batch inference")
        print("3. Investigate if layout changes can be controlled/optimized")

def main():
    """Main analysis function."""
    print("BERT Compilation Analysis")
    print("=" * 50)
    
    analyzer = BERTCompilationAnalyzer()
    
    try:
        # Load models
        analyzer.load_models()
        
        # Analyze tensor properties
        tensor_analysis = analyzer.analyze_tensor_properties(batch_size=1)
        
        # Analyze memory allocation
        memory_analysis = analyzer.analyze_memory_allocation([1, 8, 32])
        
        # Analyze graph structure
        analyzer.analyze_graph_structure()
        
        # Performance analysis
        performance_analysis = analyzer.performance_analysis_by_batch_size([1, 2, 4, 8, 16, 32])
        
        # Generate insights
        analyzer.generate_insights(tensor_analysis, memory_analysis, performance_analysis)
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
