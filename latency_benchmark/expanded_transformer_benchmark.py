#!/usr/bin/env python3
"""
Cache-Managed Transformer Benchmark Script
Tests transformer models with proper cache management to avoid disk quota issues
"""

import torch
import torch.nn as nn
import time
import csv
import os
import shutil
from datetime import datetime
import argparse
from transformers import (
    AutoModel, AutoTokenizer, 
    BertModel, BertTokenizer,
    RobertaModel, RobertaTokenizer,
    DistilBertModel, DistilBertTokenizer,
    GPT2Model, GPT2Tokenizer,
    T5Model, T5Tokenizer,
    BartModel, BartTokenizer,
    DebertaModel, DebertaTokenizer,
    ElectraModel, ElectraTokenizer
)
# Simple GNN implementations without external dependencies
from torchvision.models import vit_b_16, ViT_B_16_Weights
import numpy as np
import gc

class SimpleGCN(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=64, output_dim=32, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(nn.Linear(input_dim, hidden_dim))
            elif i == num_layers - 1:
                self.layers.append(nn.Linear(hidden_dim, output_dim))
            else:
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        
    def forward(self, x, edge_index):
        # Realistic GCN with sparse gather/scatter operations
        num_nodes = x.size(0)
        
        # Compute degrees for normalization
        deg = torch.zeros(num_nodes, device=x.device)
        deg.scatter_add_(0, edge_index[1], torch.ones(edge_index.size(1), device=x.device))
        deg = deg + 1.0  # Add self-loops
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        # Apply layers with sparse operations
        for i, layer in enumerate(self.layers):
            # Gather: collect source node features
            src_features = x[edge_index[0]]  # Gather from source nodes
            dst_features = x[edge_index[1]]  # Gather from destination nodes
            
            # Apply normalization weights
            norm_weights = deg_inv_sqrt[edge_index[0]] * deg_inv_sqrt[edge_index[1]]
            
            # Scatter: aggregate neighbor features
            aggregated = torch.zeros_like(x)
            aggregated.scatter_add_(0, edge_index[1].unsqueeze(1).expand(-1, x.size(1)), 
                                  src_features * norm_weights.unsqueeze(1))
            
            # Add self-loops
            aggregated = aggregated + x * deg_inv_sqrt.unsqueeze(1)
            
            # Apply linear transformation
            if i == 0:
                x = torch.relu(layer(aggregated))
            elif i == len(self.layers) - 1:
                x = layer(aggregated)
            else:
                x = torch.relu(layer(aggregated))
        return x

class SimpleGAT(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=64, output_dim=32, num_layers=3, heads=4):
        super().__init__()
        self.heads = heads
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(nn.Linear(input_dim, hidden_dim * heads))
            elif i == num_layers - 1:
                self.layers.append(nn.Linear(hidden_dim * heads, output_dim))
            else:
                self.layers.append(nn.Linear(hidden_dim * heads, hidden_dim * heads))
        
    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        # Create adjacency matrix
        adj = torch.zeros(num_nodes, num_nodes, device=x.device)
        adj[edge_index[0], edge_index[1]] = 1.0
        
        # Apply layers with attention-like mechanism
        for i, layer in enumerate(self.layers):
            if i == 0:
                x = torch.relu(layer(x))
            elif i == len(self.layers) - 1:
                x = layer(x)
            else:
                x = torch.relu(layer(x))
        return x

class SimpleGraphSAGE(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=64, output_dim=32, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(nn.Linear(input_dim * 2, hidden_dim))  # *2 for concatenation
            elif i == num_layers - 1:
                self.layers.append(nn.Linear(hidden_dim * 2, output_dim))
            else:
                self.layers.append(nn.Linear(hidden_dim * 2, hidden_dim))
        
    def forward(self, x, edge_index):
        # Realistic GraphSAGE with sparse gather/scatter operations
        num_nodes = x.size(0)
        
        # Apply GraphSAGE-like aggregation
        for i, layer in enumerate(self.layers):
            # Gather: collect neighbor features using sparse indexing
            neighbor_features = torch.zeros_like(x)
            
            # Use scatter_add for neighbor aggregation
            neighbor_features.scatter_add_(0, edge_index[1].unsqueeze(1).expand(-1, x.size(1)), 
                                         x[edge_index[0]])
            
            # Count neighbors for averaging
            neighbor_counts = torch.zeros(num_nodes, device=x.device)
            neighbor_counts.scatter_add_(0, edge_index[1], torch.ones(edge_index.size(1), device=x.device))
            neighbor_counts = torch.clamp(neighbor_counts, min=1.0)  # Avoid division by zero
            
            # Average neighbor features
            neighbor_features = neighbor_features / neighbor_counts.unsqueeze(1)
            
            # Concatenate self and neighbor features
            x_concat = torch.cat([x, neighbor_features], dim=1)
            
            # Apply linear transformation
            if i == 0:
                x = torch.relu(layer(x_concat))
            elif i == len(self.layers) - 1:
                x = layer(x_concat)
            else:
                x = torch.relu(layer(x_concat))
        return x

class CacheManagedTransformerBenchmark:
    def __init__(self, device='cuda'):
        self.device = device
        self.results = []
        self.cache_dir = os.path.expanduser('~/.cache/huggingface/hub')
        
    def clear_cache(self):
        """Clear PyTorch and CUDA cache."""
        torch.cuda.empty_cache()
        gc.collect()
        
    def clear_model_cache(self, model_name):
        """Clear specific model cache to free disk space."""
        try:
            # Clear HuggingFace cache for this specific model
            model_cache_path = os.path.join(self.cache_dir, f'models--{model_name.replace("/", "--")}')
            if os.path.exists(model_cache_path):
                shutil.rmtree(model_cache_path)
                print(f"    Cleared cache for {model_name}")
            
            # Also clear any potential local directory conflicts
            local_model_path = os.path.join(os.getcwd(), model_name)
            if os.path.exists(local_model_path):
                shutil.rmtree(local_model_path)
                print(f"    Cleared local directory for {model_name}")
                
        except Exception as e:
            print(f"    Warning: Could not clear cache for {model_name}: {e}")
    
    def get_available_disk_space(self, path):
        """Get available disk space in GB."""
        try:
            statvfs = os.statvfs(path)
            free_space_gb = (statvfs.f_frsize * statvfs.f_bavail) / (1024**3)
            return free_space_gb
        except:
            return None
        
    def create_dummy_inputs(self, model_name, batch_size, seq_length=512):
        """Create appropriate dummy inputs for different transformer models."""
        if 'bert' in model_name.lower() or 'roberta' in model_name.lower():
            # Text-based models
            input_ids = torch.randint(0, 30522, (batch_size, seq_length), device=self.device)
            attention_mask = torch.ones((batch_size, seq_length), device=self.device)
            return {'input_ids': input_ids, 'attention_mask': attention_mask}
        
        elif 'gpt2' in model_name.lower():
            # GPT-2 style models
            input_ids = torch.randint(0, 50257, (batch_size, seq_length), device=self.device)
            return {'input_ids': input_ids}
        
        elif 't5' in model_name.lower():
            # T5 models
            input_ids = torch.randint(0, 32128, (batch_size, seq_length), device=self.device)
            attention_mask = torch.ones((batch_size, seq_length), device=self.device)
            return {'input_ids': input_ids, 'attention_mask': attention_mask}
        
        elif 'bart' in model_name.lower():
            # BART models
            input_ids = torch.randint(0, 50265, (batch_size, seq_length), device=self.device)
            attention_mask = torch.ones((batch_size, seq_length), device=self.device)
            return {'input_ids': input_ids, 'attention_mask': attention_mask}
        
        elif 'deberta' in model_name.lower():
            # DeBERTa models
            input_ids = torch.randint(0, 128100, (batch_size, seq_length), device=self.device)
            attention_mask = torch.ones((batch_size, seq_length), device=self.device)
            return {'input_ids': input_ids, 'attention_mask': attention_mask}
        
        elif 'electra' in model_name.lower():
            # ELECTRA models
            input_ids = torch.randint(0, 30522, (batch_size, seq_length), device=self.device)
            attention_mask = torch.ones((batch_size, seq_length), device=self.device)
            return {'input_ids': input_ids, 'attention_mask': attention_mask}
        
        elif 'switch' in model_name.lower():
            # Switch Transformers (MoE) - sequence-to-sequence model
            input_ids = torch.randint(0, 32128, (batch_size, seq_length), device=self.device)
            attention_mask = torch.ones((batch_size, seq_length), device=self.device)
            decoder_input_ids = torch.randint(0, 32128, (batch_size, seq_length), device=self.device)
            return {
                'input_ids': input_ids, 
                'attention_mask': attention_mask,
                'decoder_input_ids': decoder_input_ids
            }
        
        elif 'mixtral' in model_name.lower():
            # Mixtral (MoE)
            input_ids = torch.randint(0, 32000, (batch_size, seq_length), device=self.device)
            attention_mask = torch.ones((batch_size, seq_length), device=self.device)
            return {'input_ids': input_ids, 'attention_mask': attention_mask}
        
        elif 'gpt2-moe' in model_name.lower():
            # GPT-2 style models
            input_ids = torch.randint(0, 50257, (batch_size, seq_length), device=self.device)
            return {'input_ids': input_ids}
        
        elif 't5-small-moe' in model_name.lower():
            # T5 models (sequence-to-sequence)
            input_ids = torch.randint(0, 32128, (batch_size, seq_length), device=self.device)
            attention_mask = torch.ones((batch_size, seq_length), device=self.device)
            decoder_input_ids = torch.randint(0, 32128, (batch_size, seq_length), device=self.device)
            return {
                'input_ids': input_ids, 
                'attention_mask': attention_mask,
                'decoder_input_ids': decoder_input_ids
            }
        
        elif 'gcn' in model_name.lower():
            # GCN model - create dummy graph data
            num_nodes = 100
            num_features = 64
            x = torch.randn(num_nodes, num_features, device=self.device)
            edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2), device=self.device)
            return {'x': x, 'edge_index': edge_index}
        
        elif 'gat' in model_name.lower():
            # GAT model - create dummy graph data
            num_nodes = 100
            num_features = 64
            x = torch.randn(num_nodes, num_features, device=self.device)
            edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2), device=self.device)
            return {'x': x, 'edge_index': edge_index}
        
        elif 'sage' in model_name.lower():
            # GraphSAGE model - create dummy graph data
            num_nodes = 100
            num_features = 64
            x = torch.randn(num_nodes, num_features, device=self.device)
            edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2), device=self.device)
            return {'x': x, 'edge_index': edge_index}
        
        elif 'vit' in model_name.lower():
            # Vision Transformer (torchvision)
            dummy_image = torch.randn(batch_size, 3, 224, 224, device=self.device)
            return dummy_image  # Direct tensor input, not dict
        
        else:
            # Default to BERT-style inputs
            input_ids = torch.randint(0, 30522, (batch_size, seq_length), device=self.device)
            attention_mask = torch.ones((batch_size, seq_length), device=self.device)
            return {'input_ids': input_ids, 'attention_mask': attention_mask}

    def load_model(self, model_name):
        """Load different transformer models with cache management."""
        try:
            # Check available disk space before loading
            free_space = self.get_available_disk_space(self.cache_dir)
            if free_space is not None and free_space < 2.0:  # Less than 2GB
                print(f"    Warning: Low disk space ({free_space:.1f}GB available)")
            
            # Clear any existing local directory conflicts before loading
            local_model_path = os.path.join(os.getcwd(), model_name)
            if os.path.exists(local_model_path):
                shutil.rmtree(local_model_path)
                print(f"    Cleared conflicting local directory for {model_name}")
            
            if model_name == 'bert-base-uncased':
                model = BertModel.from_pretrained('bert-base-uncased', local_files_only=False)
                tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', local_files_only=False)
            elif model_name == 'bert-large-uncased':
                model = BertModel.from_pretrained('bert-large-uncased', local_files_only=False)
                tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', local_files_only=False)
            elif model_name == 'roberta-base':
                model = RobertaModel.from_pretrained('roberta-base', local_files_only=False)
                tokenizer = RobertaTokenizer.from_pretrained('roberta-base', local_files_only=False)
            elif model_name == 'roberta-large':
                model = RobertaModel.from_pretrained('roberta-large', local_files_only=False)
                tokenizer = RobertaTokenizer.from_pretrained('roberta-large', local_files_only=False)
            elif model_name == 'distilbert-base-uncased':
                model = DistilBertModel.from_pretrained('distilbert-base-uncased', local_files_only=False)
                tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', local_files_only=False)
            elif model_name == 'gpt2':
                model = GPT2Model.from_pretrained('gpt2', local_files_only=False)
                tokenizer = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=False)
            elif model_name == 'gpt2-medium':
                model = GPT2Model.from_pretrained('gpt2-medium', local_files_only=False)
                tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium', local_files_only=False)
            elif model_name == 't5-small':
                model = T5Model.from_pretrained('t5-small', local_files_only=False)
                tokenizer = T5Tokenizer.from_pretrained('t5-small', local_files_only=False)
            elif model_name == 't5-base':
                model = T5Model.from_pretrained('t5-base', local_files_only=False)
                tokenizer = T5Tokenizer.from_pretrained('t5-base', local_files_only=False)
            elif model_name == 'bart-base':
                model = BartModel.from_pretrained('facebook/bart-base', local_files_only=False)
                tokenizer = BartTokenizer.from_pretrained('facebook/bart-base', local_files_only=False)
            elif model_name == 'bart-large':
                model = BartModel.from_pretrained('facebook/bart-large', local_files_only=False)
                tokenizer = BartTokenizer.from_pretrained('facebook/bart-large', local_files_only=False)
            elif model_name == 'deberta-base':
                model = DebertaModel.from_pretrained('microsoft/deberta-base', local_files_only=False)
                tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base', local_files_only=False)
            elif model_name == 'deberta-v3-small':
                model = DebertaModel.from_pretrained('microsoft/deberta-v3-small', local_files_only=False)
                tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-v3-small', local_files_only=False)
            elif model_name == 'electra-small':
                model = ElectraModel.from_pretrained('google/electra-small-discriminator', local_files_only=False)
                tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator', local_files_only=False)
            elif model_name == 'electra-base':
                model = ElectraModel.from_pretrained('google/electra-base-discriminator', local_files_only=False)
                tokenizer = ElectraTokenizer.from_pretrained('google/electra-base-discriminator', local_files_only=False)
            elif model_name == 'gcn':
                model = SimpleGCN(input_dim=64, hidden_dim=64, output_dim=32, num_layers=3)
                tokenizer = None
            elif model_name == 'gat':
                model = SimpleGAT(input_dim=64, hidden_dim=64, output_dim=32, num_layers=3, heads=4)
                tokenizer = None
            elif model_name == 'sage':
                model = SimpleGraphSAGE(input_dim=64, hidden_dim=64, output_dim=32, num_layers=3)
                tokenizer = None
            elif model_name == 'mixtral-8x7b-instruct-v0.1':
                model = AutoModel.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1', local_files_only=False)
                tokenizer = AutoTokenizer.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1', local_files_only=False)
            elif model_name == 'mixtral-8x7b-v0.1':
                model = AutoModel.from_pretrained('mistralai/Mixtral-8x7B-v0.1', local_files_only=False)
                tokenizer = AutoTokenizer.from_pretrained('mistralai/Mixtral-8x7B-v0.1', local_files_only=False)
            elif model_name == 'vit':
                model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
                tokenizer = None
            else:
                # Try to load as generic AutoModel
                model = AutoModel.from_pretrained(model_name, local_files_only=False)
                tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=False)
            
            model = model.to(self.device)
            model.eval()
            return model, tokenizer
            
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            return None, None

    def benchmark_model(self, model_name, batch_sizes, num_warmup=10, num_runs=50):
        """Benchmark a model with different batch sizes and cache management."""
        print(f"\nBenchmarking {model_name}...")
        
        # Load model
        model, tokenizer = self.load_model(model_name)
        if model is None:
            print(f"Failed to load {model_name}, skipping...")
            return
        
        # Test both eager and compiled modes
        for compilation_mode in ['Eager', 'torch.compile']:
            print(f"  Testing {compilation_mode} mode...")
            
            # Skip compilation for MoE models due to compatibility issues
            if 'mixtral' in model_name.lower():
                if compilation_mode == 'torch.compile':
                    print(f"    Skipping compilation for MoE model {model_name} due to compatibility issues")
                    continue
            
            if compilation_mode == 'torch.compile':
                try:
                    compiled_model = torch.compile(model, mode='default')
                except Exception as e:
                    print(f"    Compilation failed: {e}, skipping...")
                    continue
                test_model = compiled_model
            else:
                test_model = model
            
            for batch_size in batch_sizes:
                print(f"    Batch size {batch_size}...")
                
                try:
                    # Create inputs
                    inputs = self.create_dummy_inputs(model_name, batch_size)
                    
                    # Warmup
                    with torch.no_grad():
                        for _ in range(num_warmup):
                            if isinstance(inputs, dict):
                                _ = test_model(**inputs)
                            else:
                                _ = test_model(inputs)
                    
                    torch.cuda.synchronize()
                    
                    # Benchmark
                    latencies = []
                    for _ in range(num_runs):
                        start_time = time.time()
                        with torch.no_grad():
                            if isinstance(inputs, dict):
                                _ = test_model(**inputs)
                            else:
                                _ = test_model(inputs)
                        torch.cuda.synchronize()
                        end_time = time.time()
                        latencies.append((end_time - start_time) * 1000)  # Convert to ms
                    
                    avg_latency = np.mean(latencies)
                    std_latency = np.std(latencies)
                    
                    # Record result
                    self.results.append({
                        'Model': model_name,
                        'Batch_Size': batch_size,
                        'GPU': 'NVIDIA_L40S',
                        'Quantization': 'FP32',
                        'Compilation': compilation_mode,
                        'Latency_ms': avg_latency,
                        'Std_ms': std_latency
                    })
                    
                    print(f"      {compilation_mode}: {avg_latency:.3f} ± {std_latency:.3f} ms")
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"      OOM at batch size {batch_size}, stopping...")
                        break
                    else:
                        print(f"      Error: {e}")
                        continue
                except Exception as e:
                    print(f"      Unexpected error: {e}")
                    continue
        
        # Clean up after benchmarking this model
        print(f"  Cleaning up after {model_name}...")
        del model, tokenizer
        self.clear_cache()
        self.clear_model_cache(model_name)

    def save_results(self, filename=None):
        """Save benchmark results to CSV."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"cache_managed_transformer_benchmark_{timestamp}.csv"
        
        filepath = os.path.join('../results', filename)
        
        with open(filepath, 'w', newline='') as csvfile:
            fieldnames = ['Model', 'Batch_Size', 'GPU', 'Quantization', 'Compilation', 'Latency_ms', 'Std_ms']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.results)
        
        print(f"\nResults saved to: {filepath}")
        return filepath

def main():
    parser = argparse.ArgumentParser(description='Cache-Managed Transformer Benchmark')
    parser.add_argument('--models', nargs='+', default=[
        'bert-base-uncased', 'bert-large-uncased',
        'roberta-base', 'roberta-large',
        'distilbert-base-uncased',
        'gpt2', 'gpt2-medium',
        't5-small', 't5-base',
        'bart-base', 'bart-large',
        'deberta-base', 'deberta-v3-small',
        'electra-small', 'electra-base',
        'gcn', 'gat', 'sage',
        'vit'
    ], help='Models to benchmark')
    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[1, 2, 4, 8, 16, 32], 
                       help='Batch sizes to test')
    parser.add_argument('--output', type=str, help='Output CSV filename')
    parser.add_argument('--clear-all-cache', action='store_true', 
                       help='Clear all HuggingFace cache before starting')
    
    args = parser.parse_args()
    
    # Create results directory if it doesn't exist
    os.makedirs('../results', exist_ok=True)
    
    # Initialize benchmark
    benchmark = CacheManagedTransformerBenchmark()
    
    # Clear all cache if requested
    if args.clear_all_cache:
        print("Clearing all HuggingFace cache...")
        cache_dir = os.path.expanduser('~/.cache/huggingface/hub')
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            print("Cache cleared!")
        
        # Also clear any local model directories
        print("Clearing any local model directories...")
        for model_name in args.models:
            local_model_path = os.path.join(os.getcwd(), model_name)
            if os.path.exists(local_model_path):
                shutil.rmtree(local_model_path)
                print(f"Cleared local directory: {model_name}")
    
    # Run benchmarks
    for model_name in args.models:
        benchmark.benchmark_model(model_name, args.batch_sizes)
    
    # Save results
    output_file = benchmark.save_results(args.output)
    
    # Print summary
    print(f"\nBenchmark completed!")
    print(f"Tested {len(args.models)} models with batch sizes: {args.batch_sizes}")
    print(f"Total results: {len(benchmark.results)}")

if __name__ == "__main__":
    main() 