# PerformanceDNA Benchmark and Analysis

This repository supports the research project on **PerformanceDNA**, which aims to uncover how specific characteristics of deep learning (DL) models influence the effectiveness of graph-based compilation strategies. The benchmark suite and analysis tools provided here help identify when and why certain models underperform with specific compilation configurations, such as eager execution, TorchDynamo, or trace-based graph instantiation.

The repo includes scripts for:
- Verifying the execution environment and its capabilities.
- Running inference latency benchmarks across multiple compilation setups.
- Extracting and inspecting `torch.compile` (Dynamo) FX graphs.
- Profiling hardware behavior using NVIDIA Nsight Compute.
- Analyzing performance bottlenecks tied to model structure ("model DNA") (TODO)

Use the steps below to set up your environment and start benchmarking.

## Environment Setup

You can run the benchmarks using either a pre-configured Docker container or a manually configured bare-metal environment.

### Option 1: Docker Container (Recommended for Quick Start)

This method uses a container with all system dependencies pre-installed.

#### 1. Find Host CUDA Version
First, check the NVIDIA driver version on the host machine to choose a compatible container.
```sh
nvidia-smi
```

#### 2. Pull Compatible PyTorch Container
Find a compatible PyTorch container from the [NVIDIA NGC Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) based on your driver version. For example:
```sh
docker pull nvcr.io/nvidia/pytorch:24.05-py3
```

#### 3. Run Container with Mounted Directory
```sh
docker run --gpus all -it --rm --shm-size=16g -v .:/workspace nvcr.io/nvidia/pytorch:24.05-py3
```

### Option 2: Bare-Metal Setup (For Custom Environments)

This method provides more control. It involves creating a local Python virtual environment. For quick reproduction, please contact repository maintainer to request for scheduling server access with this setup.

**Note:** This setup requires a specific Python version with development headers. 

#### 1. Activate the Virtual Environment
A Python 3.10+ virtual environment named `pytorch_env_310` is available in the project home. Activate it using:
```sh
source pytorch_env_310/bin/activate
```

#### 2. Verify Platform Readiness
Before running benchmarks, run the verification script to ensure all dependencies, compilers, and hardware are correctly configured.
```sh
python setup_scripts/verify_platform.py
```
If any checks fail, address the issues before proceeding.

## Running the Benchmarks

Once your environment is active (either inside Docker or the local venv):

### 1. Benchmark Inference Latency
The primary script for benchmarking is `generic_benchmark.py`. You can specify the model and batch sizes.

```sh
# Run ResNet50 with default batch sizes (1, 2, 4, 8, 16, 32)
python latency_benchmark/generic_benchmark.py --model resnet50

# Run BERT with custom batch sizes
python latency_benchmark/generic_benchmark.py --model bert-base-uncased --batch_sizes 1 4 8

# Available models: resnet50, bert-base-uncased, vit, align, lstm
```
Results are appended to `results/benchmark_results.csv`.

### 2. Extract Dynamo FX Graph
To understand what `torch.compile` is doing, you can extract its internal graph representation.

```sh
python latency_benchmark/dynamo_fxGraph_extraction.py --model resnet50
```
The output is saved to `results/resnet50_dynamo_graph.txt`.

### 3. (Optional) Profiling with Nsight Compute
For deep hardware analysis, use `ncu` to profile the execution. (Further details to be added).

```sh
# Example of profiling with Nsight Compute
ncu -o my_profile_report --set full python latency_benchmark/generic_benchmark.py --model resnet50 --batch_sizes 1
```
```