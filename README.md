# PerformanceDNA Benchmark and Analysis

## 0. Find Server CUDA and cuDNN Version

To find the CUDA and cuDNN versions on your server, use the following commands:

```sh
nvcc --version
nvidia-smi
python3 -c "import torch; print(torch.backends.cudnn.version())"
```

### Profiling Models in PyTorch Docker for Inference Latency (with Nsight Compute for Detailed Metrics)

#### 1. Pull Compatible PyTorch Container [See Compatibility Matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html) from NVIDIA NGC. For example:
```sh
docker pull nvcr.io/nvidia/pytorch:22.10-py3
```

### 2. Run Container with Mounted Directory
```sh
docker run --gpus all -it --rm --shm-size=16g -v <Path-to-local-workspace>:/workspace nvcr.io/nvidia/pytorch:22.10-py3
```

### 3. Benchmark Latency
```sh
python3 latency_benchmark/<script_name>
```

### 4. (Optional) Profiling with Nsight Compute to Get Hardware Metrics For Analysis
```sh
python3 torch_inference.py
PROFILE_SECTIONS=compute python3 torch_inference.py
PROFILE_SECTIONS=memory python3 torch_inference.py
PROFILE_SECTIONS=scheduler python3 torch_inference.py
PROFILE_SECTIONS=roofline python3 torch_inference.py
```

### Optional Steps for Visualizing Nsight Compute Results
Custom Output File Names:
File names needs to be adjusted when changing flags in torch_inference.py

Install Nsight Compute Locally and transfer results to host for GUI:
If needed, download and install Nsight Compute/System on your local machine to analyze the profiling results.
