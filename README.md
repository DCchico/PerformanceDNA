# ProfilingDNAResutls

## Profiling PyTorch Models with Nsight Compute in Docker

### 0. Find Server CUDA and cuDNN Version

To find the CUDA and cuDNN versions on your server, use the following commands:

```sh
nvcc --version
nvidia-smi
python3 -c "import torch; print(torch.backends.cudnn.version())"
```

### 1. Pull 22.10 PyTorch Container from NVIDIA NGC
```sh
docker pull nvcr.io/nvidia/pytorch:22.10-py3
```

### 2. Run Container with Mounted Directory
```sh
docker run --gpus all -it --rm --shm-size=16g -v ~/difei/profiler/ncu:/workspace nvcr.io/nvidia/pytorch:22.10-py3
```
### 3. Profiling with Nsight Compute
```sh
python3 torch_inference.py
PROFILE_SECTIONS=compute python3 torch_inference.py
PROFILE_SECTIONS=memory python3 torch_inference.py
PROFILE_SECTIONS=scheduler python3 torch_inference.py
PROFILE_SECTIONS=roofline python3 torch_inference.py
```

### Additional Steps
Custom Output File Names:
File names needs to be adjusted when changing flags in torch_inference.py

Install Nsight Compute Locally and transfer results to host for GUI:
If needed, download and install Nsight Compute/System on your local machine to analyze the profiling results.
