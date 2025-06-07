import torch
import torchvision.models as models
import numpy as np
import time
import os

flag_inspect = False
flag_cudnn = True
flag_profile = os.getenv('PROFILE_MEASURE', '0') == '1'
profiling_sections = os.getenv('PROFILE_SECTIONS', 'default')

if flag_inspect:
    print(torch.__version__)
    print(torch.backends.cudnn.version())
    print(torch.cuda.is_available())
    print(torch.cuda.current_device())
    print(torch.cuda.get_device_name(torch.cuda.current_device()))

# Check if CUDA is available
if not torch.cuda.is_available():
    exit(1)
device = torch.device("cuda")

# Disable cuDNN
torch.backends.cudnn.enabled = flag_cudnn

# Load a pre-trained model using the updated API
model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
model.to(device)
model.eval()

# Create a random input tensor
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
input_tensor = torch.tensor(input_data).to(device)

# Warm-up iterations
if not flag_profile:
    print("Running warm-up iterations...")
    for _ in range(10):
        with torch.no_grad():
            output = model(input_tensor)
    print("Warm-up completed.")
    print("Running measurement iterations...")
    times = []
    for _ in range(100):
        start = time.perf_counter()
        with torch.no_grad():
            output = model(input_tensor)
        end = time.perf_counter()
        times.append(end - start)

    avg_time = sum(times) / len(times)
    print(f"Average inference time: {avg_time * 1000:.2f} ms")
    print("Starting Profiling...")

    # Construct Nsight Compute command
    output_file = f"ncu_alexnet_{profiling_sections}"
    ncu_command = f'PROFILE_MEASURE=1 ncu --target-processes all -o {output_file}'

    if profiling_sections == 'default':
        ncu_command += ' python3 torch_inference.py'
    elif profiling_sections == 'compute':
        ncu_command += ' --section ComputeWorkloadAnalysis python3 torch_inference.py'
    elif profiling_sections == 'memory':
        ncu_command += ' --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Chart python3 torch_inference.py'
    elif profiling_sections == 'scheduler':
        ncu_command += ' --section SchedulerStats --section WarpStateStats python3 torch_inference.py'
    elif profiling_sections == 'roofline':
        ncu_command += ' --section SpeedOfLight_RooflineChart python3 torch_inference.py'
    else:
        print(f"Unknown profiling section: {profiling_sections}")
        exit(1)

    # Re-run the script with profiling enabled
    os.system(ncu_command)
else:
    print("Running profiling iteration...")
    with torch.no_grad():
        output = model(input_tensor)
    print("Profiling completed.")
