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

# Load a pre-trained ResNet-50 model using the updated API
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
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
    output_file = f"ncu_resnet50_{profiling_sections}"

    if profiling_sections == 'default':
        ncu_command = f'PROFILE_MEASURE=1 ncu --target-processes all -o {output_file} python3 torch_resnet50_inference.py'
    elif profiling_sections == 'compute':
        ncu_command = f'PROFILE_MEASURE=1 ncu --target-processes all --section ComputeWorkloadAnalysis -o {output_file} python3 torch_resnet50_inference.py'
    elif profiling_sections == 'memory':
        ncu_command = f'PROFILE_MEASURE=1 ncu --target-processes all --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Chart -o {output_file} python3 torch_resnet50_inference.py'
    elif profiling_sections == 'scheduler':
        ncu_command = f'PROFILE_MEASURE=1 ncu --target-processes all --section SchedulerStats --section WarpStateStats -o {output_file} python3 torch_resnet50_inference.py'
    elif profiling_sections == 'roofline':
        ncu_command = f'PROFILE_MEASURE=1 ncu --target-processes all --section SpeedOfLight_RooflineChart -o {output_file} python3 torch_resnet50_inference.py'
    elif profiling_sections == 'all':
        ncu_command = f'PROFILE_MEASURE=1 ncu --target-processes all --set full -o {output_file} python3 torch_resnet50_inference.py'
    elif profiling_sections == 'separate_all':
        ncu_commands = [
            f'PROFILE_MEASURE=1 ncu --target-processes all -o ncu_resnet50_default python3 torch_resnet50_inference.py',
            f'PROFILE_MEASURE=1 ncu --target-processes all --section ComputeWorkloadAnalysis -o ncu_resnet50_compute python3 torch_resnet50_inference.py',
            f'PROFILE_MEASURE=1 ncu --target-processes all --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Chart -o ncu_resnet50_memory python3 torch_resnet50_inference.py',
            f'PROFILE_MEASURE=1 ncu --target-processes all --section SchedulerStats --section WarpStateStats -o ncu_resnet50_scheduler python3 torch_resnet50_inference.py',
            f'PROFILE_MEASURE=1 ncu --target-processes all --section SpeedOfLight_RooflineChart -o ncu_resnet50_roofline python3 torch_resnet50_inference.py',
        ]
        for cmd in ncu_commands:
            os.system(cmd)
        exit(0)  # Exit after running all commands
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
