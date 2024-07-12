import torch
import torchvision.models as models
import numpy as np
import time
import os

flag_inspect = False
flag_cudnn = False
flag_profile = os.getenv('PROFILE_MEASURE', '0') == '1'

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
    # Re-run the script with profiling enabled
    os.system('PROFILE_MEASURE=1 ncu --target-processes all -o ncu_alexnet_no_cudnn python3 torch_inference.py')
else:
    print("Running profiling iterations...")
    with torch.no_grad():
        output = model(input_tensor)
    print("Profiling completed.")

