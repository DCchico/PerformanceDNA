import torch
import torchvision.models as models
import numpy as np

print(torch.__version__)
print(torch.backends.cudnn.version())
print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(torch.cuda.current_device()))

# Check if CUDA is available
if not torch.cuda.is_available():
    exit(1)
device = torch.device("cuda")

# Load a pre-trained model using the updated API
model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
model.to(device)
model.eval()

# Create a random input tensor
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
input_tensor = torch.tensor(input_data).to(device)

# Run inference
with torch.no_grad():
    output = model(input_tensor)

print("Inference completed")
