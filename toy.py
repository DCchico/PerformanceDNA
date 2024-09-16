import torch
import torch.nn as nn
from torchinfo import summary
import torchvision.models as models


# Load ResNet-50 model
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

# Display the model summary with detailed info using torchinfo
model_summary = summary(model, input_size=(1, 3, 224, 224), verbose=2)

def calculate_sgemm_macs(conv_layer_info):
    """Calculate MACs for SGEMM for each convolution layer"""
    total_macs = 0
    
    # Parse through each convolution layer's info
    for layer in conv_layer_info:
        conv_module = layer.module
        
        if isinstance(conv_module, nn.Conv2d):
            # Get the input/output shapes directly from torchinfo
            in_channels = conv_module.in_channels  # Input channels
            out_channels = conv_module.out_channels  # Output channels
            input_h, input_w = layer.input_size[2:]  # Input height and width
            output_h, output_w = layer.output_size[2:]  # Output height and width
            kernel_h, kernel_w = conv_module.kernel_size  # Kernel height and width
            
            # MACs calculation for SGEMM
            macs = in_channels * kernel_h * kernel_w * out_channels * output_h * output_w
            total_macs += macs
            
            # Display layer info and MACs
            print(f"Layer: {conv_module} - Input: {layer.input_size}, Output: {layer.output_size}, MACs = {macs}")
    
    print(f"Total SGEMM MACs: {total_macs}")

# Extract convolution layer details from the model summary
conv_layers_info = [
    layer for layer in model_summary.summary_list
    if isinstance(layer.module, nn.Conv2d)
]

# Calculate MACs for SGEMM
calculate_sgemm_macs(conv_layers_info)
