import torch
import torch.nn as nn
import sys
import os

# Add quantized repo to path
quantized_repo_path = os.path.abspath('/usr/scratch/difei/quantized.pytorch')
if quantized_repo_path:
    sys.path.insert(0, quantized_repo_path)

try:
    import models
    from models.modules.quantize import QuantMeasure, QConv2d, QLinear, RangeBN
    QUANTIZED_MODELS_AVAILABLE = True
except ImportError as e:
    print(f"Quantized models not available: {e}")
    QUANTIZED_MODELS_AVAILABLE = False

def detect_fp32_fallback(model, dummy_input):
    """Detect which operations fall back to FP32 in a quantized model"""
    print(f"\n{'='*60}")
    print("DETECTING FP32 FALLBACK OPERATIONS")
    print(f"{'='*60}")
    
    # Hook to track tensor dtypes
    dtype_tracker = {}
    
    def hook_fn(module, input, output):
        module_name = module.__class__.__name__
        
        # Track input dtypes
        if module_name not in dtype_tracker:
            dtype_tracker[module_name] = {'inputs': [], 'outputs': []}
        
        input_dtypes = [t.dtype for t in input if isinstance(t, torch.Tensor)]
        output_dtypes = [t.dtype for t in (output if isinstance(output, tuple) else [output]) if isinstance(t, torch.Tensor)]
        
        dtype_tracker[module_name]['inputs'].append(input_dtypes)
        dtype_tracker[module_name]['outputs'].append(output_dtypes)
    
    # Register hooks on all modules
    hooks = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Only leaf modules
            hook = module.register_forward_hook(hook_fn)
            hooks.append(hook)
    
    # Run inference
    model.eval()
    with torch.no_grad():
        _ = model(dummy_input)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Analyze results
    print(f"{'Module Type':<20} {'Input Dtypes':<30} {'Output Dtypes':<30} {'FP32 Fallback':<15}")
    print("-" * 95)
    
    fp32_fallbacks = []
    
    for module_name, dtypes in dtype_tracker.items():
        if not dtypes['inputs'] or not dtypes['outputs']:
            continue
            
        # Get most common dtypes
        input_dtypes = dtypes['inputs'][0]  # First forward pass
        output_dtypes = dtypes['outputs'][0]
        
        input_str = str(input_dtypes) if input_dtypes else "None"
        output_str = str(output_dtypes) if output_dtypes else "None"
        
        # Check for FP32 fallback
        has_fp32_fallback = False
        if input_dtypes:
            if torch.float32 in input_dtypes and torch.qint8 not in input_dtypes:
                has_fp32_fallback = True
        if output_dtypes:
            if torch.float32 in output_dtypes and torch.qint8 not in output_dtypes:
                has_fp32_fallback = True
        
        if has_fp32_fallback:
            fp32_fallbacks.append(module_name)
        
        fallback_str = "YES" if has_fp32_fallback else "NO"
        print(f"{module_name:<20} {input_str:<30} {output_str:<30} {fallback_str:<15}")
    
    print("-" * 95)
    print(f"Total modules with FP32 fallback: {len(fp32_fallbacks)}")
    if fp32_fallbacks:
        print("Modules with FP32 fallback:")
        for module in fp32_fallbacks:
            print(f"  - {module}")
    
    return fp32_fallbacks

def analyze_quantized_model(model_name, model_config):
    """Analyze a specific quantized model"""
    if not QUANTIZED_MODELS_AVAILABLE:
        print("Quantized models not available")
        return
    
    print(f"\nAnalyzing model: {model_name}")
    
    # Load model
    model_constructor = models.__dict__[model_name]
    model = model_constructor(**model_config)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Detect FP32 fallbacks
    fp32_fallbacks = detect_fp32_fallback(model, dummy_input)
    
    # Additional analysis
    print(f"\n{'='*60}")
    print("ADDITIONAL ANALYSIS")
    print(f"{'='*60}")
    
    # Count different module types
    module_counts = {}
    for module in model.modules():
        module_type = module.__class__.__name__
        module_counts[module_type] = module_counts.get(module_type, 0) + 1
    
    print("Module type distribution:")
    for module_type, count in sorted(module_counts.items()):
        print(f"  {module_type}: {count}")
    
    # Check for quantization layers
    quant_layers = 0
    for module in model.modules():
        if isinstance(module, (QuantMeasure, QConv2d, QLinear, RangeBN)):
            quant_layers += 1
    
    print(f"\nQuantization-aware layers: {quant_layers}")
    print(f"Total layers: {sum(module_counts.values())}")
    print(f"Quantization coverage: {quant_layers/sum(module_counts.values())*100:.1f}%")

def main():
    if not QUANTIZED_MODELS_AVAILABLE:
        print("Quantized models not available")
        return
    
    # Test with a quantized model
    model_config = {"depth": 18, "dataset": "imagenet"}
    analyze_quantized_model("resnet_quantized", model_config)

if __name__ == "__main__":
    main() 