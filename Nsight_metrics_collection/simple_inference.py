import onnxruntime as rt
import numpy as np

# Simple model (ensure it is available)
model_path = '../mnist.onnx'

sess_options = rt.SessionOptions()
sess_options.enable_profiling = False
session = rt.InferenceSession(model_path, sess_options=sess_options, providers=['CUDAExecutionProvider'])

# input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
input_data = np.random.randn(1, 1, 28, 28).astype(np.float32)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Run inference
results = session.run([output_name], {input_name: input_data})
print("Inference completed")
