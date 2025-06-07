import onnxruntime as rt
import numpy as np
import time

# Path to ONNX model
# model_path = '../resnet50.onnx'
model_path = '../alexnet.onnx'
# model_path = '../ssd.onnx'
# model_path = '../ssd-mobilenet.onnx'

# Create session options with profiling disabled
sess_options = rt.SessionOptions()
sess_options.enable_profiling = False

# Load the model and create a session with GPU support
session = rt.InferenceSession(model_path, sess_options=sess_options, providers=['CUDAExecutionProvider'])

# Input data (replace with your data)
if model_path in ('../resnet50.onnx', '../alexnet.onnx'):
    input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
elif model_path == '../ssd.onnx':
    input_data = np.random.randn(1, 3, 1200, 1200).astype(np.float32)
# elif model_path == 'ssd-mobilenet.onnx':
#     input_data = np.random.randn(1, 1200, 1200, 3).astype(np.float32)
input_data_ortvalue = rt.OrtValue.ortvalue_from_numpy(input_data, 'cuda', 0)

# I/O Binding
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

io_binding = session.io_binding()
io_binding.bind_input(name=input_name, device_type='cuda', device_id=0, element_type=np.float32, shape=input_data_ortvalue.shape(), buffer_ptr=input_data_ortvalue.data_ptr())
io_binding.bind_output(output_name, 'cuda')

# Warm up
for _ in range(1):
    # start_inf_time = time.time()
    results = session.run_with_iobinding(io_binding)
    # end_inf_time = time.time()
    # inf_time = end_inf_time - start_inf_time
    # print("Warm up Inference time in ms:", inf_time * 1000)

# Perform 1000 inferences
# iterations = 1000
# total_inference_time_seconds = 0.0

# for _ in range(iterations):
#     start_inf_time = time.time()
#     results = session.run_with_iobinding(io_binding)
#     end_inf_time = time.time()
#     inf_time = end_inf_time - start_inf_time
#     total_inference_time_seconds += inf_time

# Calculate average inference time
# inference_time_seconds_ms = total_inference_time_seconds / iterations * 1000
# print(f"Average Inference Time: {inference_time_seconds_ms} ms")
