for model in resnet50 densenet121 efficientnet_b0 mobilenet_v2; do
  for mode in eager torchscript_trace torchscript_script inductor tensorrt; do
    echo "==== $model | $mode ===="
    python cnns_latency_benchmark.py --model $model --mode $mode --batch_sizes 1 2 4 8 16 32 --iterations 100
  done
done