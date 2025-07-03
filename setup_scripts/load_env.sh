source pytorch_env_310/bin/activate
export PATH="/usr/scratch/difei/PerformanceDNA/pytorch_env_310/bin:$PATH"
python --version

python -c "import torch; print(f\"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}\")"

python verify_platform.py
