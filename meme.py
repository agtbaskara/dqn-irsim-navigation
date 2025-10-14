import torch

# Check if CUDA is available
print(f"CUDA available: {torch.cuda.is_available()}")

# Check CUDA version PyTorch was built with
print(f"CUDA version: {torch.version.cuda}")

# Check PyTorch version
print(f"PyTorch version: {torch.__version__}")

# If CUDA is available, check device name
if torch.cuda.is_available():
    print(f"Current CUDA device: {torch.cuda.get_device_name(0)}")