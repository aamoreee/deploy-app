import torch

# Periksa apakah GPU tersedia
gpu_available = torch.cuda.is_available()
print(f"GPU Available: {gpu_available}")

# Jika GPU tersedia, tampilkan detailnya
if gpu_available:
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Built With CUDA: {torch.backends.cudnn.version()}")
