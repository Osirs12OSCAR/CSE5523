import torch

print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device Name:", torch.cuda.get_device_name(0))
    print("CUDA Version:", torch.version.cuda)
    print("PyTorch Version:", torch.__version__)
else:
    print("CUDA is not available. Check your installation.")