import torch

print(torch.cuda.is_available())  # Should print True if CUDA is enabled
print(torch.backends.cudnn.version())  # Should print the cuDNN version
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))