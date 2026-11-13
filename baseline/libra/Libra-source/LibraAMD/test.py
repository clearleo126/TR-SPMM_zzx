import torch
print("torch:", torch.__version__)
print("hip:", torch.version.hip)
print("cuda:", torch.version.cuda)
print("is_available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)