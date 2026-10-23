import torch

print("Wersja PyTorch:", torch.__version__)

x = torch.rand(3, 3)
print(x)

print("CUDA dostępna:", torch.cuda.is_available())