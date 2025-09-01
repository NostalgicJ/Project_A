import torch

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Example tensor creation
x = torch.randn(1, device=device)
print(x)
