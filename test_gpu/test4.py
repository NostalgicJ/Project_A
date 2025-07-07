import torch
print(torch.backends.mps.is_available())  # MPS 사용 가능 여부
print(torch.backends.mps.is_built())      # MPS 지원 여부