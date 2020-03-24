import numpy as np 
import torch
import torchvision

x = torch.randn(2, 3, 1)
print(x)
print(x.sum(0)) # 变为3x1
print(x.sum(1)) # 变为2x1
print(x.sum(2)) # 变为2x3
print(x.sum())  # 总和
print(torch.sum(x))   # 总和
print(torch.sum(x, 0))# 按第0维求和
print(torch.sum(x, 1))# 按第1维求和
print(torch.sum(x, 2))# 按第2维求和