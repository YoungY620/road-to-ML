import torch 

t = torch.arange(0, 100)
print(t)
indexes = (torch.rand(size=(5,))*100).type(torch.LongTensor)
print(t[indexes])