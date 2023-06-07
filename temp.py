import torch

a = torch.Tensor([[[1,2,3],[1,2,3]],
                  [[1,2,3],[1,2,3]]])

a = torch.cat(a,0)

print(a)