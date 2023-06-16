import numpy
import numpy as np
import torch

a = torch.tensor([[1, 2, 3],
              [4, 5, 6]])

# print(sum(a[-1]))
print(a.logsumexp(-1))

a1 = torch.tensor([1, 2, 3])
a2 = torch.tensor([4, 5, 6])

print(a1.logsumexp(0))
print(a2.logsumexp(0))