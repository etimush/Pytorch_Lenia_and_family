import torch

x = torch.ones((10,10,2), device="cuda:0", dtype=torch.float32)
clip = torch.rand_like(x, device="cuda:0")[:,:,0:1]
x = x.clip(torch.zeros_like(clip, device="cuda:0"), clip)

print(x, clip
      )