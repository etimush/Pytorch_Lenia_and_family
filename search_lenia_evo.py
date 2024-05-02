import random

from Lenia import Lenia, Lenia2, Lenia3
import torch
from evol_utils import mutate, reproduce
out_features, dt, k ,n , kpc = 4,1/1.2, 51,3,4

models = ["evo_models/Week_1_m1","evo_models/Week_1_m3","evo_models/Week_1_m4", "evo_models/lenia25", "evo_models/lenia245"]
ncas = []
for model in models:
    nca =  torch.jit.script(Lenia(out_features,dt,k,n,kpc).cuda())
    nca.load_state_dict(torch.load(model))
    ncas.append(nca)



for j in range(10000):
    parents = random.choices(ncas, k=2)

    nca = reproduce(parents[0], parents[1])
    nca = mutate(nca)
    x = torch.zeros((1, out_features, 100, 100), dtype=torch.float16, device="cuda:0")
    random_inint = torch.rand((1,out_features,52,52),dtype=torch.float16, device="cuda:0")
    x[:,:,(x.shape[-2]//2) - (random_inint.shape[-2]//2):(x.shape[-2]//2) + (random_inint.shape[-2]//2),(x.shape[-1]//2) - (random_inint.shape[-1]//2):(x.shape[-1]//2) + (random_inint.shape[-1]//2)] = random_inint

    print(j)
    for i in range(1000):
        with torch.no_grad():
            x = nca(x)
        grid = nca.grid

        if grid.sum() == 0 or grid.sum() == torch.nan or grid.sum() > 14000:
            break
    if grid.sum() > 0 and grid.mean() > 0.001 and grid.sum() < 14000:
        string  = "lenia" + str(j+100)
        torch.save(nca.state_dict(), f"./evo_search_models/"+string)



