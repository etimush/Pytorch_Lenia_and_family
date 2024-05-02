from Lenia import Lenia, Lenia2, Lenia3
import torch
out_features, dt, k ,n , kpc = 4,1/1.2, 51,3,4
scale = 10/10000
for j in range(10000):
    nca = torch.jit.script(Lenia(out_features,dt,k,n,kpc).eval().cuda().requires_grad_(False))
    x= torch.zeros((1,out_features,100,100),dtype=torch.float16).cuda()
    random_inint = torch.rand((1,out_features,52,52),dtype=torch.float32).cuda()
    x[:,:,(x.shape[-2]//2) - (random_inint.shape[-2]//2):(x.shape[-2]//2) + (random_inint.shape[-2]//2),(x.shape[-1]//2) - (random_inint.shape[-1]//2):(x.shape[-1]//2) + (random_inint.shape[-1]//2)] = random_inint

    print(j)
    for i in range(1000):
        with torch.no_grad():
            x = nca(x)
            grid = x
            sum = x.sum()

        if sum == 0 or sum == torch.nan or sum > 14000:
            break
    if sum > 0 and sum > 0.001 and sum < 14000:
        string  = "lenia" + str(j)
        torch.save(nca.state_dict(), f"./random_search_save/"+string)



