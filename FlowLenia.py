import torch
import numpy as np
import utils


class Lenia_Classic(torch.nn.Module):
    def __init__(self, C, dt , K , kernels, device, X, Y, mode = "soft", has_food = None):
        super(Lenia_Classic, self).__init__()

        self.dt = dt
        self.C = C
        self.kernels = kernels
        self.device = device
        self.midX = X//2
        self.midY = Y//2
        self.bell = lambda x, m, s: torch.exp(-((x - m) / s) ** 2 / 2)
        self.mode = mode
        Ds = [torch.tensor(np.linalg.norm(np.asarray(np.ogrid[-self.midX:self.midX, -self.midY:self.midY], dtype=object)) / K*len(k["b"])/k["r"], device=device, dtype=torch.float32) for k in self.kernels]
        self.Ks = [(D<len(k["b"])) *  torch.tensor(k["b"],device=device)[torch.minimum(D.int(),torch.tensor(len(k["b"]))-1)] * self.bell(D % 1, 0.5, 0.15) for D,k in zip(Ds,self.kernels)]
        self.fkernels = [torch.fft.fft2(torch.fft.fftshift(k/k.sum())) for k in self.Ks]

    def growth(self,U,m,s):
        return self.bell(U,m,s)*2 -1

    def soft_clip(self,x):
        return 1 / (1 + torch.exp(-4 * (x - 0.5)))

    def forward(self, x):
        fXs = [torch.fft.fft2(x[:,:, i]) for i in range(x.shape[-1])]
        Us = [torch.fft.ifft2(fkernel*fXs[k["c0"]]).real for fkernel, k in zip(self.fkernels, self.kernels)]
        Gs = [self.growth(U,k["m"],k["s"]) for U,k in zip(Us,self.kernels)]
        Hs = torch.dstack([sum(k["h"]*G if k["c1"] == c1 else torch.zeros_like(G, device=self.device) for G,k in zip(Gs, self.kernels)  ) for c1 in range(self.C)])
        if self.mode == "soft":
            x = self.soft_clip(x + self.dt*Hs)
        if self.mode =="hard":
            x = torch.clip(x + self.dt*Hs, 0, 1)
        return x


class Lenia_Diff(torch.nn.Module):
    def __init__(self, C, dt , K, kernels, device, X, Y, mode = "soft", has_food = None):
        super(Lenia_Diff, self).__init__()

        self.dt = torch.tensor(dt, device=device)
        self.C = torch.tensor(C, device=device)
        self.kernels = kernels

        self.device = device
        self.midX = X//2
        self.midY = Y//2
        self.bell = lambda x, m, s: torch.exp(-((x - m) / s) ** 2 / 2)
        self.bell_kernel = lambda x, a, w, r, R: torch.exp(-((x / (r * R)) - a) ** 2 / (2*w**2))
        self.mode = mode
        Ds = [torch.tensor(np.linalg.norm(np.asarray(np.ogrid[-self.midX:self.midX, -self.midY:self.midY], dtype=object)) / K*len(k["b"])/k["r"], device=device, dtype=torch.float32) for k in self.kernels]
        self.Ks = [(D<len(k["b"])) *  torch.tensor(k["b"],device=device)[torch.minimum(D.int(),torch.tensor(len(k["b"]))-1)] * self.bell_kernel(D % 1, torch.tensor(k["a"],device=device)[torch.minimum(D.int(),torch.tensor(len(k["a"]))-1)], torch.tensor(k["w"],device=device)[torch.minimum(D.int(),torch.tensor(len(k["w"]))-1)],k["r"],K) for D,k in zip(Ds,self.kernels)]
        self.fkernels = [torch.fft.fft2(torch.fft.fftshift(k/k.sum())) for k in self.Ks]

    def growth(self,U,m,s):
        return self.bell(U,m,s)*2 -1

    def soft_clip(self,x):
        return 1 / (1 + torch.exp(-4 * (x - 0.5)))

    @torch.no_grad()
    def forward(self, x):
        fXs = [torch.fft.fft2(x[:,:,i]) for i in range(x.shape[-1])]
        Us = [torch.fft.ifft2(fkernel*fXs[k["c0"]]).real for fkernel, k in zip(self.fkernels, self.kernels)]
        Gs = [self.growth(U,k["m"],k["s"]) for U,k in zip(Us,self.kernels)]
        Hs = torch.dstack([sum(k["h"]*G if k["c1"] == c1 else torch.zeros_like(G, device=self.device) for G,k in zip(Gs, self.kernels)  ) for c1 in range(self.C)])
        if self.mode == "soft":
            x = self.soft_clip(x + self.dt*Hs)
        if self.mode =="hard":
            x = torch.clip(x + self.dt*Hs, 0, 1)
        return x

def construct_mesh_grid(X,Y):
    x, y = torch.arange(X), torch.arange(Y)
    mx, my = torch.meshgrid(x, y)
    pos = torch.dstack((mx, my)) + .5
    #pos = pos.permute((2,1,0))
    return pos.to("cuda:0")

def construct_ds(dd):
    dxs = []
    dys = []
    for dx in range(-dd, dd+1):
        for dy in range(-dd, dd+1):
            dxs.append(dx)
            dys.append(dy)
    dxs = torch.tensor(dxs, device="cuda:0")
    dys = torch.tensor(dys, device="cuda:0")
    return dxs, dys

class ReintegrationTracker():
    def __init__(self, X, Y, dt,dd=2, sigma=0.95):
        self.X = X
        self.Y = Y
        self.dd = dd
        self.dt = dt
        self.sigma = sigma
        self.pos = construct_mesh_grid(X,Y)
        self.dxs, self.dys = construct_ds(dd)


    def step(self, grid, mu, dx, dy):

        gridR = torch.roll(grid, (dx,dy), (0,1))
        muR = torch.roll(mu, (dx,dy), (0,1))
        #dpmu = (torch.stack([torch.abs(self.pos[...,None] - (muR + torch.tensor([di, dj],device="cuda:0")[None,None,:,None])) for di in (-self.X,0, self.X) for dj in (-self.Y,0, self.Y) ])).min(dim=0)[0]
        #dpmu = (torch.stack([torch.abs(self.pos[...,None] - (muR + torch.tensor([di, dj],device="cuda:0")[None,None,:,None])) for di in (-self.X,0, self.X) for dj in (-self.Y,0, self.Y) ])).min(dim=0)[0]
        dpmu = (self.pos[...,None]-muR).abs()
        sz = (.5 - dpmu + self.sigma)
        area = torch.prod(torch.clip(sz,0,min(1,2*self.sigma)), dim= 2) / (4*self.sigma**2)
        ngrid = gridR * area

        return ngrid
    def apply(self, grid, F):

        ma = self.dd - self.sigma
        mu = self.pos[..., None] + (self.dt*F).clip(-ma,ma)
        mu = torch.clip(mu, self.sigma, self.X - self.sigma)
        ngrid = torch.stack([self.step(grid, mu, dx, dy) for dx, dy in zip(self.dxs, self.dys)])



        return ngrid.sum(dim=0)







def sobel_x(x):

    k_x = (torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]],
    dtype=torch.float32, device = "cuda:0")
           .tile((1, 1, 1,1)))

    sx =torch.vstack([torch.nn.functional.conv2d(x[None,:,:,c], k_x, groups= 1, stride=1, padding="same") for c in range(x.shape[-1])])
    return sx.permute((1,2,0))

def sobel_y(x):
    k_y = torch.tensor([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]],
                       dtype=torch.float32, device="cuda:0").T.tile((1, 1, 1, 1))
    sy =torch.vstack([torch.nn.functional.conv2d(x[None,:,:,c], k_y, groups= 1, stride=1, padding="same") for c in range(x.shape[-1])])
    return sy.permute((1,2,0))

def sobel(x):
    sx = sobel_x(x.to(torch.float32))
    sy = sobel_y(x.to(torch.float32))
    sxy = torch.cat((sy[:,:,None,:], sx[:,:,None,:]), dim= 2)
    return sxy

class Lenia_Flow(torch.nn.Module):
    def __init__(self, C, dt , K , kernels, device, X, Y, has_food,mode = "soft"):
        super(Lenia_Flow, self).__init__()
        self.rt = ReintegrationTracker(X,Y,dt)
        self.has_food = has_food
        self.dt = dt
        self.C = C
        self.n = 2.
        self.theta_x = 2
        self.kernels = kernels
        self.device = device
        self.midX = X//2
        self.midY = Y//2
        self.bell = lambda x, m, s: torch.exp(-((x - m) / s) ** 2 / 2)
        self.bell_kernel = lambda x, a, w, b: (b* torch.exp(-(x[..., None]  - a) ** 2 / w)).sum(-1)
        self.mode = mode
        self.c0 = None
        self.c1  =None
        self.m = None
        self.s =  None
        self.h = None
        self.pk = None
        self.fkernels = self.construct_kernels(K,device)



    def construct_kernels(self, K, device):
        if self.pk == "sparse":
            Ds = [torch.tensor(
                np.linalg.norm(np.mgrid[-self.midX:self.midX, -self.midY:self.midY], axis=0) / K*len(k["b"])/k["r"],
                device=self.device, dtype=torch.float32) for k in self.kernels]
        else:
            Ds = [torch.tensor(
                np.linalg.norm(np.mgrid[-self.midX:self.midX, -self.midY:self.midY], axis=0) / ((K+15)*k["r"]),
                device=self.device, dtype=torch.float32) for k in self.kernels]

        Ks = torch.dstack([self.sigmoid(-(D - 1) * 10) * self.bell_kernel(D, torch.tensor(k["a"], device=device),
                                                                              torch.tensor(k["w"], device=device),
                                                                              torch.tensor(k["b"], device=device)) for
                               D, k in zip(Ds, self.kernels)])
        fkernels = torch.fft.fft2(torch.fft.fftshift(Ks / Ks.sum(dim=(0, 1), keepdims=True), dim=(0, 1)),
                                       dim=(0, 1))
        return fkernels
    def growth(self,U,m,s):
        return self.bell(U,m,s)*2 -1

    def soft_clip(self,x):
        return 1 / (1 + torch.exp(-4 * (x - 0.5)))

    def sigmoid(self,x):
        return 0.5 * (torch.tanh(x / 2) + 1)

    def forward(self, x):


        fXs = torch.fft.fft2(x, dim= (0,1))

        if self.c1 != None:
            fXk = fXs[ :, :, self.c0]
        else:
            fXk = torch.dstack([fXs[:, :, k["c0"]] for k in self.kernels])

        Us = torch.fft.ifft2(self.fkernels*fXk, dim=(0,1)).real
        Gs = self.growth(Us,self.m,self.s) *self.h
        if self.c1 != None :
            Hs = torch.dstack([ Gs[:, :, self.c1[c]].sum(dim=-1) for c in range(self.C- self.has_food) ])

        else:
            Hs = torch.dstack([sum(k["h"] * Gs[:,:,i] if k["c1"] == c1 else torch.zeros_like(Gs[:,:,i], device=self.device) for i, k in zip(range(Gs.shape[-1]), self.kernels)) for c1 in range(self.C)])


        grad_u = sobel(Hs)  # (c,2,y,x)

        grad_x = sobel(x[:,:,self.has_food:].sum(dim=-1, keepdims=True))   # (1,2,y,x)

        alpha = (((x[:,:,None, self.has_food:] / self.theta_x) ** self.n)).clip(0,1)

        F = grad_u * (1 - alpha) - grad_x * alpha
        if self.has_food:
            x_overlap = ((x[:,:,-1][...,None]-0.1 )* .9).clip(torch.zeros_like(x[:,:,0:1]), x[:,:,0:1])
            food= x[:,:,0:1] - x_overlap
            x = self.rt.apply(x[:,:,self.has_food:], F)+ torch.cat([x_overlap/self.C for _ in range(self.C-1)], dim=-1) - (x[:,:,self.has_food:]*.0005)/(self.C-1)
            x = torch.cat((food, x), dim= -1)
        else :
            x = self.rt.apply(x, F)

        return x


class Lenia_Flow_Param(torch.nn.Module):
    def __init__(self, C, dt , K , kernels, device, X, Y, mode = "soft"):
        super(Lenia_Flow_Param, self).__init__()
        self.rt = ReintegrationTrackerParam(X,Y,dt)
        self.dt = dt
        self.C = C
        self.n = 2.
        self.theta_x = 2
        self.kernels = kernels
        self.device = device
        self.midX = X//2
        self.midY = Y//2
        self.bell = lambda x, m, s: torch.exp(-((x - m) / s) ** 2 / 2)
        self.bell_kernel = lambda x, a, w, b: (b* torch.exp(-(x[..., None]  - a) ** 2 / w)).sum(-1)
        self.mode = mode
        self.c0 = None
        self.c1  =None
        self.m = None
        self.s =  None
        self.h = None
        self.pk = None
        self.fkernels = self.construct_kernels(K,device)



    def construct_kernels(self, K, device):
        if self.pk == "sparse":
            Ds = [torch.tensor(
                np.linalg.norm(np.mgrid[-self.midX:self.midX, -self.midY:self.midY], axis=0) / K*len(k["b"])/k["r"],
                device=self.device, dtype=torch.float32) for k in self.kernels]
        else:
            Ds = [torch.tensor(
                np.linalg.norm(np.mgrid[-self.midX:self.midX, -self.midY:self.midY], axis=0) / ((K+15)*k["r"]),
                device=self.device, dtype=torch.float32) for k in self.kernels]

        Ks = torch.dstack([self.sigmoid(-(D - 1) * 10) * self.bell_kernel(D, torch.tensor(k["a"], device=device),
                                                                              torch.tensor(k["w"], device=device),
                                                                              torch.tensor(k["b"], device=device)) for
                               D, k in zip(Ds, self.kernels)])
        fkernels = torch.fft.fft2(torch.fft.fftshift(Ks / Ks.sum(dim=(0, 1), keepdims=True), dim=(0, 1)),
                                       dim=(0, 1))
        return fkernels
    def growth(self,U,m,s):
        return self.bell(U,m,s)*2 -1

    def soft_clip(self,x):
        return 1 / (1 + torch.exp(-4 * (x - 0.5)))

    def sigmoid(self,x):
        return 0.5 * (torch.tanh(x / 2) + 1)

    def forward(self, x):

        fXs = torch.fft.fft2(x, dim= (0,1))
        if self.c1 != None:
            fXk = fXs[ :, :, self.c0]
        else:
            fXk = torch.dstack([fXs[:, :, k["c0"]] for k in self.kernels])
        Us = torch.fft.ifft2(self.fkernels*fXk, dim=(0,1)).real
        Gs = self.growth(Us,self.m,self.s) *self.h
        if self.c1 != None:
            Hs = torch.dstack([ Gs[:, :, self.c1[c]].sum(dim=-1) for c in range(self.C) ])
        else:
            Hs = torch.dstack([sum(k["h"] * Gs[:,:,i] if k["c1"] == c1 else torch.zeros_like(Gs[:,:,i], device=self.device) for i, k in zip(range(Gs.shape[-1]), self.kernels)) for c1 in range(self.C)])


        grad_u = sobel(Hs)  # (c,2,y,x)

        grad_x = sobel(x.sum(dim=-1, keepdims=True))   # (1,2,y,x)

        alpha = (((x[:,:,None, :] / self.theta_x) ** self.n)).clip(0,1)

        F = grad_u * (1 - alpha) - grad_x * alpha
        x = self.rt.apply(x, F)

        return x


class ReintegrationTrackerParam():
    def __init__(self, X, Y, dt,dd=5, sigma=0.65):

        self.X = X
        self.Y = Y
        self.dd = dd
        self.dt = dt
        self.sigma = sigma
        self.pos = construct_mesh_grid(X,Y)
        self.dxs, self.dys = construct_ds(dd)

    def step_flow(self, X, H, mu, dx, dy):
        """Summary
        """
        Xr = torch.roll(X, (dx, dy), dims=(0, 1))
        Hr = torch.roll(H, (dx, dy), dims=(0, 1))  # (x, y, k)
        mur = torch.roll(mu, (dx, dy), dims=(0, 1))

        if self.border == 'torus':
            dpmu = torch.min(torch.stack(
                [torch.absolute(self.pos[..., None] - (mur + torch.array([di, dj])[None, None, :, None]))
                 for di in (-self.X, 0, self.X) for dj in (-self.Y, 0, self.Y)]
            ), dim=0)
        else:
            dpmu = torch.absolute(self.pos[..., None] - mur)

        sz = .5 - dpmu + self.sigma
        area = torch.prod(torch.clip(sz, 0, min(1, 2 * self.sigma)), dim=2) / (4 * self.sigma ** 2)
        nX = Xr * area
        return nX, Hr
    def apply(self, grid, H,F):

        ma = self.dd - self.sigma
        mu = self.pos[..., None] + (self.dt*F).clip(-ma,ma)
        mu = torch.clip(mu, self.sigma, self.X - self.sigma)
        ngrid = torch.stack([self.step_flow(grid, H,mu, dx, dy) for dx, dy in zip(self.dxs, self.dys)])



        return ngrid.sum(dim=0)



def conv_ones(x):
    k_y = torch.tensor([[1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1]],
                       dtype=torch.float32, device="cuda:0").T.tile((1, 1, 1, 1))
    sy =torch.vstack([torch.nn.functional.conv2d(x[None,:,:,c], k_y, groups= 1, stride=1, padding="same") for c in range(x.shape[-1])])
    return sy.permute((1,2,0))


class Lenia_Diff_MassConserve(torch.nn.Module):
    def __init__(self, C, dt, K, kernels, device, X, Y, mode="soft", has_food=None):
        super(Lenia_Diff_MassConserve, self).__init__()

        self.has_food = has_food
        self.dt = dt
        self.C = C
        self.n = 2.
        self.theta_x = 2
        self.kernels = kernels
        self.device = device
        self.midX = X // 2
        self.midY = Y // 2
        self.bell = lambda x, m, s: torch.exp(-((x - m) / s) ** 2 / 2)
        self.bell_kernel = lambda x, a, w, b: (b * torch.exp(-(x[..., None] - a) ** 2 / w)).sum(-1)
        self.mode = mode
        self.c0 = None
        self.c1 = None
        self.m = None
        self.s = None
        self.h = None
        self.pk = None
        self.fkernels = self.construct_kernels(K, device)

    def growth(self, U, m, s):
        return self.bell(U, m, s) * 2 - 1

    def soft_clip(self, x):
        return 1 / (1 + torch.exp(-4 * (x - 0.5)))



    def sigmoid(self,x):
        return 0.5 * (torch.tanh(x / 2) + 1)

    @torch.no_grad()
    def forward(self, x):

        fXs = torch.fft.fft2(x, dim=(0, 1))
        if self.c1 != None:
            fXk = fXs[:, :, self.c0]
        else:
            fXk = torch.dstack([fXs[:, :, k["c0"]] for k in self.kernels])
        Us = torch.fft.ifft2(self.fkernels * fXk, dim=(0, 1)).real
        Gs = self.growth(Us, self.m, self.s) * self.h
        if self.c1 != None:
            Hs = torch.dstack([Gs[:, :, self.c1[c]].sum(dim=-1) for c in range(self.C - self.has_food)])
        else:
            Hs = torch.dstack([sum(k["h"] * Gs[:, :, i] if k["c1"] == c1 else torch.zeros_like(Gs[:, :, i], device=self.device) for i, k in zip(range(Gs.shape[-1]), self.kernels)) for c1 in range(self.C)])

        # --- Mass Conservation Implementation ---

        if self.has_food:
            x_overlap = ((x[:,:,-1][...,None]-0.1 )* .9).clip(torch.zeros_like(x[:,:,0:1]), x[:,:,0:1])
            food= x[:,:,0:1] - x_overlap
            x = self.mass_conservation_step(x[:,:,self.has_food:], Hs)+ torch.cat([x_overlap/self.C for _ in range(self.C-1)], dim=-1) - (x[:,:,self.has_food:]*.0005)/(self.C-1)
            x = torch.cat((food, x), dim= -1)
        else :

            x = self.mass_conservation_step(x, Hs)


        return x
    def construct_kernels(self, K, device):
        if self.pk == "sparse":
            Ds = [torch.tensor(
                np.linalg.norm(np.mgrid[-self.midX:self.midX, -self.midY:self.midY], axis=0) / K*len(k["b"])/k["r"],
                device=self.device, dtype=torch.float32) for k in self.kernels]
        else:
            Ds = [torch.tensor(
                np.linalg.norm(np.mgrid[-self.midX:self.midX, -self.midY:self.midY], axis=0) / ((K+15)*k["r"]),
                device=self.device, dtype=torch.float32) for k in self.kernels]

        Ks = torch.dstack([self.sigmoid(-(D - 1) * 10) * self.bell_kernel(D, torch.tensor(k["a"], device=device),
                                                                              torch.tensor(k["w"], device=device),
                                                                              torch.tensor(k["b"], device=device)) for
                               D, k in zip(Ds, self.kernels)])
        fkernels = torch.fft.fft2(torch.fft.fftshift(Ks / Ks.sum(dim=(0, 1), keepdims=True), dim=(0, 1)),
                                       dim=(0, 1))
        return fkernels

    def mass_conservation_step(self, x_previous, Hs):
        H, W, C = x_previous.shape

        # Create neighbor indices (including self)
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
        rows = torch.arange(0, H, device=self.device).view(-1, 1)  # Column vector
        cols = torch.arange(0, W, device=self.device).view(1, -1)  # Row vector

        # Collect Hs and mass from the *previous* step for each neighbor
        neighbors_Hs = []
        neighbors_masses = []
        for row_offset, col_offset in offsets:
            neighbor_rows = torch.remainder(rows + row_offset, H)
            neighbor_cols = torch.remainder(cols + col_offset, W)
            neighbors_Hs.append(Hs[neighbor_rows, neighbor_cols, :])  # (H,W,C)
            neighbors_masses.append(x_previous[neighbor_rows, neighbor_cols, :])  # (H,W,C)

        neighborhood_Hs = torch.stack(neighbors_Hs, dim=0)  # (9,H,W,C)
        neighborhood_masses = torch.stack(neighbors_masses, dim=0)  # (9,H,W,C)

        #Compuet E^t_x,y
        E = torch.exp(neighborhood_Hs).sum(dim = 0)
        neighbors_E = []
        for row_offset, col_offset in offsets:
            neighbor_rows = torch.remainder(rows + row_offset, H)
            neighbor_cols = torch.remainder(cols + col_offset, W)
            neighbors_E.append(E[neighbor_rows, neighbor_cols, :])  # (H,W,C)

        neighborhood_E  = torch.stack(neighbors_E, dim=0) #(9,H,W,C)


        # Calculate redistribution  = e^S/N(E)*N(Mass)
        redistributions = (torch.exp(Hs)/neighborhood_E)*neighborhood_masses

        # 3. Calculate the new mass: Sum the redistributions
        new_mass = redistributions.sum(dim=0)  # (H,W,C)

        return new_mass