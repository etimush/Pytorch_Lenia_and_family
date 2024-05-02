import random
import torch

def sobel_x(x):
    k_x = torch.tensor([[-1, 0, +1],
                            [-2, 0, +2],
                            [-1, 0, +1]],
    dtype=torch.float16).tile((x.shape[1], 1, 1, 1)).cuda()
    sx =torch.nn.functional.conv2d(x, k_x, groups= x.shape[1], stride=1, padding="same")
    return sx

def sobel_y(x):
    k_y = torch.tensor([[-1, -2, -1],
                            [0, 0, 0],
                            [+1, 2, +1]],
    dtype=torch.float16).tile((x.shape[1], 1, 1, 1)).cuda()
    sy =torch.nn.functional.conv2d(x, k_y, groups= x.shape[1], stride=1, padding="same")
    return sy

def sobel(x):
    sx = sobel_x(x)
    sy = sobel_y(x)
    sxy = torch.cat((sx[:,:,None,:,:], sy[:,:,None,:,:]), dim= 2)
    return sxy

class Lenia(torch.nn.Module):
    def __init__(self, out_features, dt , k, n , kpc, device):
        super(Lenia, self).__init__()
        self.rand = (torch.rand((3))*10)
        self.out = out_features
        self.device = device
        self.dt = dt
        self.k = k
        self.n =n
        self.n2 = 2
        self.theta_x = 1.0
        self.kpc = kpc
        self.kernels = torch.nn.Conv2d(out_features,self.kpc*out_features,self.k,1,padding="same", padding_mode="circular", groups=out_features, bias=False, dtype=torch.float16)

        self.heigts = torch.nn.Parameter(torch.rand((self.kpc*out_features,self.n))*self.rand[0],)
        self.radii = torch.nn.Parameter(torch.rand((self.kpc*out_features,self.n)))
        for j in range(self.kpc*out_features):
            for i in range(self.n):
                self.radii.data[j,i] = ((1*i)/self.n)+0.1 + (self.k/2000)*self.rand[1]*2

        self.width = torch.nn.Parameter(torch.randn((self.kpc*out_features,self.n))*self.rand[2]/3)
        self.mu_conv = torch.nn.Conv2d(self.kpc*out_features,self.kpc*out_features,1,1,padding="same", padding_mode="circular", groups=self.kpc*out_features, dtype=torch.float16)
        self.sigma_conv = torch.nn.Conv2d(self.kpc*out_features,self.kpc*out_features,1,1,padding="same", padding_mode="circular", groups=self.kpc*out_features, bias=False, dtype=torch.float16)
        self.weights = torch.nn.Conv2d(self.kpc*out_features,out_features,1,1,padding="same", bias=False,dtype=torch.float16)
        self.grid = torch.tensor([])
        self.kernels = self.create_gaussian_bumps_kernel(self.n, self.heigts, self.radii, self.width, self.k,
                                                         self.kernels)

        self.normalise_kernels()



    def forward(self, x):
        ## Normal Lenia
        self.grid = x
        u = self.kernel_pass(x)
        u = self.growth_func(u)  # growth functionm from results of kernel convolution
        u = self.weighted_update(u)
        ## Flow Part
        grad_u = sobel(u) #(1,c,2,x,y)
        grad_x = sobel(x.sum(dim=1, keepdims=True))# (1,1,2,x,y)
        alpha = ((x[:,:,None,:,:]/self.theta_x)**self.n2).clip(0.0,1.0)
        F = grad_u *(1-alpha)  - grad_x * alpha
        


        x = self.update(u)
        return x

    def update(self, u):
        #self.grid = self.grid  + (self.dt*u)                # time step
        self.grid = self.grid+u*self.dt
        self.grid = torch.clip(self.grid, 0,1)
        return self.grid

    def normalise_kernels(self):
        #for i in range(self.k_inner.weight.shape[0]):
            #self.k_inner.weight.data[i] = self.k_inner.weight.data[i].sub_(torch.min(self.k_inner.weight.data[i])).div_(torch.max(self.k_inner.weight.data[i]) - torch.min(self.k_inner.weight.data[i]))
        for i in range(self.weights.weight.data.shape[0]):


            self.weights.weight.data[i] = self.weights.weight.data[i].sub_(torch.min(self.weights.weight.data[i]))
            self.weights.weight.data[i] = self.weights.weight.data[i].div_(self.weights.weight.data[i].sum())
        self.mu_conv.weight.data[:] = 1
        self.mu_conv.bias.data[:] = - torch.rand(self.kpc*self.out)/3

        self.sigma_conv.weight.data[:] = 1/(2 * ((torch.rand(self.kpc*self.out)/10)[:,None,None,None] ** 2))



    def growth_func(self, u):
        #torch.max(torch.zeros_like(u), 1 - (u - mu[None,:,None,None]) ** 2 / (9 * sigma[None,:,None,None] ** 2)) ** 4 * 2 - 1
        return  (2*torch.exp(self.sigma_conv(-(self.mu_conv(u)) ** 2) ) -1)

    def kernel_pass(self, x):
        #u = self.k_inner(x)
        sum_k = self.kernels.weight.data.sum(dim = (-1,-2)).squeeze()
        u = self.kernels(x)
        u = u / sum_k[:,None,None] if len(sum_k.shape) > 0 else sum_k
        return u


    def weighted_update(self,g):
        g = self.weights(g)
        g = g / self.weights.weight.data.sum(dim =1)
        return g



    def create_gaussian_bumps_kernel(self,k:int, height:torch.nn.Parameter, radii:torch.nn.Parameter, widths:torch.nn.Parameter, size:int, kernels:torch.nn.Conv2d):

        if radii.shape[1] != k or widths.shape[1] != k:
            raise ValueError("Number of radii and widths should be equal to k.")
        for i in range(kernels.weight.data.shape[0]):


            # Create a grid of coordinates
            r = torch.arange(0, size, device=self.device) - (size - 1) / 2
            x, y = torch.meshgrid(r, r)

            # Initialize the kernel
            kernel = torch.zeros_like(x)

            for j in range(k):
                # Calculate the radial distance from the center
                radial_distance = torch.sqrt(x ** 2 + y ** 2)

                # Calculate Gaussian values with circular symmetry
                gaussian_kernel = height[i,j] * torch.exp(-(radial_distance - radii[i,j]*(size/2)) ** 2 / (2 * widths[i,j] ** 2))

                # Accumulate the Gaussian bumps
                kernel += gaussian_kernel

            # Normalize the kernel to make the sum equal to 1
            kernel = (kernel-torch.min(kernel))/(torch.max(kernel)-torch.min(kernel))
            #kernel/kernel.sum()
            kernels.weight.data[i,:,:,:] = kernel

        return kernels