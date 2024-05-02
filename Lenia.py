import random
import torch

class Lenia(torch.nn.Module):
    def __init__(self, out_features, dt , k, n , kpc):
        super(Lenia, self).__init__()
        self.rand = (torch.rand((3))*10)
        self.out = out_features
        self.dt = dt
        self.k = k
        self.n =n
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
        self.grid = x
        u = self.kernel_pass(x)
        u = self.growth_func(u)  # growth functionm from results of kernel convolution
        u = self.weighted_update(u)
        x = self.update(u)
        return x

    def circular_sobel_filter(self,size, device='cuda:0', dtype=torch.float32):

        x = torch.linspace(-1, 1, size, device=device, dtype=dtype)
        y = torch.linspace(-1, 1, size, device=device, dtype=dtype)
        xx, yy = torch.meshgrid(x, y)
        distance = torch.sqrt(xx ** 2 + yy ** 2)



        # Compute Sobel in the radial direction
        sobel_filter = 1 / distance

        # Normalize the filter
        sobel_filter /= torch.min(torch.abs(sobel_filter))

        return sobel_filter
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
        sum_k = self.kernels.weight.data.sum(dim = (-1,-2)).swapaxes(0,1)
        u = self.kernels(x)
        u = u / sum_k[:,:,None,None] if len(sum_k.shape) > 0 else sum_k
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
            r = torch.arange(0, size, device="cuda:0") - (size - 1) / 2
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



class Lenia2(torch.nn.Module):
    def __init__(self, out_features, dt , k, n , kpc):
        super(Lenia2, self).__init__()
        self.rand = torch.rand((3))*10
        self.out = out_features
        self.dt = dt
        self.k = k
        self.n =n
        self.kpc = kpc
        self.kernels_self = torch.nn.Conv2d(out_features,self.kpc*out_features,self.k,1,padding="same", padding_mode="circular", groups=out_features)
        self.heigts_self = torch.nn.Parameter(torch.rand((self.kpc*out_features,self.n))*self.rand[0])
        self.radii_self = torch.nn.Parameter(torch.rand((self.kpc*out_features,self.n)))
        for j in range(self.kpc*out_features):
            for i in range(self.n):
                self.radii_self.data[j,i] = (1/self.n)*i + (random.random()-0.5) * (self.k/100)*self.rand[1]
        self.width_self = torch.nn.Parameter(torch.randn((self.kpc*out_features,self.n))*self.rand[2])
        self.mu_self = torch.nn.Parameter(torch.rand(self.kpc*out_features)*10)
        self.sigma_self = torch.nn.Parameter(torch.rand(self.kpc*out_features)*10)

        self.kernels_cross  = torch.nn.Conv2d(out_features,self.kpc*out_features,self.k,1,padding="same", padding_mode="circular")
        self.heigts_cross = torch.nn.Parameter(torch.rand((self.kpc*out_features, self.n)) * self.rand[0])
        self.radii_cross = torch.nn.Parameter(torch.rand((self.kpc* out_features, self.n)))
        for j in range(self.kpc*out_features):
            for i in range(self.n):
                self.radii_cross.data[j, i] = (1 / self.n) * i + (random.random() - 0.5) * (self.k / 100) * self.rand[1]
        self.width_cross= torch.nn.Parameter(torch.randn((self.kpc*out_features, self.n)) * self.rand[2])
        self.mu_cross = torch.nn.Parameter(torch.rand(self.kpc* out_features)*10)

        self.sigma_cross = torch.nn.Parameter(torch.rand( self.kpc*out_features)*10)


        self.weights = torch.nn.Conv2d(self.kpc*2*out_features,out_features,1,1,padding="same", bias=False)
        self.grid = torch.tensor([])

        self.normalise_kernels()
        self.kernels_self = self.create_gaussian_bumps_kernel(self.n, self.heigts_self, self.radii_self,
                                                                self.width_self, self.k, self.kernels_self)
        self.kernels_cross = self.create_gaussian_bumps_kernel(self.n, self.heigts_cross, self.radii_cross,
                                                               self.width_cross, self.k, self.kernels_cross)

    def forward(self, x, steps: int ):
        self.grid = x

        for i in range(steps):
            u = self.kernel_pass(x)
            u = self.growth_func(u, self.mu_self, self.sigma_self)  # growth functionm from results of kernel convolution
            u2 = self.kernels_cross(x[:,:,:,:]).squeeze(dim = 2)
            u2 = self.growth_func(u2,self.mu_cross, self.sigma_cross)
            u = torch.cat((u,u2),dim = 1)
            u = self.weighted_update(u)
            x = self.update(u)

        return x
    def update(self, u):
        self.grid = self.grid + (self.dt*u)                # time step
        self.grid = torch.clip(self.grid, 0,1)
        return self.grid

    def normalise_kernels(self):
        new_weights  = torch.zeros_like(self.weights.weight.data)
        for i in range(self.weights.weight.data.shape[0]):
            self.weights.weight.data[i] = self.weights.weight.data[i].sub_(torch.min(self.weights.weight.data[i]))
            self.weights.weight.data[i] = self.weights.weight.data[i].div_(self.weights.weight.data[i].sum())
        #new_weights[:, self.kpc*self.out:,:,:] = self.weights.weight.data[:, self.kpc*self.out:,:,:]
        for j in range(self.out):
            new_weights[j,j*self.kpc:(j+1)*self.kpc] = self.weights.weight.data[j,j*self.kpc:(j+1)*self.kpc]
            new_weights[j,self.kpc*(self.out+j):self.kpc*(self.out+j+1)] = self.weights.weight.data[j,self.kpc*(self.out+j):self.kpc*(self.out+j+1)]

            if j == 0:
                new_weights[j, -self.kpc :] = self.weights.weight.data[j, -self.kpc:]
            else:
                new_weights[j, self.kpc * (self.out + j -1):self.kpc * (self.out + j )] = self.weights.weight.data[j,self.kpc * (self.out + j -1):self.kpc * (self.out + j )]
        self.weights.weight.data = new_weights
        for i in range(self.weights.weight.data.shape[0]):
            self.weights.weight.data[i] = self.weights.weight.data[i].sub_(torch.min(self.weights.weight.data[i]))
            self.weights.weight.data[i] = self.weights.weight.data[i].div_(self.weights.weight.data[i].sum())







    def growth_func(self, u, mu: torch.nn.Parameter, sigma: torch.nn.Parameter):
        #torch.max(torch.zeros_like(u), 1 - (u - mu[None,:,None,None]) ** 2 / (9 * sigma[None,:,None,None] ** 2)) ** 4 * 2 - 1
        return  (2*torch.exp((-(u - mu[None,:,None,None]) ** 2) / (2 * (sigma[None,:,None,None] ** 2))) -1)

    def kernel_pass(self, x):
        #u = self.k_inner(x)
        sum_k = self.kernels_self.weight.data.sum(dim = (-1,-2)).squeeze()
        u = self.kernels_self(x)
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
            r = torch.arange(0, size, device="cuda:0") - (size - 1) / 2
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


class Lenia3(torch.nn.Module):
    def __init__(self, out_features, dt , k, n , kpc):
        super(Lenia3, self).__init__()
        self.rand = torch.rand((3))*10
        self.out = out_features
        self.dt = dt
        self.k = k
        self.n =n
        self.kpc = kpc
        self.kernels_self = torch.nn.Conv2d(out_features,self.kpc*out_features,self.k,1,padding="same", padding_mode="circular", groups=out_features)
        self.heigts_self = torch.nn.Parameter(torch.rand((self.kpc*out_features,self.n))*self.rand[0])
        self.radii_self = torch.nn.Parameter(torch.rand((self.kpc*out_features,self.n)))
        for j in range(self.kpc*out_features):
            for i in range(self.n):
                self.radii_self.data[j,i] = ((1*i)/self.n)+0.1 + (self.k/2000)*self.rand[1]*2
        self.width_self = torch.nn.Parameter(torch.randn((self.kpc*out_features,self.n))*self.rand[2])
        self.mu_self = torch.nn.Parameter(torch.rand(self.kpc*out_features)/100)
        self.sigma_self = torch.nn.Parameter(torch.rand(self.kpc*out_features)/100)

        self.kernels_cross = torch.nn.Conv3d(out_features, self.kpc*out_features, (2,self.k,self.k), 1, padding="same",
                                       padding_mode="circular" , groups=out_features)
        self.heigts_cross = torch.nn.Parameter(torch.rand((self.kpc*out_features, self.n)) * self.rand[0])
        self.radii_cross = torch.nn.Parameter(torch.rand((self.kpc* out_features, self.n)))
        for j in range(self.kpc*out_features):
            for i in range(self.n):
                self.radii_cross.data[j,i] = ((1*i)/self.n)+0.1 + (self.k/2000)*self.rand[1]*2
        self.width_cross= torch.nn.Parameter(torch.randn((self.kpc*out_features, self.n)) * self.rand[2])
        self.mu_cross = torch.nn.Parameter(torch.rand(self.kpc* out_features)/100)

        self.sigma_cross = torch.nn.Parameter(torch.rand( self.kpc*out_features)/100)


        self.weights = torch.nn.Conv2d(self.kpc*2*out_features,out_features,1,1,padding="same", bias=False)
        self.grid = torch.tensor([])

        self.normalise_kernels()
        self.kernels_self = self.create_gaussian_bumps_kernel(self.n, self.heigts_self, self.radii_self,
                                                              self.width_self, self.k, self.kernels_self)
        self.kernels_cross = self.create_gaussian_bumps_kernel3d(self.n, self.heigts_cross, self.radii_cross,
                                                                 self.width_cross, self.k, self.kernels_cross)

    def forward(self, x, steps: int ):
        self.grid = x

        for i in range(steps):
            u = self.kernel_pass(x)
            u = self.growth_func(u, self.mu_self, self.sigma_self)  # growth functionm from results of kernel convolution
            u2 = self.kernels_cross(x[:,:,None,:,:]).squeeze(dim = 2)
            u2 = self.growth_func(u2,self.mu_cross, self.sigma_cross)
            u = torch.cat((u,u2),dim = 1)
            u = self.weighted_update(u)
            x = self.update(u)

        return x
    def update(self, u):
        self.grid = self.grid + (self.dt*u)                # time step
        self.grid = torch.clip(self.grid, 0,1)
        return self.grid

    def normalise_kernels(self):
        new_weights  = torch.zeros_like(self.weights.weight.data)
        for i in range(self.weights.weight.data.shape[0]):
            self.weights.weight.data[i] = self.weights.weight.data[i].sub_(torch.min(self.weights.weight.data[i]))
            self.weights.weight.data[i] = self.weights.weight.data[i].div_(self.weights.weight.data[i].sum())

        for j in range(self.out):
            new_weights[j,j*self.kpc:(j+1)*self.kpc] = self.weights.weight.data[j,j*self.kpc:(j+1)*self.kpc]
            new_weights[j,self.kpc*(self.out+j):self.kpc*(self.out+j+1)] = self.weights.weight.data[j,self.kpc*(self.out+j):self.kpc*(self.out+j+1)]

            if j == 0:
                new_weights[j, -self.kpc :] = self.weights.weight.data[j, -self.kpc:]
            else:
                new_weights[j, self.kpc * (self.out + j -1):self.kpc * (self.out + j )] = self.weights.weight.data[j,self.kpc * (self.out + j -1):self.kpc * (self.out + j )]
        self.weights.weight.data = new_weights
        for i in range(self.weights.weight.data.shape[0]):
            self.weights.weight.data[i] = self.weights.weight.data[i].sub_(torch.min(self.weights.weight.data[i]))
            self.weights.weight.data[i] = self.weights.weight.data[i].div_(self.weights.weight.data[i].sum())







    def growth_func(self, u, mu: torch.nn.Parameter, sigma: torch.nn.Parameter):
        #torch.max(torch.zeros_like(u), 1 - (u - mu[None,:,None,None]) ** 2 / (9 * sigma[None,:,None,None] ** 2)) ** 4 * 2 - 1
        return  (2*torch.exp((-(u - mu[None,:,None,None]) ** 2) / (2 * (sigma[None,:,None,None] ** 2))) -1)

    def kernel_pass(self, x):
        #u = self.k_inner(x)
        sum_k = self.kernels_self.weight.data.sum(dim = (-1,-2)).squeeze()
        u = self.kernels_self(x)
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
            r = torch.arange(0, size, device="cuda:0") - (size - 1) / 2
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

    def create_gaussian_bumps_kernel3d(self,k:int, height:torch.nn.Parameter, radii:torch.nn.Parameter, widths:torch.nn.Parameter, size:int, kernels:torch.nn.Conv3d):

        if radii.shape[1] != k or widths.shape[1] != k:
            raise ValueError("Number of radii and widths should be equal to k.")
        for i in range(kernels.weight.data.shape[0]):


            # Create a grid of coordinates
            r = torch.arange(0, size, device="cuda:0") - (size - 1) / 2
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
            kernels.weight.data[i,:,:,:,:] = torch.cat((kernel[None,:,:], kernel[None,:,:]), dim =0)[None,:,:,:]

        return kernels