import tkinter as tk
import cv2
import random
import threading
import pickle
from tkinter import *
import numpy as np

from FlowLenia import *
from pyperlin import FractalPerlin2D

SPACE_CLASSIC= {
                "bn" : 3,
                "b"  :{"high": 1., "low":-1.0},
                "m"  :{"high": .7, "low":0},
                "s"  :{"high": .2, "low":0},
                "h"  :{"high": 1., "low":0.01},
                "r"  :{"high": 1., "low":0.01},
                "w"  :{"high": 1, "low":0.01},
                "a"  :{"high": 1., "low":0.01}
}


class Saver:
    def __init__(self, kernel, dt, C, R, i, is_flow, c0, c1, pk, hf, class_name, save_path):
        self.dict = {}
        self.dict["C"] = C
        self.dict["R"] = R
        self.dict["dt"] = dt
        self.dict["kernels"] = kernel
        self.dict["pattern"] = None
        self.dict["pk"] = pk
        self.dict["hf"] = hf
        self.i = str(i)
        self.is_flow = is_flow
        self.c0 = c0
        self.c1 = c1
        self.class_name = class_name
        self.save_path = save_path


    def save_to_dict(self):
        if self. is_flow:
            self.dict["c0"] = self.c0
            self.dict["c1"] = self.c1
        return self.dict

    def save(self):
        with open(self.save_path + self.class_name + '_' + self.i + '.pkl', 'wb') as f:
            pickle.dump(self.dict, f)

class App(threading.Thread):
    def __init__(self, dict, save_path, i, class_name):
        threading.Thread.__init__(self)
        self.save_path = save_path
        self.dict = dict
        self.i = i
        self.class_name = class_name
        self.start()
    def callback(self):
        self.root.quit()
    def saveFile(self):
        with open(self.save_path+ self.class_name+ '_' + self.i + '.pkl', 'wb') as f:
            pickle.dump(self.dict, f)
    def run(self):
        self.root = tk.Tk()
        self.root.protocol("WM_DELETE_WINDOW", self.callback)
        button = Button(text="save", command=self.saveFile)
        button.pack()
        self.root.mainloop()
def kernels_config_constructor_classic(kernel_config, n, C):
    if kernel_config !=None:
        return kernel_config["kernels"]
    else:
        kernel_config = [
            {'b':[random.uniform(SPACE_CLASSIC["b"]["low"],SPACE_CLASSIC["b"]["high"]) for _ in range(random.randint(1,SPACE_CLASSIC["bn"]))],
            "m":random.uniform(SPACE_CLASSIC["m"]["low"],SPACE_CLASSIC["m"]["high"]),
            "s":random.uniform(SPACE_CLASSIC["s"]["low"],SPACE_CLASSIC["s"]["high"]),
            "h": random.uniform(SPACE_CLASSIC["h"]["low"],SPACE_CLASSIC["h"]["high"]),
            "r":random.uniform(SPACE_CLASSIC["r"]["low"],SPACE_CLASSIC["r"]["high"]),
            "c0":random.randint(0,C-1),
            "c1": random.randint(0,C-1)
             }
            for _ in range(n)
        ]
        return kernel_config

def kernels_config_constructor_df(kernel_config, n, C):
    if kernel_config !=None:
        return kernel_config["kernels"]
    else:

        kernel_config = [
            {"b":[random.uniform(SPACE_CLASSIC["b"]["low"],SPACE_CLASSIC["b"]["high"]) for _ in range(SPACE_CLASSIC["bn"])],
            "m":random.uniform(SPACE_CLASSIC["m"]["low"],SPACE_CLASSIC["m"]["high"]),
            "s":random.uniform(SPACE_CLASSIC["s"]["low"],SPACE_CLASSIC["s"]["high"]),
            "h": random.uniform(SPACE_CLASSIC["h"]["low"],SPACE_CLASSIC["h"]["high"]),
            "r":random.uniform(SPACE_CLASSIC["r"]["low"],SPACE_CLASSIC["r"]["high"]),
            "c0": int(C*(i/n)),
            "c1": i%C,
            "w": [random.uniform(SPACE_CLASSIC["w"]["low"], SPACE_CLASSIC["w"]["high"]) for _ in
                   range(SPACE_CLASSIC["bn"])],
            "a": [random.uniform(SPACE_CLASSIC["a"]["low"], SPACE_CLASSIC["a"]["high"]) for _ in
                   range(SPACE_CLASSIC["bn"])]
             }
            for i in range(n)
        ]
        for config in kernel_config:
            config["s"] = random.uniform(0, 0.9*(config["m"]/(2*np.log(2))))
        return kernel_config

def get_setting(setting_config, has_food, has_food_range = [1,2,3,4], no_food_range = [1,2,3], dt_range = [4,10], kernel_range = [2,25], n_range = [3,15], pk_choice =["dense", "sparse"] ):
    if setting_config != None:
        C, dt, k, n, pk, has_food = setting_config["C"], setting_config["dt"], setting_config["R"], len(setting_config["kernels"]), setting_config["pk"] if "pk" in setting_config else None, setting_config["hf"] if "hf" in setting_config else None
        return  C, dt, k, n, pk, has_food
    else:
        C = random.choice(no_food_range) if not has_food else random.choice(has_food_range)
        dt = 1 / random.randint(min(dt_range),max(dt_range))
        k = random.randint(min(kernel_range), max(kernel_range))
        n = random.randint(min(n_range),max(n_range))
        pk = random.choice(pk_choice)
        return C, dt, k, n, pk, has_food
def get_starting_pattern(pattern_confing, starting_area, C, X, Y,has_food,num_food =500,food_size = 2 ,device= "cuda:0",  full =True, wavelength = 10):
    if pattern_confing != None and pattern_confing["pattern"] != None:
        x = torch.zeros(( X, Y, C), dtype=torch.float32, device=device)

        if type(pattern_confing["pattern"]) == list:
            pattern= torch.tensor(pattern_confing["pattern"]).permute((1,2,0))
            x[X // 2 - pattern.shape[0] // 2: X // 2 + pattern.shape[0] // 2 + 1 if pattern.shape[
                                                                                        0] % 2 != 0 else X // 2 +
                                                                                                         pattern.shape[
                                                                                                             0] // 2,
            Y // 2 - pattern.shape[1] // 2: Y // 2 + pattern.shape[1] // 2 + 1 if pattern.shape[
                                                                                      1] % 2 != 0 else Y // 2 +
                                                                                                       pattern.shape[
                                                                                                           1] // 2,
            :] = pattern
        else:
            pattern = torch.tensor(pattern_confing["pattern"])
            x = pattern

        return x
    else:
        if not full:
            x = torch.zeros(( X, Y, C), dtype=torch.float32, device=device)
            x[
            X // 2 - starting_area // 2: X // 2 + starting_area // 2 + 1 if starting_area % 2 != 0 else X // 2 +
                                                                                                                 starting_area // 2,
            Y // 2 - starting_area // 2: Y // 2 + starting_area // 2 + 1 if starting_area % 2 != 0 else Y // 2 +
                                                                                                                 starting_area // 2,:] = perlin((C,starting_area, starting_area), wavelengths= [wavelength]*2, black_prop=0.25)
        else:

            x = perlin((C,X,Y), wavelengths= [wavelength]*2, black_prop=0.25)


        if has_food:
            x_food = get_food_pos(X, Y, num_spots=num_food, food_size=food_size) * 1
            x = torch.cat((x_food, x), dim=-1)
        return x

def cs_constructor(C, n, config, is_flow , from_saved, has_food= True, food_channel_only = False):
    if not is_flow :
        return None, None, n
    elif is_flow and from_saved:
        return config["c0"], config["c1"], len(config["kernels"])

    if has_food:
        M = np.ones((C+1,C+1), dtype=int)*n
    else:
        M = np.ones((C , C ), dtype=int) * n
    C = M.shape[0]
    c0 = []
    c1 = [[] for _ in range(C)]

    i = 0
    for s in range(C):
        for t in range(C):
            n = M[s, t]
            if n:
                c0 = c0 + [s] * n
                c1[t] = c1[t] + list(range(i, i + n))
            i += n
    if has_food:
        c2 = c1.pop(0)
        if food_channel_only:
            for j in range(len(c2)):
                c1[-1].append(c2[j])
        else:
            for j in range(len(c2)):
                c1[int((len(c1)/len(c2))*j)].append(c2[j])


    return c0, c1, int(M.sum())

def adjust_params(nca, c0, c1,pk, device="cuda:0"):


    if (nca.__class__ == Lenia_Flow) or (nca.__class__ == Lenia_Diff_MassConserve):
        nca.c0 = c0
        nca.c1 = c1
        nca.h = torch.stack([torch.tensor(k["h"], device=device)for k in nca.kernels], dim = 0)
        nca.s = torch.stack([torch.tensor(k["s"], device=device)for k in nca.kernels], dim = 0)
        nca.m = torch.stack([torch.tensor(k["m"], device=device)for k in nca.kernels], dim = 0)
        nca.pk = pk
        return nca
    else: return nca

def get_food_pos(X, Y, num_spots =100, food_size = 5):
    places = [[random.randint(food_size, X-food_size), random.randint(food_size, Y-food_size)] for _ in range(num_spots)]
    x = torch.zeros((X,Y,1), device="cuda:0")
    for place in places:
        x[place[0]- food_size: place[0]+ food_size, place[1]-food_size:place[1]+ food_size] = 1
    return x

def render(grid, with_food, waitkey =1, multiplier = 1):

    if not with_food:
        if grid.shape[-1] == 2:
            img = np.zeros((grid.shape[0], grid.shape[1], 3))
            img[:, :, [1, 2]] = grid
        else:
            img = grid



    if with_food:
        if grid.shape[-1] == 2:
            img = np.zeros((grid.shape[0], grid.shape[1], 3))
            img[:, :, 0] = grid[:,:,0]
            img[:, :, :] += grid[:, :, 1][:, :, None]
        if grid.shape[-1] == 3:
            img = grid
        if grid.shape[-1] == 4:
            img = np.zeros((grid.shape[0], grid.shape[1], 3))
            img[:,:,:] = grid[:,:,1:4]
            img[:, :, :] += grid[:, :, 0][:,:,None]


    img = cv2.resize(img, (img.shape[0]*multiplier,img.shape[1]*multiplier), interpolation=cv2.INTER_AREA)
    cv2.imshow("Lenia", img)
    key = cv2.waitKey(waitkey)
    return key


def mix(alpha, ca_sub, ca_over, keys):
    mixed_ca = {}
    new_kernels = []
    for k in ca_sub:
        if k != "kernels":
            mixed_ca[k] = ca_sub[k]

    for k1,k2 in zip(ca_sub["kernels"], ca_over["kernels"]):
        new_kernel = {}
        for key in k1:
            if key not in keys:
                new_kernel[key] = k1[key]
            else:
                if type(k1[key]) == list:
                    new_vals = [(alpha*v1) + ((1-alpha)*v2) for v1,v2 in zip(k1[key], k2[key])]
                else:
                    new_vals = (alpha * k1[key]) + ((1 - alpha) * k2[key])
                new_kernel[key] = new_vals
        new_kernels.append(new_kernel)

    mixed_ca["kernels"] = new_kernels

    return mixed_ca

@torch.no_grad()
def rand_search(ca, keys, bias):
    mixed_ca = {}
    new_kernels = []
    for k in ca:
        if k != "kernels":
            mixed_ca[k] = ca[k]

    for k in ca["kernels"]:
        new_kernel = {}
        for key in k:
            if key not in keys:
                new_kernel[key] = k[key]
            else:
                if type(k[key]) == list:
                    new_vals = [val + (random.uniform(-0.1,0.1)*val) + bias for val in k[key]]
                else:
                    new_vals = k[key] + (random.uniform(-0.1,0.1)*k[key]) + bias
                new_kernel[key] = new_vals
        new_kernels.append(new_kernel)

    mixed_ca["kernels"] = new_kernels

    return mixed_ca

def perlin(shape:tuple, wavelengths:tuple, black_prop:float=0.3,device='cuda:0'):
    C,H,W = tuple(shape)
    lams = tuple(int(wave) for wave in wavelengths)
    # Extend image so that its integer wavelengths of noise
    W_new=int(W+(lams[0]-W%lams[0]))
    H_new=int(H+(lams[1]-H%lams[1]))
    frequency = [H_new//lams[0],W_new//lams[1]]
    gen = torch.Generator(device=device) # for GPU acceleration
    gen.seed()
    # Strange 1/0.7053 factor to get images noise in range (-1,1), quirk of implementation I think...
    fp = FractalPerlin2D((C,H_new,W_new), [frequency], [1/0.7053], generator=gen)()[:,:H,:W].moveaxis(0,2) # (B*3,H,W) noise)

    return torch.clamp((fp+(0.5-black_prop)*2)/(2*(1.-black_prop)),0,1)

def construct_ca(s_uuid, save_path,full_screen_sim_x, full_screen_sim_y, mode, dict, ca_type, keys= ["m", "s", "w", "a", "b", "h"], has_food = False, is_flow = False, device = "cuda:0", vary = False, starting_area = 100, full_noise = True, has_food_range = [1,2,3,4], no_food_range = [3], dt_range = [10], kernel_range = [31], n_range = [15], pk_choice =["sparse"], bias = 0 ):
    if vary:
        random_config = rand_search(dict, keys, bias)
    else:
        random_config = dict
    C, dt, k, n,pk, has_food = get_setting(random_config, has_food,has_food_range = has_food_range, no_food_range=no_food_range, dt_range=dt_range, kernel_range=kernel_range,n_range=n_range, pk_choice=pk_choice )
    c0, c1, n = cs_constructor(C, n, random_config, is_flow=is_flow, from_saved=dict is not None, has_food=has_food)
    kernels = kernels_config_constructor_df(random_config, n, C)
    C = C + has_food if has_food is not None else C
    nca = ca_type(C, dt, k, kernels, device, full_screen_sim_x, full_screen_sim_y, mode=mode, has_food=has_food).to(
        device).eval()
    nca = adjust_params(nca, c0, c1, pk)
    x = get_starting_pattern(random_config, starting_area, C - has_food if has_food is not None else C, full_screen_sim_x,
                             full_screen_sim_y, has_food, num_food=500, full=full_noise, wavelength=k)
    saver = Saver(kernels, dt, C - has_food if has_food is not None else C, k, s_uuid, is_flow, c0, c1, pk, has_food,
                  nca.__class__.__name__, save_path)
    return nca, x, saver

@torch.no_grad()
def dichotomy(N_steps,params_over, params_sub,refinement,threshold,keys,s_uuid, save_path, full_screen_sim_x, full_screen_sim_y, mode,
                                       ca_type,
                                       has_food, is_flow, starting_area, full_noise):

    params_sub = params_sub
    params_over = params_over
    t_crit =0.5
    for i in range(refinement):
        print(i)
        params_mix = mix(0.5, params_sub, params_over, keys = keys)
        nca, x, saver = construct_ca(s_uuid, save_path, full_screen_sim_x, full_screen_sim_y, mode, params_mix,
                                           ca_type,
                                           has_food=has_food, is_flow=is_flow, starting_area=starting_area, vary = False, full_noise=full_noise)
        #saver.dict["pattern"] = x

        for _ in range(N_steps):
            x = nca(x)
        mass = x.mean()
        if mass < threshold:
            params_sub = params_mix
            t_crit += 0.5 ** (i + 2)
        elif mass > threshold:
            params_over = params_mix
            t_crit -= 0.5 ** (i + 2)

    saver.save()
    return t_crit

def to_tenser(kernels, device):
    new_kernels = []

    for k in kernels:
        new_kernel = {}
        for key in k:

                new_kernel[key] = torch.tensor(k[key], device=device)
        new_kernels.append(new_kernel)

    return new_kernels