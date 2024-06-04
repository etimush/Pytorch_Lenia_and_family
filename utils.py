import random
import tkinter as tk
import numpy as np
import cv2
import threading
import pickle
import torch
from tkinter import *
from FlowLenia import *
SPACE_CLASSIC= {
                "bn" : 3,
                "b"  :{"high": 1., "low":0.001},
                "m"  :{"high": .5, "low":.05},
                "s"  :{"high": .18, "low":.001},
                "h"  :{"high": 1., "low":.01},
                "r"  :{"high": 1., "low":.2},
                "w"  :{"high": .5, "low":.01},
                "a"  :{"high": 1., "low":.0}
}




class App(threading.Thread):
    def __init__(self, kernel, dt, C, R, i, is_flow, c0, c1, class_name, pk, hf):
        threading.Thread.__init__(self)
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
        self.start()
    def callback(self):
        self.root.quit()
    def saveFile(self):
        if self. is_flow:
            self.dict["c0"] = self.c0
            self.dict["c1"] = self.c1
        with open('./supervised_saved/'+ self.class_name+ '_' + self.i + '.pkl', 'wb') as f:
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
            "c0":random.randint(0,C-1),
            "c1": random.randint(0,C-1),
            "w": [random.uniform(SPACE_CLASSIC["w"]["low"], SPACE_CLASSIC["w"]["high"]) for _ in
                   range(SPACE_CLASSIC["bn"])],
            "a": [random.uniform(SPACE_CLASSIC["a"]["low"], SPACE_CLASSIC["a"]["high"]) for _ in
                   range(SPACE_CLASSIC["bn"])]
             }
            for _ in range(n)
        ]
        return kernel_config

def get_setting(setting_config, has_food):
    if setting_config != None:
        C, dt, k, n, pk, has_food = setting_config["C"], setting_config["dt"], setting_config["R"], len(setting_config["kernels"]), setting_config["pk"] if "pk" in setting_config else None, setting_config["hf"]
        return  C, dt, k, n, pk, has_food
    else:
        C = random.choice([1,2,3,4]) if not has_food else random.choice([1,2,3])
        dt = 1 / random.randint(5,10)
        k = random.randint(2, 51)
        n = random.randint(2,15)
        pk = random.choice(["dense", "sparse"])
        return C, dt, k, n, pk, has_food
def get_starting_pattern(pattern_confing, starting_area, C, X, Y,has_food,num_food =1500,food_size = 2 ,device= "cuda:0"):
    if pattern_confing != None and pattern_confing["pattern"] != None:
        x = torch.zeros(( X, Y, C), dtype=torch.float32, device=device)
        pattern= torch.tensor(pattern_confing["pattern"]).permute((1,2,0))
        x[X // 2 - pattern.shape[0] // 2: X // 2 + pattern.shape[0] // 2 + 1 if pattern.shape[0] %2 != 0 else  X // 2 + pattern.shape[0] // 2,
        Y // 2 - pattern.shape[1] // 2: Y // 2 + pattern.shape[1] // 2 + 1 if pattern.shape[1] %2 != 0 else Y // 2 + pattern.shape[1] // 2,:] = pattern
        return x
    else:
        x = torch.zeros(( X, Y, C), dtype=torch.float32, device=device)
        x[
        X // 2 - starting_area // 2: X // 2 + starting_area // 2 + 1 if starting_area % 2 != 0 else X // 2 +
                                                                                                             starting_area // 2,
        Y // 2 - starting_area // 2: Y // 2 + starting_area // 2 + 1 if starting_area % 2 != 0 else Y // 2 +
                                                                                                             starting_area // 2,:] = torch.rand((starting_area,starting_area,C), device=device)

        if has_food:
            x_food = get_food_pos(X, Y, num_spots=num_food, food_size=food_size) * 1
            x = torch.cat((x_food, x), dim=-1)
        return x

def cs_constructor(C, n, config, is_flow , from_saved):
    if not is_flow :
        return None, None, n
    elif is_flow and from_saved:
        return config["c0"], config["c1"], len(config["kernels"])

    c2 = []
    for j in range(n):
       c2.append(int((C/n)*j))
    M = np.ones((C,C), dtype=int)*n
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
    print(c1)
    return c0, c1,c2, int(M.sum())

def adjust_params(nca, c0, c1,pk, device="cuda:0"):
    if nca.__class__ != Lenia_Flow:
        return nca

    if nca.__class__ == Lenia_Flow:
        nca.c0 = c0
        nca.c1 = c1
        nca.h = torch.stack([torch.tensor(k["h"], device=device)for k in nca.kernels], dim = 0)
        nca.s = torch.stack([torch.tensor(k["s"], device=device)for k in nca.kernels], dim = 0)
        nca.m = torch.stack([torch.tensor(k["m"], device=device)for k in nca.kernels], dim = 0)
        nca.pk = pk
        return nca

def get_food_pos(X, Y, num_spots =100, food_size = 5):
    places = [[random.randint(food_size, X-food_size), random.randint(food_size, Y-food_size)] for _ in range(num_spots)]
    x = torch.zeros((X,Y,1), device="cuda:0")
    for place in places:
        x[place[0]- food_size: place[0]+ food_size, place[1]-food_size:place[1]+ food_size] = 1
    return x

def render(grid, with_food):

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


    img = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_AREA)
    cv2.imshow("Lenia", img)
    cv2.waitKey(1)