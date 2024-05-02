
from tkinter import *
#from Lenia import Lenia, Lenia3, Lenia2
from FlowLenia import Lenia
import cv2
import torch
import tkinter as tk
import threading
from evol_utils import mutate, reproduce
import pyglet



device = "cpu"




class App(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.start()
    def callback(self):
        self.root.quit()
    def saveFile(self):
        torch.save(nca.state_dict(), f"./supervised_saved/saved_model_15")
    def run(self):
        self.root = tk.Tk()
        self.root.protocol("WM_DELETE_WINDOW", self.callback)
        button = Button(text="save", command=self.saveFile)
        button.pack()
        self.root.mainloop()

torch.backends.cudnn.benchmark = True
out_features, dt, k ,n , kpc = 3,1/2, 51,3,1
nca = Lenia(out_features,dt,k,n,kpc, device).to(device)
#nca2 = Lenia(out_features,dt,k,n,kpc).eval().to(device).requires_grad_(False)
full_screen_sim_x, full_screen_sim_y = 384, 216 #1920,1080
phone_screen_sim_x, phone_screen_sim_y = 180,360  # 607, 1080
#440
#nca.load_state_dict(torch.load("evo_search_models/lenia29"))



#evolutionary stuff##########################
"""nca2.load_state_dict(torch.load("evo_models/Week_1_m4"))
nca = reproduce(nca,nca2)
nca = mutate(nca)"""
############################################
"""cv2.namedWindow("Lenia", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Lenia", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)"""
cv2.namedWindow("Lenia", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Lenia", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
rand_locs1 = torch.randint(0, 384, (1,100000))
rand_locs2 = torch.randint(0, 216, (1,100000))
rand_locs = torch.cat((rand_locs2,rand_locs1), dim=0)
x= torch.zeros((1,out_features,full_screen_sim_y,full_screen_sim_x),dtype=torch.float32,device = device)
random_inint = torch.rand((1,out_features,52,52),dtype=torch.float32)
x[:,:,(x.shape[-2]//2) - (random_inint.shape[-2]//2):(x.shape[-2]//2) + (random_inint.shape[-2]//2),(x.shape[-1]//2) - (random_inint.shape[-1]//2):(x.shape[-1]//2) + (random_inint.shape[-1]//2)] = random_inint

app = App()
v = True





for i in range(10000):
    with torch.no_grad():
        x = nca(x)
        print(x.shape)
        grid = x.cpu().clone().permute((0, 2, 3, 1)).detach().float().numpy()

    img = cv2.resize(grid[0, :, :, :], (1920, 1080), interpolation=cv2.INTER_AREA)
    cv2.waitKey(1)
    cv2.imshow("Lenia", img)



for key in nca.state_dict():
    print(key, nca.state_dict()[key])