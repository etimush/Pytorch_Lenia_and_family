from FlowLenia import Lenia_Classic, Lenia_Diff, Lenia_Flow
import cv2
import torch
from utils import *
import configs
import uuid


####Setup####
device = "cuda:0"
cv2.namedWindow("Lenia", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Lenia", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
full_screen_sim_x, full_screen_sim_y = 480, 270  # 1920,1080
config_type = None
has_food = True
s_uuid = str(uuid.uuid4())
is_flow = True
config = configs.load_saved_file(config_type)


############

###Set up Lenia#####

C, dt, k, n,pk, has_food = get_setting(config, has_food)

c0, c1,c2, n = cs_constructor(C, n, config, is_flow=is_flow, from_saved=config_type is not None)
kernels = kernels_config_constructor_df(config, n, C)
nca = Lenia_Flow(C, dt, k, kernels, device, full_screen_sim_x, full_screen_sim_y,has_food ,mode="soft").to(device)
nca = adjust_params(nca, c0, c1,pk)
x = get_starting_pattern(config, 40, C, full_screen_sim_x, full_screen_sim_y, has_food)


###################
app = App(kernels, dt, C, k, s_uuid, is_flow, c0, c1, nca.__class__.__name__, pk, has_food)  # need to save the c0 and c1 config

#### Display Info#####
"""kernels = nca.Ks[0].squeeze().cpu().clone().detach().float().numpy()[ full_screen_sim_y//2 - k: full_screen_sim_y//2 +k, full_screen_sim_x//2 - k : full_screen_sim_x//2 + k]
cv2.imshow("k", kernels)
cv2.waitKey(0)"""
####################

#### Simulation ####
for i in range(1000000):
    print(i)
    with torch.no_grad():
        x = nca(x)
        grid = x.cpu().clone().permute((1, 0, 2)).detach().float().numpy()
    """if i != 0 and i% 500 == 0:
        nf = get_food_pos(full_screen_sim_x,full_screen_sim_y,100, 2)
        x[:,:,0:1] += nf"""
    render(grid, has_food)
