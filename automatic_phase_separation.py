import utils
from utils import *
import configs
import uuid
#cv2.namedWindow("Lenia", cv2.WND_PROP_FULLSCREEN)
#cv2.setWindowProperty("Lenia", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
is_flow = False
device = "cuda:0"
has_food = False
mode = "hard"
mix_channel_destination = False #<- if true channel destinations are also mixed, if false it eitherpicks the channel destination you specify or random
channel_destination_priority = "sub" #<-- options = ["sub', "over", "random"]
full_screen_sim_x, full_screen_sim_y = int(150), int(150)  # <-dims, the bigger the slower
ca_type = Lenia_Diff #<-- Lenia_Flow for flow lenia, Lenia_classic for classic lenia, Lenia_Diff for extended lenia
starting_area = 50 #<-- size of random starting area
full = True
s_uuid = "sub"
base_path = "./base/Lenia_Diff_prior.pkl"
save_path = "./mixing_save/"
n_steps = 1000
base_config = configs.load_saved_file(None)
found_sub = False
found_over = False
thresh = 0.034
print(base_config)
i = 0
while not (found_over and found_sub):
    i+=1
    print(i)
###Set up Lenia. Do not change anything here #####
    nca, x, saver = utils.construct_ca(s_uuid, save_path, full_screen_sim_x, full_screen_sim_y, mode, base_config, ca_type,
                                       has_food=has_food, is_flow=is_flow, starting_area=starting_area, vary=base_config is not None, full_noise=full)
    #saver.dict["pattern"] = x


    for _ in range(n_steps):
        x = nca(x)
        #grid = x.cpu().clone().permute((1, 0, 2)).detach().float().numpy()
        #render(grid, has_food)

    mean = max(x.mean(dim = (0,1)))
    if (mean < thresh) and not found_sub:
        saver.i = "sub"
        saver.save()
        found_sub = True
    if (mean > thresh) and not found_over:
        saver.i = "over"
        saver.save()
        found_over = True

print("Both Found !!")