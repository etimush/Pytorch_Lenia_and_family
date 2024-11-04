import utils
from utils import *
import configs
import uuid
####Setup. Here is what you change####
device = "cuda:0" #<-- set to cpu if no gpu
#cv2.namedWindow("Lenia", cv2.WND_PROP_FULLSCREEN)
#cv2.setWindowProperty("Lenia", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
starting_area = 50 #<-- size of random starting area
full_noise = True#<-- if starting area covers full screen or not, overides size
full_screen_sim_x, full_screen_sim_y = int(150), int(150)  # <-dims, the bigger the slower
config_type = None # <-- None for random, file name for a parameter set saved on file
has_food = False # <-- only works on flow lenia, set to true for food
s_uuid = "prior"
is_flow = False # <-- set to true if using flow lenia, allows for some extra parameters
config = configs.load_saved_file(config_type) #<-- config.load_saved_files for loading from a file, config.Name for loading know lenia creatrures, check configs for names
ca_type = Lenia_Diff #<-- Lenia_Flow for flow lenia, Lenia_classic for classic lenia, Lenia_Diff for differentiable Lenia
save_path = "./base/" #<-- change for prefered path to save parameters
mode = "hard" #<-- hard for hard clip, soft for soft clip
points_to_search  = 100
n_steps = 500
thresh = 0.01

found = False

""" Variables for random parameter generation when not loading form file
    These variables give the range to or choice for the parameter constructor to select from
    If a range is not wanted set these to a single value """

has_food_range = [1,2,3,4] #<-- number of possible channels channels if food is available
no_food_range = [3] #<-- number of possible channels if no food is available
dt_range = [5,10] #<-- posible range of for integration factor dt = 1/choice(dt_range)
kernel_range = [25,31] #<-- range for kernel size
n_range = [9,12] #<-- range for number of kenels
pk_choice =["sparse","dense"] #<- choice for wether sparse or dense kernels
mean = 0

while not found:

    live_count = 0
    nca, x, saver = utils.construct_ca(s_uuid, save_path, full_screen_sim_x, full_screen_sim_y, mode, config, ca_type,
                                       has_food=has_food, is_flow=is_flow, starting_area=starting_area,
                                       full_noise=full_noise, has_food_range=has_food_range,
                                       no_food_range=no_food_range,
                                       dt_range=dt_range, kernel_range=kernel_range, n_range=n_range,
                                       pk_choice=pk_choice)
    saver.dict["pattern"] = x
    dict = saver.save_to_dict()

    for i in range(points_to_search):
        print(i)
        nca, x, _ = utils.construct_ca(s_uuid, save_path, full_screen_sim_x, full_screen_sim_y, mode, dict=dict,
                                           ca_type=ca_type,vary=True,
                                           has_food=has_food, is_flow=is_flow, starting_area=starting_area,
                                           full_noise=full_noise, has_food_range=has_food_range,
                                           no_food_range=no_food_range,
                                           dt_range=dt_range, kernel_range=kernel_range, n_range=n_range,
                                           pk_choice=pk_choice)
        for _ in range(n_steps):

            x = nca(x)
            #grid = x.cpu().clone().permute((1, 0, 2)).detach().float().numpy()
            #render(grid, has_food, waitkey=0)
        mean = x.mean()
        if mean > thresh:
            live_count+=1

    perc_alive = live_count/points_to_search
    print(perc_alive)
    if 0.6 > perc_alive > 0.4:
        saver.save()
        found = True