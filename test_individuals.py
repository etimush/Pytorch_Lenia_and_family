""" Code based on lenia tutorials and flow lenia writen in Pytorch by Etienne Guichard
    This will create a full screen image and a save button, you need to tab out to the save button
    Parameters are not saved raw, a set of smaller parameters is saved and used to reconstruct the state of the CA """

from utils import *
import configs
import uuid


####Setup. Here is what you change####
device = "cuda:0" #<-- set to cpu if no gpu
cv2.namedWindow("Lenia", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Lenia", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
starting_area = 100 #<-- size of random starting area
full_screen_sim_x, full_screen_sim_y = int(480), int(270)  # <-dims, the bigger the slower
config_type = None # <-- None for random, file name for a parameter set saved on file
has_food = False # <-- only works on flow lenia, set to true for food
s_uuid = str(uuid.uuid4())
is_flow = True # <-- set to true if using flow lenia, allows for some extra parameters
config = configs.load_saved_file(config_type) #<-- config.load_saved_files for loading from a file, config.Name for loading know lenia creatrures, check configs for names
ca_type = Lenia_Diff #<-- Lenia_Flow for flow lenia, Lenia_classic for classic lenia, Lenia_Diff for extended lenia
save_path = "./supervised_saved/" #<-- change for prefered path to save parameters
mode = "hard" #<-- hard for hard clip, soft for soft clip


""" Variables for random parameter generation when not loading form file
    These variables give the range to or choice for the parameter constructor to select from
    If a range is not wanted set these to a single value """

has_food_range = [1,2,3,4] #<-- number of possible channels channels if food is available
no_food_range = [1,2,3] #<-- number of possible channels if no food is available
dt_range = [4,10] #<-- posible range of for integration factor
kernel_range = [2,25] #<-- range for kernel size
n_range = [3,15] #<-- range for number of kenels
pk_choice =["dense", "sparse"] #<- choice for wether sparse or dense kernels

############

###Set up Lenia. Do not change anything here #####

C, dt, k, n,pk, has_food = get_setting(config, has_food,has_food_range = has_food_range, no_food_range=no_food_range, dt_range=dt_range, kernel_range=kernel_range,n_range=n_range, pk_choice=pk_choice )
c0, c1, n = cs_constructor(C, n, config, is_flow=is_flow, from_saved=config_type is not None, has_food=has_food)
kernels = kernels_config_constructor_df(config, n, C)
C = C +has_food if has_food is not None else C
nca = ca_type(C , dt, k, kernels, device, full_screen_sim_x, full_screen_sim_y ,mode=mode, has_food=has_food).to(device).eval()
nca = adjust_params(nca, c0, c1,pk)
x = get_starting_pattern(config, starting_area, C - has_food if has_food is not None else C, full_screen_sim_x, full_screen_sim_y, has_food, num_food=1500)


################### Save Button ##############################
app = App(kernels, dt, C - has_food if has_food is not None else C, k, s_uuid, is_flow, c0, c1, nca.__class__.__name__, pk, has_food, save_path)  # need to save the c0 and c1 config


#### Simulation ####
while True:
    x = nca(x)
    grid = x.cpu().clone().permute((1, 0, 2)).detach().float().numpy()
    render(grid, has_food)

