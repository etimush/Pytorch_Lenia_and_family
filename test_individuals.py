""" Code based on lenia tutorials and flow lenia writen in Pytorch by Etienne Guichard
    This will create a full screen image and a save button, you need to tab out to the save button
    Parameters are not saved raw, a set of smaller parameters is saved and used to reconstruct the state of the CA """
import utils
from utils import *
import configs
import uuid

refresh = False
####Setup. Here is what you change####
device = "cuda:0"  # <-- set to cpu if no gpu
#cv2.namedWindow("Lenia", cv2.WND_PROP_FULLSCREEN)
#cv2.setWindowProperty("Lenia", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
starting_area = 50  # <-- size of random starting area
full_noise = False  # <-- if starting area covers full screen or not, overides size
full_screen_sim_x, full_screen_sim_y = int(300), int(300)  # <-dims, the bigger the slower
config_type = "mixed/Lenia_Diff_whirlwind.pkl" # <-- None for random, file name for a parameter set saved on file
has_food = False  # <-- only works on flow lenia, set to true for food
s_uuid = str(uuid.uuid4())
is_flow = False  # <-- set to true if using flow lenia, allows for some extra parameters
config = configs.load_saved_file(config_type)  # <-- config.load_saved_files for loading from a file, config.Name for loading know lenia creatrures, check configs for names
ca_type = Lenia_Diff  # <-- Lenia_Flow for flow lenia, Lenia_classic for classic lenia, Lenia_Diff for differentiable Lenia
save_path = "./base/"  # <-- change for prefered path to save parameters
mode = "hard"  # <-- hard for hard clip, soft for soft clip

""" Variables for random parameter generation when not loading form file
    These variables give the range to or choice for the parameter constructor to select from
    If a range is not wanted set these to a single value """

has_food_range = [1, 2, 3, 4]  # <-- number of possible channels channels if food is available
no_food_range = [1,3]  # <-- number of possible channels if no food is available
dt_range = [10]  # <-- posible range of for integration factor dt = 1/choice(dt_range)
kernel_range = [31]  # <-- range for kernel size
n_range = [9]  # <-- range for number of kenels
pk_choice = ["dense"]  # <- choice for wether sparse or dense kernels

############

###Set up Lenia. Do not change anything here #####

nca, x, saver = utils.construct_ca(s_uuid, save_path, full_screen_sim_x, full_screen_sim_y, mode, config, ca_type,
                                   has_food=has_food, is_flow=is_flow, starting_area=starting_area,
                                   full_noise=full_noise, has_food_range=has_food_range, no_food_range=no_food_range,
                                   dt_range=dt_range, kernel_range=kernel_range,n_range=n_range,pk_choice=pk_choice )

dict = saver.save_to_dict()
################### Save Button ##############################
app = App(dict, save_path, s_uuid, nca.__class__.__name__)  # need to save the c0 and c1 config
def event(event,x,y,flags,param):
    global refresh
    if event == cv2.EVENT_MBUTTONDOWN:
        refresh = True
#### Simulation ####
while True:
    if refresh:
        refresh = False
        x =  get_starting_pattern(None,starting_area, x.shape[2], x.shape[0], x.shape[1], has_food = has_food,full=full_noise, wavelength=dict["R"])
    x = nca(x)
    #print(x.mean())
    grid = x.cpu().clone().permute((1, 0, 2)).detach().float().numpy()

    render(grid, has_food, waitkey=1, multiplier=3)
    cv2.setMouseCallback('Lenia', event)
