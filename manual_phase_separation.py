import utils
from utils import *
import configs
import uuid
cv2.namedWindow("Lenia", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Lenia", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
is_flow = False
device = "cuda:0"
has_food = False
mode = "hard"
mix_channel_destination = False #<- if true channel destinations are also mixed, if false it eitherpicks the channel destination you specify or random
channel_destination_priority = "sub" #<-- options = ["sub', "over", "random"]
full_screen_sim_x, full_screen_sim_y = int(150), int(150)  # <-dims, the bigger the slower
ca_type = Lenia_Diff #<-- Lenia_Flow for flow lenia, Lenia_classic for classic lenia, Lenia_Diff for extended lenia
starting_area = 50 #<-- size of random starting area
full = False
s_uuid = "sub"
base_path = "./base/Lenia_Diff_338064c9-f15d-42a1-adac-7bc365c0f4ea.pkl"
save_path = "./mixing_save/"
points_to_search  = 200
n_steps = 500
base_config = configs.load_saved_file(base_path)


###Set up Lenia. Do not change anything here #####
nca, x, saver = utils.construct_ca(s_uuid, save_path, full_screen_sim_x, full_screen_sim_y, mode, base_config, ca_type,
                                   has_food=has_food, is_flow=is_flow, starting_area=starting_area, vary=True)
saver.dict["pattern"] = x
dict = saver.save_to_dict()

################### Save Button ##############################
app = App(dict,save_path, s_uuid,nca.__class__.__name__)  # need to save the c0 and c1 config


while True:
    x = nca(x)
    grid = x.cpu().clone().permute((1, 0, 2)).detach().float().numpy()
    render(grid, has_food)





