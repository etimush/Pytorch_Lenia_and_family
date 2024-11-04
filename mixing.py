import utils
from utils import *
import configs

#cv2.namedWindow("Lenia", cv2.WND_PROP_FULLSCREEN)
#cv2.setWindowProperty("Lenia", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
is_flow = False
device = "cuda:0"
has_food = False
mode = "hard"

full_screen_sim_x, full_screen_sim_y = int(150), int(150)  # <-dims, the bigger the slower
ca_type = Lenia_Diff #<-- Lenia_Flow for flow lenia, Lenia_classic for classic lenia, Lenia_Diff for extended lenia
starting_area = 50 #<-- size of random starting area
full = True
save_path = "./mixed/"
sub_critical_path = "./mixing_save/Lenia_Diff_sub.pkl"
over_critical_path = "./mixing_save/Lenia_Diff_over.pkl"

ca_subCritical = configs.load_saved_file(sub_critical_path)
ca_overCritical = configs.load_saved_file(over_critical_path)
keys = ["m","s","w","a","b","h"]


###Set up Lenia. Do not change anything here #####

t_crit = utils.dichotomy(1000,ca_overCritical,ca_subCritical,9,0.0048311111119, keys, "mixed", save_path,full_screen_sim_x, full_screen_sim_y,mode,ca_type,has_food, is_flow,starting_area, full_noise=full)

print(t_crit)
params = configs.load_saved_file("mixed/Lenia_Diff_mixed.pkl")

nca, x, saver = construct_ca("mix", save_path, full_screen_sim_x, full_screen_sim_y, mode, params,
                                           ca_type,
                                           has_food=has_food, is_flow=is_flow, starting_area=starting_area, vary=False, full_noise=full)


while True:
    grid = x.cpu().clone().permute((1, 0, 2)).detach().float().numpy()
    render(grid, has_food, waitkey=1, multiplier=3)
    x = nca(x)
