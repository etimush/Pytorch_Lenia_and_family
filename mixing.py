import utils
from utils import *
import configs

#cv2.namedWindow("Lenia", cv2.WND_PROP_FULLSCREEN)
#cv2.setWindowProperty("Lenia", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
refresh = False
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

def event(event,x,y,flags,param):
    global refresh
    if event == cv2.EVENT_MBUTTONDOWN:
        refresh = True
###Set up Lenia. Do not change anything here #####

t_crit = utils.dichotomy(1000,ca_overCritical,ca_subCritical,9,0.034, keys, "mixed", save_path,full_screen_sim_x, full_screen_sim_y,mode,ca_type,has_food, is_flow,starting_area, full_noise=full)
def mass_conservation_step(self, x_previous, Hs):
        """
        Implements the mass diffusion with A/R scores as per the equations
        using Hs as A/R scores.

        Args:
            x_previous: the mass from the *previous* time step.
            Hs: The A/R scores
        """
        H, W, C = x_previous.shape

        # Create neighbor indices (including self)
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
        rows = torch.arange(0, H, device=self.device).view(-1, 1)  # Column vector
        cols = torch.arange(0, W, device=self.device).view(1, -1)  # Row vector

        # Get Hs and mass from the *previous* step for each neighbor
        neighbors_Hs = []
        neighbors_masses = []
        for row_offset, col_offset in offsets:
            neighbor_rows = torch.remainder(rows + row_offset, H)
            neighbor_cols = torch.remainder(cols + col_offset, W)
            neighbors_Hs.append(Hs[neighbor_rows, neighbor_cols, :])  # (H,W,C)
            neighbors_masses.append(x_previous[neighbor_rows, neighbor_cols, :])  # (H,W,C)

        neighborhood_Hs = torch.stack(neighbors_Hs, dim=0)  # (9,H,W,C)
        neighborhood_masses = torch.stack(neighbors_masses, dim=0)  # (9,H,W,C)

        # 1. Calculate E for each cell based on its neighbours masses.
        exp_neighbor_masses = torch.exp(neighborhood_masses)  # (9,H,W,C)
        E = exp_neighbor_masses.sum(dim=0, keepdim=True)  # (1,H,W,C)

        # 2. Calculate the redistribution using the actual values and equations (exp of AR scores divided by E and times neighbour mass)
        exp_Hs = torch.exp(neighborhood_Hs)
        redistributions = (exp_Hs / E) * neighborhood_masses

        # 3. Calculate the new mass: Sum the redistributions
        new_mass = redistributions.sum(dim=0)  # (H,W,C)

        return new_mass
print(t_crit)
params = configs.load_saved_file("mixed/Lenia_Diff_mixed.pkl")

nca, x, saver = construct_ca("mix", save_path, full_screen_sim_x, full_screen_sim_y, mode, params,
                                           ca_type,
                                           has_food=has_food, is_flow=is_flow, starting_area=starting_area, vary=False, full_noise=full)


while True:
    if refresh:
        refresh = False
        x =  get_starting_pattern(None,starting_area, x.shape[2], x.shape[0], x.shape[1], has_food = has_food,full=full, wavelength=saver.dict["R"])
    grid = x.cpu().clone().permute((1, 0, 2)).detach().float().numpy()
    render(grid, has_food, waitkey=1, multiplier=3)
    x = nca(x)
    cv2.setMouseCallback('Lenia', event)
