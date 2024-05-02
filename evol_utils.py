import torch
import random
def reproduce(p1, p2):
    dominant_p = p1
    non_d_p = p2
    sd_dp = dominant_p.state_dict()
    sd_ndp = non_d_p.state_dict()
    for key in sd_dp:
        coin = random.random()
        sd_dp[key] = sd_dp[key] if coin <= 0.5 else sd_ndp[key]

    dominant_p.load_state_dict(sd_dp)
    return dominant_p

def mutate(self):

    state_dict = self.state_dict()

    for key in state_dict:
        mask = torch.rand_like(state_dict[key]) < 0.05
        mutation = (torch.randn_like(state_dict[key]))/ 100
        state_dict[key] = state_dict[key] + (mutation*mask)
    self.load_state_dict(state_dict)
    return self