from torchPIV.PIVbackend import ToTensor, PIVDataset
from optical_flow_2021 import HornSchunck

import torch


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

if device == "cuda":
    torch.cuda.empty_cache()

image_dim = (1080, 1920)  # height, width
grid_spacing = 20

dataset = PIVDataset("../Test Data/cilinder_PROCESSED", ".jpg",
                     "sequential", transform=ToTensor(dtype=torch.uint8))

p = HornSchunck([dataset[0]], alpha=100, num_iter=100, device=device)
p.optical_flow()
p.plot_velocity(grid_spacing=grid_spacing)
