from optical_flow_2021 import HornSchunck, OpticalDataset
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

if device == "cuda":
    torch.cuda.empty_cache()

dataset = OpticalDataset("../Test Data/cilinder_PROCESSED", (250, 750, 0, 1200), device)

p = HornSchunck([dataset[0]], alpha=100, num_iter=1000, device=device)
p.optical_flow()
p.plot_velocity(grid_spacing=15)
