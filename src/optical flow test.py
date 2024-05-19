from torchPIV.PIVbackend import ToTensor, PIVDataset
from optical_flow_2021 import optical_flow, plot_velocity

import torch
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.animation as animation


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

if device == "cuda":
    torch.cuda.empty_cache()

image_dim = (1080, 1920)  # height, width
grid_spacing = 50

dataset = PIVDataset("../Test Data/cilinder_PROCESSED", ".jpg",
                     "sequential", transform=ToTensor(dtype=torch.uint8))
dataset = [dataset[0]]
res = np.zeros((len(dataset), 2, *image_dim))
curls = np.zeros((len(dataset), *image_dim))

i = 0
for img1, img2 in tqdm(dataset):
    a, b = img1.to(device), img2.to(device)
    u, v = optical_flow(a, b, 100, 1000, None)

    res[i, 0], res[i, 1] = u.cpu(), v.cpu() * -1  # flip v for correct image coordinate directions

    padding = (1, 1, 1, 1)
    u_pad, v_pad = torch.nn.functional.pad(u, padding), torch.nn.functional.pad(u, padding)

    # curl/vorticity, https://www.youtube.com/watch?v=JFWqCQHg-Hs (from 30:25 mark)
    dfydx = u_pad[2:, 1:-1] - u_pad[0:-2, 1:-1]
    dfxdy = v_pad[1:-1, 2:] - v_pad[1:-1, 0:-2]

    curls[i] = (dfydx - dfxdy).cpu()

    i += 1

plot_velocity(res, grid_spacing=grid_spacing)

# fig, ax = plt.subplots()
#
# min_curl, max_curl = np.min(curls), np.max(curls)
# largest = max_curl if max_curl > abs(min_curl) else abs(min_curl)
#
# stream = ax.streamplot(x, y, vx0, vy0, color='black')
#
# image = ax.imshow(curls[0], vmin=-largest, vmax=largest, cmap='bwr')
# fig.colorbar(image, ax=ax)
#
# def update(idx):
#     image.set_data(curls[idx])
#     ax.set_title(f"t = {idx/240:.3f} s")
#     return image,
#
# ani = animation.FuncAnimation(fig=fig, func=update, frames=len(dataset), interval=100)
#
# # writer = animation.PillowWriter(fps=5)
# # ani.save('../Test Data/cilinder.gif', writer=writer)
#
# plt.show()
