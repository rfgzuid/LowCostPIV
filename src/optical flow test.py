from torchPIV.PIVbackend import ToTensor, PIVDataset
from optical_flow_2021 import optical_flow

import torch
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.animation as animation


# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

dataset = PIVDataset("../Test Data/plume simulation_PROCESSED", ".jpg",
                     "sequential", transform=ToTensor(dtype=torch.uint8))
res = np.zeros((len(dataset), 2, 2048, 2048))

i = 0
for img1, img2 in tqdm(dataset):
    u, v = optical_flow(img1, img2, 100., 10)
    res[i, 0], res[i, 1] = u, v
    i += 1

fig, ax = plt.subplots()

abs_velocities = np.sqrt(res[:, 0] ** 2 + res[:, 1] ** 2)
min_abs, max_abs = np.min(abs_velocities), np.max(abs_velocities)

image = ax.imshow(abs_velocities[0], vmin=min_abs, vmax=max_abs, cmap='jet')

def update(idx):
    image.set_data(abs_velocities[idx])
    return image,

ani = animation.FuncAnimation(fig=fig, func=update, frames=len(dataset), interval=1000/60)

# writer = animation.PillowWriter(fps=15,
#                                 metadata=dict(artist='Me'),
#                                 bitrate=1800)
# ani.save('Test Data/plume.gif', writer=writer)

plt.show()
