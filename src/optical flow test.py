from torchPIV.PIVbackend import ToTensor, PIVDataset
from optical_flow_2021 import optical_flow

import torch
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.animation as animation


device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
print(device)

if device == "cuda":
    torch.cuda.empty_cache()

dataset = PIVDataset("../Test Data/Cilinder_PROCESSED", ".jpg",
                     "sequential", transform=ToTensor(dtype=torch.uint8))
res = np.zeros((100, 2, 1080, 1920))

i = 0
for img1, img2 in tqdm([dataset[0]]):
    a, b = img1.to(device), img2.to(device)
    us, vs, ds = optical_flow(a, b, 100., 10000, None)

    res[:, 0], res[:, 1] = us.cpu(), vs.cpu()
    i += 1

plt.plot(ds)
plt.yscale('log')
plt.show()

# abs_velocities = np.sqrt(res[:, 0] ** 2 + res[:, 1] ** 2)
# max_abs = np.max(abs_velocities)
#
# fig, ax = plt.subplots()
# u0, v0 = res[0]
#
# # https://stackoverflow.com/questions/24116027/slicing-arrays-with-meshgrid-array-indices-in-numpy
# x, y = np.meshgrid(np.linspace(50, 2000, 80, dtype=np.int16),
#                    np.linspace(50, 2000, 80, dtype=np.int16))
# xx, yy = x[...], y[...]
# vx0, vy0 = u0[yy, xx], v0[yy, xx]
#
# y = np.flip(y, axis=0)  # meshgrid to correct img coordinates
# vectors = ax.quiver(x, y, vx0, vy0, color='black', scale=1000/max_abs)
#
# def update(idx):
#     u, v = res[idx]
#     vx, vy = u[yy, xx], v[yy, xx]
#     vectors.set_UVC(vx, vy)
#     return vectors,
#
# ani = animation.FuncAnimation(fig=fig, func=update, frames=len(dataset))
# plt.show()

fig, ax = plt.subplots()

abs_velocities = np.sqrt(res[:, 0] ** 2 + res[:, 1] ** 2)
min_abs, max_abs = np.min(abs_velocities), np.max(abs_velocities)

image = ax.imshow(abs_velocities[0], vmin=min_abs, vmax=max_abs, cmap='jet')
fig.colorbar(image, ax=ax)

def update(idx):
    image.set_data(abs_velocities[idx])
    return image,

ani = animation.FuncAnimation(fig=fig, func=update, frames=100, interval=1000/60)
# ani = animation.FuncAnimation(fig=fig, func=update, frames=len(dataset), interval=1000)

# writer = animation.PillowWriter(fps=5,
#                                 metadata=dict(artist='Me'),
#                                 bitrate=1800)
# ani.save('../Test Data/plume optical 3.gif', writer=writer)

plt.show()
