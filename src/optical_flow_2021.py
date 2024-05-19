"""
PIV method based on
    "Extracting turbulence parameters of smoke via video analysis" (2021), https://doi.org/10.1063/5.0059326
Horn-Schunck optical flow (https://en.wikipedia.org/wiki/Horn-Schunck_method)

Torch adaptation of pure python implementation
    https://github.com/scivision/Optical-Flow-LucasKanade-HornSchunck/blob/master/HornSchunck.py
"""

import torch
from torch.nn.functional import conv2d, pad

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def optical_flow(img1: torch.Tensor, img2: torch.Tensor, alpha: float, num_iter: int, eps: float | None):
    device = img1.device

    img1, img2 = torch.tensor(img1)[None, None, :, :].float(), torch.tensor(img2)[None, None, :, :].float()
    u, v = torch.zeros_like(img1, device=device), torch.zeros_like(img1, device=device)  # initialize velocities to 0

    I_x, I_y, I_t = compute_derivatives(img1, img2)

    avg_kernel = torch.tensor([[[[1/12, 1/6, 1/12],
                               [1/6, 0, 1/6],
                               [1/12, 1/6, 1/12]]]], dtype=torch.float32, device=device)

    for i in range(num_iter):
        u_avg = conv2d(u, avg_kernel, padding=1)
        v_avg = conv2d(v, avg_kernel, padding=1)

        der = (I_x * u_avg + I_y * v_avg + I_t) / (alpha ** 2 + I_x ** 2 + I_y ** 2)

        u_new = u_avg - I_x * der
        v_new = v_avg - I_y * der

        # MSE early stopping https://www.ipol.im/pub/art/2013/20/article.pdf
        delta = torch.sum((u_new - u)**2) + torch.sum((v_new - v)**2)
        delta /= img1.shape[-2] * img1.shape[-1]

        if eps is not None and delta < eps:
            print('Early stopping', i)
            break

        u, v = u_new, v_new

    return u.squeeze(), v.squeeze()


def compute_derivatives(img1, img2):
    device = img1.device

    kernel_x = torch.tensor([[[[-1/4, 1/4],
                               [-1/4, 1/4]]]], dtype=torch.float32, device=device)
    kernel_y = torch.tensor([[[[-1/4, -1/4],
                               [1/4, 1/4]]]], dtype=torch.float32, device=device)
    kernel_t = torch.ones((1, 1, 2, 2), dtype=torch.float32, device=device) / 4

    padding = (0, 1, 0, 1)  # add a column right and a row at the bottom
    img1, img2 = pad(img1, padding), pad(img2, padding)

    fx = conv2d(img1, kernel_x) + conv2d(img2, kernel_x)
    fy = conv2d(img1, kernel_y) + conv2d(img2, kernel_y)
    ft = conv2d(img1, -kernel_t) + conv2d(img2, kernel_t)

    return fx, fy, ft


def plot_velocity(velocities: np.ndarray, grid_spacing=50) -> None:
    num_frames, _, height, width = velocities.shape
    fig, ax = plt.subplots()

    abs_velocities = np.sqrt(velocities[:, 0] ** 2 + velocities[:, 1] ** 2)
    min_abs, max_abs = np.min(abs_velocities), np.max(abs_velocities)

    u0, v0 = velocities[0]

    num_x, num_y = width // grid_spacing, height // grid_spacing

    # https://stackoverflow.com/questions/24116027/slicing-arrays-with-meshgrid-array-indices-in-numpy
    x, y = np.meshgrid(np.linspace(grid_spacing, grid_spacing * num_x, num_x + 1),
                       np.linspace(grid_spacing, grid_spacing * num_y, num_y + 1))
    xx, yy = x[...].astype(np.uint16), y[...].astype(np.uint16)

    vx0, vy0 = u0[yy, xx], v0[yy, xx]
    vectors = ax.streamplot(x, y, vx0, vy0, color='black')  # , scale=10_000 / max_abs)

    image = ax.imshow(abs_velocities[0], vmin=min_abs, vmax=max_abs, cmap='turbo')
    fig.colorbar(image, ax=ax)

    def update(idx):
        image.set_data(abs_velocities[idx])

        u, v = velocities[idx]
        vx, vy = u[yy, xx], v[yy, xx]
        vectors.set_UVC(vx, vy)

        ax.set_title(f"t = {idx / 240:.3f} s")
        return image, vectors

    ani = animation.FuncAnimation(fig=fig, func=update, frames=velocities.shape[0], interval=100)

    # writer = animation.PillowWriter(fps=5)
    # ani.save('../Test Data/cilinder.gif', writer=writer)

    plt.show()
