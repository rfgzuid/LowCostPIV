"""
PIV method based on
    "Extracting turbulence parameters of smoke via video analysis" (2021), https://doi.org/10.1063/5.0059326
Horn-Schunck optical flow (https://en.wikipedia.org/wiki/Horn-Schunck_method)

Torch adaptation of pure python implementation
    https://github.com/scivision/Optical-Flow-LucasKanade-HornSchunck/blob/master/HornSchunck.py
"""

import torch
from torch.nn.functional import conv2d, pad
from tqdm import tqdm


def optical_flow(img1, img2, alpha, num_iter, eps):
    device = img1.device

    a, b = img1[None, None, :, :].float(), img2[None, None, :, :].float()
    u, v = torch.zeros_like(a), torch.zeros_like(a)

    Ix, Iy, It = compute_derivatives(a, b)

    avg_kernel = torch.tensor([[[[1/12, 1/6, 1/12],
                                 [1/6, 0, 1/6],
                                 [1/12, 1/6, 1/12]]]], dtype=torch.float32, device=device)

    for i in range(num_iter):
        u_avg = conv2d(u, avg_kernel, padding=1)
        v_avg = conv2d(v, avg_kernel, padding=1)

        der = (Ix * u_avg + Iy * v_avg + It) / (alpha ** 2 + Ix ** 2 + Iy ** 2)

        u_new = u_avg - Ix * der
        v_new = v_avg - Iy * der

        # MSE early stopping https://www.ipol.im/pub/art/2013/20/article.pdf
        delta = torch.sum((u_new - u) ** 2) + torch.sum((v_new - v) ** 2)
        delta /= a.shape[-2] * a.shape[-1]

        if eps is not None and delta < eps:
            # print('Early stopping', i)
            break

        u, v = u_new, v_new
    return u.squeeze(), v.squeeze()


def compute_derivatives(img1: torch.Tensor, img2: torch.Tensor):
    device = img1.device

    kernel_x = torch.tensor([[[[-1 / 4, 1 / 4],
                               [-1 / 4, 1 / 4]]]], dtype=torch.float32, device=device)
    kernel_y = torch.tensor([[[[-1 / 4, -1 / 4],
                               [1 / 4, 1 / 4]]]], dtype=torch.float32, device=device)
    kernel_t = torch.ones((1, 1, 2, 2), dtype=torch.float32, device=device) / 4

    padding = (0, 1, 0, 1)  # add a column right and a row at the bottom
    img1, img2 = pad(img1, padding), pad(img2, padding)

    fx = conv2d(img1, kernel_x) + conv2d(img2, kernel_x)
    fy = conv2d(img1, kernel_y) + conv2d(img2, kernel_y)
    ft = conv2d(img1, -kernel_t) + conv2d(img2, kernel_t)

    return fx, fy, ft
