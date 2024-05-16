"""
PIV method based on
    "Extracting turbulence parameters of smoke via video analysis" (2021), https://doi.org/10.1063/5.0059326
Horn-Schunck optical flow (https://en.wikipedia.org/wiki/Horn-Schunck_method)

Torch adaptation of pure python implementation
    https://github.com/scivision/Optical-Flow-LucasKanade-HornSchunck/blob/master/HornSchunck.py
"""

import torch
from torch.nn.functional import conv2d
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def optical_flow(img1: torch.Tensor, img2: torch.Tensor, alpha: float, num_iter: int, eps: float | None):
    device = img1.device

    img1, img2 = torch.tensor(img1)[None, None, :, :].float(), torch.tensor(img2)[None, None, :, :].float()
    u, v = torch.zeros_like(img1, device=device), torch.zeros_like(img1, device=device)  # initialize velocities to 0

    I_x, I_y, I_t = compute_derivatives(img1, img2)

    avg_kernel = torch.tensor([[[[1/12, 1/6, 1/12],
                               [1/6, 0, 1/6],
                               [1/12, 1/6, 1/12]]]], dtype=torch.float32, device=device)
    deltas = []

    us, vs = torch.zeros((100, *u.shape)), torch.zeros((100, *v.shape))

    for i in range(num_iter):
        u_avg = conv2d(u, avg_kernel, padding=1)
        v_avg = conv2d(v, avg_kernel, padding=1)

        der = (I_x * u_avg + I_y * v_avg + I_t) / (alpha ** 2 + I_x ** 2 + I_y ** 2)

        u_new = u_avg - I_x * der
        v_new = v_avg - I_y * der

        delta = torch.sum((u_new - u)**2) + torch.sum((v_new - v)**2)
        deltas.append(delta.item())

        if eps is not None and delta < eps:
            break

        if i % 100 == 0:
            us[i//100], vs[i//100] = u_new, v_new

        u, v = u_new, v_new

    return us.squeeze(), vs.squeeze(), deltas


def compute_derivatives(img1, img2):
    device = img1.device

    kernel_x = torch.tensor([[[[-1/4, 1/4],
                               [-1/4, 1/4]]]], dtype=torch.float32, device=device)
    kernel_y = torch.tensor([[[[-1/4, -1/4],
                               [1/4, 1/4]]]], dtype=torch.float32, device=device)
    kernel_t = torch.ones((1, 1, 2, 2), dtype=torch.float32, device=device) / 4

    fx = conv2d(img1, kernel_x, padding=1)[:, :, 1:, 1:] + conv2d(img2, kernel_x, padding=1)[:, :, 1:, 1:]
    fy = conv2d(img1, kernel_y, padding=1)[:, :, 1:, 1:] + conv2d(img2, kernel_y, padding=1)[:, :, 1:, 1:]
    ft = conv2d(img1, -kernel_t, padding=1)[:, :, 1:, 1:] + conv2d(img2, kernel_t, padding=1)[:, :, 1:, 1:]

    return fx, fy, ft
