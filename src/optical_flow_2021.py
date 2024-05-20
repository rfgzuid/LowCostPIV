"""
PIV method based on
    "Extracting turbulence parameters of smoke via video analysis" (2021), https://doi.org/10.1063/5.0059326
Horn-Schunck optical flow (https://en.wikipedia.org/wiki/Horn-Schunck_method)

Torch adaptation of pure python implementation
    https://github.com/scivision/Optical-Flow-LucasKanade-HornSchunck/blob/master/HornSchunck.py
"""

import torch
from torch.utils.data import Dataset
from torch.nn.functional import conv2d, pad

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from tqdm import tqdm
import os
import cv2


class HornSchunck:
    def __init__(self, dataset: Dataset, alpha: float, num_iter: int, device: torch.device):
        self.dataset = dataset
        self.device = device

        self.img_shape = dataset[0][0].shape
        self.alpha = alpha
        self.num_iter = num_iter

        self.eps = None

        self.velocities = torch.zeros((len(self.dataset), 2, *self.img_shape)).to(self.device)

        if device not in ['cpu', 'cuda']:
            raise ValueError("Please specify either cpu or cuda")
        elif device == 'cuda' and not torch.cuda.is_available():
            raise ValueError("No support for cuda available")

    def optical_flow(self):
        for idx, (a, b) in tqdm(enumerate(self.dataset), total=len(self.dataset)):
            a, b = a.to(self.device), b.to(self.device)
            a, b = a[None, None, :, :].float(), b[None, None, :, :].float()

            u, v = torch.zeros_like(a), torch.zeros_like(a)

            Ix, Iy, It = self.compute_derivatives(a, b)

            avg_kernel = torch.tensor([[[[1/12, 1/6, 1/12],
                                         [1/6, 0, 1/6],
                                         [1/12, 1/6, 1/12]]]], dtype=torch.float32, device=self.device)

            for i in range(self.num_iter):
                u_avg = conv2d(u, avg_kernel, padding=1)
                v_avg = conv2d(v, avg_kernel, padding=1)

                der = (Ix * u_avg + Iy * v_avg + It) / (self.alpha ** 2 + Ix ** 2 + Iy ** 2)

                u_new = u_avg - Ix * der
                v_new = v_avg - Iy * der

                # MSE early stopping https://www.ipol.im/pub/art/2013/20/article.pdf
                delta = torch.sum((u_new - u) ** 2) + torch.sum((v_new - v) ** 2)
                delta /= a.shape[-2] * a.shape[-1]

                if self.eps is not None and delta < self.eps:
                    print('Early stopping', i)
                    break

                u, v = u_new, v_new

            self.velocities[idx, 0], self.velocities[idx, 1] = u.squeeze(), v.squeeze()

    @staticmethod
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

    def plot_velocity(self, grid_spacing: int = 50, filename: str | None = None):
        num_frames, _, height, width = self.velocities.shape
        velocities = self.velocities.cpu().numpy()

        fig, ax = plt.subplots()

        abs_velocities = np.sqrt(velocities[:, 0] ** 2 + velocities[:, 1] ** 2)
        min_abs, max_abs = np.min(abs_velocities), np.max(abs_velocities)

        u0, v0 = velocities[0]

        num_x, num_y = width // grid_spacing, height // grid_spacing

        # https://stackoverflow.com/questions/24116027/slicing-arrays-with-meshgrid-array-indices-in-numpy
        x, y = np.meshgrid(np.linspace(grid_spacing, grid_spacing * num_x - grid_spacing, num_x + 1),
                           np.linspace(grid_spacing, grid_spacing * num_y - grid_spacing, num_y + 1))
        xx, yy = x[...].astype(np.uint16), y[...].astype(np.uint16)

        vx0, vy0 = u0[yy, xx], v0[yy, xx]
        vectors = ax.quiver(x, y, vx0, vy0, color='black', scale=0.1, units='xy')

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

        if filename is not None:
            writer = animation.PillowWriter(fps=5)
            ani.save(f'../Test Data/{filename}', writer=writer)

        plt.show()



class OpticalDataset(Dataset):
    def __init__(self, folder, device):
        # assume the files are sorted and all have the correct file type
        filenames = [os.path.join(folder, name) for name in os.listdir(folder)]
        self.img_pairs = list(zip(filenames[:-1], filenames[1:]))

        self.device = device

    def __len__(self):
        return len(self.img_pairs)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:

        if torch.is_tensor(index):
            index = index.tolist()

        pair = self.img_pairs[index]

        img_b = cv2.imread(pair[1], cv2.IMREAD_GRAYSCALE)
        img_a = cv2.imread(pair[0], cv2.IMREAD_GRAYSCALE)

        cv2.imshow('before', img_b)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # https://wmich.edu/sites/default/files/attachments/u883/2017/Open_Optical_Flow_Paper_v1.pdf
        img_a = cv2.GaussianBlur(img_a, (5, 5), 0.6*5)
        img_b = cv2.GaussianBlur(img_b, (5, 5), 0.6*5)

        img_a = torch.tensor(img_a, dtype=torch.float32, device=self.device)
        img_b = torch.tensor(img_b, dtype=torch.float32, device=self.device)

        # illumination correction https://link-springer-com.tudelft.idm.oclc.org/article/10.1007/s00348-015-2036-1
        # https://github.com/Tianshu-Liu/OpenOpticalFlow/blob/master/correction_illumination.m
        a_mean, b_mean = torch.mean(img_a).item(), torch.mean(img_b).item()
        img_b *= (a_mean / b_mean)  # global illumination correction

        # N = 1  # local illumination correction (N must be odd because of image padding in convolution)
        # mean_filter = torch.ones((1, 1, N, N), device=self.device) / (N**2)
        #
        # padding = (int(N/2), int(N/2), int(N/2), int(N/2))
        # img_a_pad, img_b_pad = pad(img_a, padding)[None, None, :, :], pad(img_b, padding)[None, None, :, :]
        #
        # local_diffs = conv2d(img_a_pad, mean_filter) - conv2d(img_b_pad, mean_filter)
        #
        # cv2.imshow('local', local_diffs.squeeze().cpu().numpy())
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        #
        # img_b += local_diffs.squeeze()
        #
        cv2.imshow('after', img_b.to(torch.uint8).cpu().numpy())
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return img_a.to(torch.uint8), img_b.to(torch.uint8)
