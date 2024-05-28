from .matching import window_array, search_array, get_field_shape, block_match, get_x_y, correlation_to_displacement
from .optical_flow import optical_flow

from torch.nn.functional import conv2d, pad, grid_sample, interpolate
from torchvision.transforms import Resize, InterpolationMode
import torch

from torch.utils.data import Dataset
import os
import cv2

import matplotlib.pyplot as plt


class SIVDataset(Dataset):
    def __init__(self, folder: str):
        # assume the files are sorted and all have the correct file type
        filenames = [os.path.join(folder, name) for name in os.listdir(folder)]
        self.img_pairs = list(zip(filenames[:-1], filenames[1:]))

    def __len__(self):
        return len(self.img_pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        pair = self.img_pairs[index]
        img_a, img_b = cv2.imread(pair[0], cv2.IMREAD_GRAYSCALE), cv2.imread(pair[1], cv2.IMREAD_GRAYSCALE)
        return torch.tensor(img_a, dtype=torch.uint8), torch.tensor(img_b, dtype=torch.uint8)


class SIV:
    def __init__(self, folder: str, device: torch.device="cpu",
                 window_size: int=128, overlap: int=64, search_area: tuple[int, int, int, int]=(0, 0, 0, 0),
                 multipass: int=1, multipass_scale: float=2.,
                 dt: float=1/240) -> None:
        self.dataset = SIVDataset(folder=folder)
        self.device = device
        self.window_size, self.overlap, self.search_area = window_size, overlap, search_area
        self.multipass, self.multipass_scale = multipass, multipass_scale
        self.dt = dt

    def run(self, mode: int):
        for idx, data in enumerate(self.dataset):
            img_a, img_b = data
            img_a, img_b = img_a.to(self.device), img_b.to(self.device)

            n_rows, n_cols = get_field_shape(img_a.shape, self.window_size, self.overlap)

            u, v = torch.zeros((n_rows, n_cols)), torch.zeros((n_rows, n_cols))

            for k in range(self.multipass):
                scale = self.multipass_scale ** (k - self.multipass + 1)
                window_size, overlap = int(self.window_size * scale), int(self.overlap * scale)

                new_size = (round(img_a.shape[1] * scale), round(img_a.shape[0] * scale))
                resize = Resize(new_size, InterpolationMode.BILINEAR)
                a, b = resize(img_a[None, :, :]).squeeze(), resize(img_b[None, :, :]).squeeze()

                offset = torch.stack((u, v))

                x, y = get_x_y(a.shape, window_size, overlap)
                x, y = x.reshape(n_rows, n_cols), y.reshape(n_rows, n_cols)

                window = window_array(a, window_size, overlap)
                area = search_array(b, window_size, overlap, area=self.search_area, offsets=offset)

                match = block_match(window, area, mode)
                du, dv = correlation_to_displacement(match, n_rows, n_cols, mode)

                u, v = u + du*self.multipass_scale, v + dv*self.multipass_scale
        return x, y, u, -v


class OpticalFlow:
    def __init__(self, folder: str=None, device: torch.device="cpu",
                 multipass: int=1, multipass_scale: float=2.,
                 alpha: float=1000., num_iter: int=100, eps: float=1e-5,
                 dt: float=1/240) -> None:
        self.dataset = SIVDataset(folder=folder)
        self.device = device
        self.multipass, self.multipass_scale = multipass, multipass_scale
        self.alpha, self.num_iter, self.eps = alpha, num_iter, eps
        self.dt = dt

    def run(self):
        for idx, data in enumerate(self.dataset):
            if idx == 0:
                img_a, img_b = data
                rows, cols = img_a.shape[-2:]

                u, v = torch.zeros_like(img_a), torch.zeros_like(img_b)
                x, y = torch.meshgrid(torch.arange(0, cols, 1),torch.arange(0, rows, 1))

                for k in range(self.multipass):
                    scale = self.multipass_scale ** (k - self.multipass + 1)
                    new_size = (round(img_a.shape[1] * scale), round(img_a.shape[0] * scale))

                    # https://discuss.pytorch.org/t/image-warping-for-backward-flow-using-forward-flow-matrix-optical-flow/99298
                    # https://discuss.pytorch.org/t/solved-torch-grid-sample/51662/2
                    xx, yy = torch.meshgrid(torch.linspace(-1, 1, cols), torch.linspace(-1, 1, rows))
                    grid = torch.stack((yy, xx), dim=2).unsqueeze(0)
                    v_grid = grid + torch.stack((-u/(cols/2), -v/(rows/2)), dim=2)
                    src = img_a[None, None, :, :].float()
                    img_a_new = grid_sample(src, v_grid, mode='bilinear').squeeze().to(torch.uint8)

                    resize = Resize(new_size, InterpolationMode.BILINEAR)
                    a, b = resize(img_a_new[None, :, :]).squeeze(), resize(img_b[None, :, :]).squeeze()

                    du, dv = optical_flow(a, b, self.alpha, self.num_iter, self.eps)

                    du = interpolate(du[None, None, :, :], img_a.shape, mode='bicubic').squeeze()
                    dv = interpolate(dv[None, None, :, :], img_a.shape, mode='bicubic').squeeze()

                    u, v = u + du/scale, v + dv/scale
        return x, y, u, -v
