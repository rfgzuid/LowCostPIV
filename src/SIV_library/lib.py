from .matching import (window_array, get_field_shape, block_match, get_x_y,
                       correlation_to_displacement, shifted_window_array)
from .optical_flow import optical_flow

import torch
from torch.utils.data import Dataset

import os
import cv2
from tqdm import tqdm


class SIVDataset(Dataset):
    def __init__(self, folder: str, transforms: list | None = None, device: str = "cpu") -> None:
        # assume the files are sorted and all have the correct file type
        filenames = [os.path.join(folder, name) for name in os.listdir(folder)]
        self.img_pairs = list(zip(filenames[:-1], filenames[1:]))

        self.img_shape = cv2.imread(filenames[0], cv2.IMREAD_GRAYSCALE).shape
        self.transforms = transforms
        self.device = device

    def __len__(self) -> int:
        return len(self.img_pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        pair = self.img_pairs[index]
        img_a, img_b = cv2.imread(pair[0], cv2.IMREAD_GRAYSCALE), cv2.imread(pair[1], cv2.IMREAD_GRAYSCALE)

        img_a = torch.tensor(img_a, dtype=torch.uint8, device=self.device)
        img_b = torch.tensor(img_b, dtype=torch.uint8, device=self.device)
        img_a, img_b = img_a[None, None, :, :], img_b[None, None, :, :]  # batch and channel dimension for transforms

        if self.transforms is not None:
            for transform in self.transforms:
                img_a = transform(img_a) if 'a' in transform.apply_to else img_a
                img_b = transform(img_b) if 'b' in transform.apply_to else img_b
        return img_a.squeeze(), img_b.squeeze()


class SIV:
    def __init__(self,
                 folder: str,
                 device: torch.device = "cpu",
                 window_size: int = 128,
                 overlap: int = 64,
                 search_area: tuple[int, int, int, int] = (0, 0, 0, 0),
                 mode: int = 1,
                 num_passes: int = 1,
                 scale_factor: float = 1/2
                 ) -> None:

        self.dataset = SIVDataset(folder=folder, device=device)
        self.device = device
        self.window_size, self.overlap, self.search_area = window_size, overlap, search_area
        self.mode, self.num_passes, self.scale_factor = mode, num_passes, scale_factor

    def run(self):
        for idx, data in tqdm(enumerate(self.dataset), total=len(self.dataset),
                              desc="SAD" if self.mode == 1 else "Correlation"):
            img_a, img_b = data
            a, b = img_a.to(self.device), img_b.to(self.device)

            img_shape = a.shape[-2:]

            # multipass loop
            for p in range(self.num_passes):
                scale = self.scale_factor ** p
                window_size, overlap = int(self.window_size * scale), int(self.overlap * scale)

                n_rows, n_cols = get_field_shape(img_shape, window_size, overlap)
                xp, yp = get_x_y(img_shape, window_size, overlap)
                xp, yp = xp.reshape(n_rows, n_cols).to(self.device), yp.reshape(n_rows, n_cols).to(self.device)

                if p == 0:
                    window = window_array(a, window_size, overlap)
                else:
                    window, up, vp = shifted_window_array(a, window_size, overlap, xp, yp, up, vp)
                area = window_array(b, window_size, overlap, area=self.search_area)

                match = block_match(window, area, self.mode)
                du, dv = correlation_to_displacement(match, self.search_area, n_rows, n_cols, self.mode)

                up, vp = (du, dv) if p == 0 else (up + du, vp + dv)

        # create result tensors
        if idx == 0:
            u = torch.zeros((len(self.dataset), n_rows, n_cols)).to(self.device)
            v = torch.zeros((len(self.dataset), n_rows, n_cols)).to(self.device)
            x, y = xp.expand(len(self.dataset), -1, -1), yp.expand(len(self.dataset), -1, -1)

        u[idx], v[idx] = up, vp
        return x, y, u, -v


class OpticalFlow:
    def __init__(self,
                 folder: str = None,
                 device: torch.device = "cpu",
                 alpha: float = 1000.,
                 num_iter: int = 100,
                 eps: float = 1e-5,
                 ) -> None:

        self.folder = folder
        self.dataset = SIVDataset(folder=folder, device=device)
        self.device = device
        self.alpha, self.num_iter, self.eps = alpha, num_iter, eps

    def run(self):
        for idx, data in tqdm(enumerate(self.dataset), total=len(self.dataset), desc='Optical flow'):
            img_a, img_b = data
            a, b = img_a.to(self.device), img_b.to(self.device)

            # initialize result tensors in the first loop
            if idx == 0:
                rows, cols = a.shape[-2:]

                y, x = torch.meshgrid(torch.arange(0, rows, 1), torch.arange(0, cols, 1))
                x = x.expand(len(self.dataset), -1, -1).to(self.device)
                y = y.expand(len(self.dataset), -1, -1).to(self.device)

                u, v = (torch.zeros((len(self.dataset), rows, cols), device=self.device),
                        torch.zeros((len(self.dataset), rows, cols), device=self.device))

            du, dv = optical_flow(a, b, self.alpha, self.num_iter, self.eps)
            u[idx], v[idx] = du, dv
        return x, y, u, -v
