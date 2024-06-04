from .matching import window_array, get_field_shape, block_match, get_x_y, correlation_to_displacement
from .optical_flow import optical_flow

import torch

from torch.utils.data import Dataset
import os
import cv2

from tqdm import tqdm


class SIVDataset(Dataset):
    def __init__(self, folder: str, transforms: list | None = None) -> None:
        # assume the files are sorted and all have the correct file type
        filenames = [os.path.join(folder, name) for name in os.listdir(folder)]
        self.img_pairs = list(zip(filenames[:-1], filenames[1:]))

        self.img_shape = cv2.imread(filenames[0]).shape
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.img_pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        pair = self.img_pairs[index]
        img_a, img_b = cv2.imread(pair[0], cv2.IMREAD_GRAYSCALE), cv2.imread(pair[1], cv2.IMREAD_GRAYSCALE)

        img_a, img_b = torch.tensor(img_a, dtype=torch.uint8), torch.tensor(img_b, dtype=torch.uint8)
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
                 ) -> None:

        self.dataset = SIVDataset(folder=folder)
        self.device = device
        self.window_size, self.overlap, self.search_area = window_size, overlap, search_area

        self.img_shape = self.dataset[0][0].shape

    def run(self, mode: int):
        n_rows, n_cols = get_field_shape(self.img_shape, self.window_size, self.overlap)

        x, y = get_x_y(self.img_shape, self.window_size, self.overlap)
        x, y = x.reshape(n_rows, n_cols), y.reshape(n_rows, n_cols)
        x, y = x.expand(len(self.dataset), -1, -1), y.expand(len(self.dataset), -1, -1)

        u, v = (torch.zeros((len(self.dataset), n_rows, n_cols), device=self.device),
                torch.zeros((len(self.dataset), n_rows, n_cols), device=self.device))

        for idx, data in tqdm(enumerate(self.dataset), total=len(self.dataset), desc="Matching"):
            img_a, img_b = data
            a, b = img_a.to(self.device), img_b.to(self.device)

            window = window_array(a, self.window_size, self.overlap)
            area = window_array(b, self.window_size, self.overlap, area=self.search_area)

            match = block_match(window, area, mode)
            du, dv = correlation_to_displacement(match, n_rows, n_cols, mode)

            u[idx], v[idx] = du, dv
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
        self.dataset = SIVDataset(folder=folder)
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
                x, y = x.expand(len(self.dataset), -1, -1), y.expand(len(self.dataset), -1, -1)

                u, v = (torch.zeros((len(self.dataset), rows, cols), device=self.device),
                        torch.zeros((len(self.dataset), rows, cols), device=self.device))

            du, dv = optical_flow(a, b, self.alpha, self.num_iter, self.eps)
            u[idx], v[idx] = du, dv
        return x, y, u, -v
