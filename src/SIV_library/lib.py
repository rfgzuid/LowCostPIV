from .matching import window_array, search_array, get_field_shape, block_match, get_x_y, correlation_to_displacement
from .optical_flow import optical_flow
from .plots import plot_optical_flow

from torch.nn.functional import grid_sample, interpolate
from torchvision.transforms import Resize, InterpolationMode
import torch

from torch.utils.data import Dataset
import os
import cv2

from tqdm import tqdm


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
    def __init__(self,
                 folder: str,
                 device: torch.device = "cpu",
                 window_size: int = 128,
                 overlap: int = 64,
                 search_area: tuple[int, int, int, int] = (0, 0, 0, 0),
                 multipass: int = 1,
                 multipass_scale: float = 2.,
                 dt: float = 1/240
                 ) -> None:

        self.dataset = SIVDataset(folder=folder)
        self.device = device
        self.window_size, self.overlap, self.search_area = window_size, overlap, search_area
        self.multipass, self.multipass_scale = multipass, multipass_scale
        self.dt = dt

        self.img_shape = self.dataset[0][0].shape

    def run(self, mode: int):
        n_rows, n_cols = get_field_shape(self.img_shape, self.window_size, self.overlap)

        x, y = get_x_y(self.img_shape, self.window_size, self.overlap)
        x, y = x.reshape(n_rows, n_cols), y.reshape(n_rows, n_cols)
        x, y = x.expand(len(self.dataset), -1, -1), y.expand(len(self.dataset), -1, -1)

        u, v = (torch.zeros((len(self.dataset), n_rows, n_cols), device=self.device),
                torch.zeros((len(self.dataset), n_rows, n_cols), device=self.device))

        for idx, data in tqdm(enumerate(self.dataset), total=len(self.dataset)):
            img_a, img_b = data
            img_a, img_b = img_a.to(self.device), img_b.to(self.device)

            for k in range(self.multipass):
                scale = self.multipass_scale ** (k - self.multipass + 1)
                window_size, overlap = int(self.window_size * scale), int(self.overlap * scale)

                new_size = (round(img_a.shape[0] * scale), round(img_a.shape[1] * scale))
                resize = Resize(new_size, InterpolationMode.BICUBIC)
                a, b = resize(img_a[None, :, :]).squeeze(), resize(img_b[None, :, :]).squeeze()

                offset = torch.stack((u[idx], v[idx]))

                window = window_array(a, window_size, overlap)
                area = search_array(b, window_size, overlap, area=self.search_area, offsets=offset)

                match = block_match(window, area, mode)
                du, dv = correlation_to_displacement(match, n_rows, n_cols, mode)

                u[idx], v[idx] = u[idx] + du, v[idx] + dv
                if k != self.multipass - 1:
                    # upscale for next pass
                    u[idx] *= self.multipass_scale
                    v[idx] *= self.multipass_scale
        return x, y, u.cpu(), -v.cpu()


class OpticalFlow:
    def __init__(self,
                 folder: str = None,
                 device: torch.device = "cpu",
                 multipass: int = 1,
                 multipass_scale: float = 2.,
                 alpha: float = 1000.,
                 num_iter: int = 100,
                 eps: float = 1e-5,
                 dt: float = 1/240
                 ) -> None:

        self.dataset = SIVDataset(folder=folder)
        self.device = device
        self.multipass, self.multipass_scale = multipass, multipass_scale
        self.alpha, self.num_iter, self.eps = alpha, num_iter, eps
        self.dt = dt

        self.img_shape = self.dataset[0][0].shape

    def run(self):
        rows, cols = self.img_shape[-2:]
        x, y = torch.meshgrid(torch.arange(0, cols, 1), torch.arange(0, rows, 1), indexing='ij')
        x, y = x.expand(len(self.dataset), -1, -1), y.expand(len(self.dataset), -1, -1)

        xx, yy = torch.meshgrid(torch.linspace(-1, 1, rows), torch.linspace(-1, 1, cols))
        xx, yy = xx.to(self.device), yy.to(self.device)

        u, v = (torch.zeros((len(self.dataset), rows, cols), device=self.device),
                torch.zeros((len(self.dataset), rows, cols), device=self.device))

        for idx, data in tqdm(enumerate(self.dataset), total=len(self.dataset)):
            img_a, img_b = data
            img_a, img_b = img_a.to(self.device), img_b.to(self.device)

            for k in range(self.multipass):
                scale = self.multipass_scale ** (k - self.multipass + 1)
                new_size = (round(img_a.shape[1] * scale), round(img_a.shape[0] * scale))

                # https://discuss.pytorch.org/t/image-warping-for-backward-flow-using-forward-flow-matrix-optical-flow/99298
                # https://discuss.pytorch.org/t/solved-torch-grid-sample/51662/2
                grid = torch.stack((yy, xx), dim=2).unsqueeze(0)
                v_grid = grid + torch.stack((-u[idx]/(cols/2), -v[idx]/(rows/2)), dim=2)
                src = img_a[None, None, :, :].float()
                img_a_new = grid_sample(src, v_grid, mode='bilinear',
                                        align_corners=False).squeeze().to(torch.uint8)

                resize = Resize(new_size, InterpolationMode.BICUBIC)
                a, b = resize(img_a_new[None, :, :]).squeeze(), resize(img_b[None, :, :]).squeeze()

                du, dv = optical_flow(a, b, self.alpha, self.num_iter, self.eps)

                du = interpolate(du[None, None, :, :], img_a.shape, mode='bicubic').squeeze()
                dv = interpolate(dv[None, None, :, :], img_a.shape, mode='bicubic').squeeze()

                u[idx], v[idx] = u[idx] + du/scale, v[idx] + dv/scale
        return x, y, u.cpu(), -v.cpu()
