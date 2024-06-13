import torch
from torch.nn.functional import grid_sample, interpolate
from torchvision.transforms import Resize, InterpolationMode
from torch.utils.data import DataLoader

from src.SIV_library.lib import OpticalFlow, SIV
from src.SIV_library.optical_flow import optical_flow

from collections.abc import Generator
from tqdm import tqdm


class Warp(torch.nn.Module):
    """Custom module that creates a warped images according to the velocity field acquired in a previous pass
    Image is expected to be of shape (N, C, H, W), just like any torchvision transform"""

    def __init__(self, x, y, u, v):
        super().__init__()
        self.x, self.y, self.u, self.v = x, y, u, v
        self.apply_to = ['a']  # apply this transform to ONLY the first image of the pair

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        rows, cols = image.shape[-2:]
        self.interpolate_field(image.shape[-2:])  # interpolate SIV results to image size

        x, y = self.x / ((cols - 1) / 2) - 1, self.y / ((rows - 1) / 2) - 1

        grid = torch.stack((x, y), dim=-1).to(x.device)
        v_grid = grid + torch.stack((-self.u / (cols / 2), self.v / (rows / 2)), dim=-1)

        img_new = grid_sample(image.float(), v_grid, mode='bicubic').to(torch.uint8)
        return img_new

    def interpolate_field(self, img_shape) -> None:
        if self.u.shape[-2:] == img_shape:
            return

        self.u = interpolate(self.u[None, None, :, :], img_shape, mode='bicubic').squeeze(dim=0)
        self.v = interpolate(self.v[None, None, :, :], img_shape, mode='bicubic').squeeze(dim=0)

        y, x = torch.meshgrid(torch.arange(0, img_shape[0], 1), torch.arange(0, img_shape[1], 1))
        self.x, self.y = x.to(self.x.device), y.to(self.y.device)


class CTF:
    """
    runs the optical flow algorithm in a coarse-to-fine pyramidal structure, allowing for larger displacements
    https://www.ipol.im/pub/art/2013/20/article.pdf
    """
    def __init__(self, optical: OpticalFlow, num_passes: int = 3, scale_factor: float = 1/2):
        self.dataset = optical.dataset
        self.num_passes, self.scale_factor = num_passes, scale_factor
        self.alpha, self.num_iter, self.eps = optical.alpha, optical.num_iter, optical.eps

    def __len__(self) -> int:
        return len(self.dataset)

    def __call__(self) -> Generator:
        img_shape = self.dataset.img_shape
        scales = [self.scale_factor ** (self.num_passes - p - 1) for p in range(self.num_passes)]
        sizes = [(round(img_shape[0] * scale), round(img_shape[1] * scale)) for scale in scales]

        loader = DataLoader(self.dataset)
        for a, b in tqdm(loader):
            for idx, size in enumerate(sizes):
                y, x = torch.meshgrid(torch.arange(0, size[0], 1), torch.arange(0, size[1], 1))
                x, y = x.to(self.dataset.device), y.to(self.dataset.device)

                resize = Resize(size, InterpolationMode.BICUBIC)
                resize.apply_to = ['a', 'b']  # apply the resize transform to both images in the pair

                transforms = [resize, Warp(x, y, u, v)] if idx != 0 else [resize]
                for t in transforms:
                    a = t(a[None, :, :, :]).squeeze(0) if 'a' in t.apply_to else a
                    b = t(b[None, :, :, :]).squeeze(0) if 'b' in t.apply_to else b

                du, dv = optical_flow(a, b, self.alpha, self.num_iter, self.eps)
                u, v = (u + du, v + dv) if idx != 0 else (du, dv)

                if idx < self.num_passes - 1:
                    u = interpolate(u[None, None, :, :], sizes[idx + 1], mode='bicubic').squeeze()
                    v = interpolate(v[None, None, :, :], sizes[idx + 1], mode='bicubic').squeeze()

                    u, v = u / self.scale_factor, v / self.scale_factor
            yield x, y, u, v


class Refine:
    """
    runs the matching algorithm and refines the result with optical flow
    https://link-springer-com.tudelft.idm.oclc.org/article/10.1007/s00348-019-2820-4?fromPaywallRec=false
    """
    def __init__(self, match: SIV, optical: OpticalFlow):
        self.match, self.optical = match, optical
        self.dataset = match.dataset
        self.alpha, self.num_iter, self.eps = optical.alpha, optical.num_iter, optical.eps

    def __len__(self) -> int:
        return len(self.dataset)

    def __call__(self) -> Generator:
        loader = DataLoader(self.dataset)
        for a, b in tqdm(loader, total=len(loader)):
            x, y, u, v = self.match.run(a, b)

            warp = Warp(x, y, u, v)
            a, b = warp(a[None, :, :, :]).squeeze(0), warp(b[None, :, :, :]).squeeze(0)

            x, y, u, v = warp.x, warp.y, warp.u, warp.v

            du, dv = self.optical.run(a, b)
            u, v = u + du, v + dv

            yield x, y, u, v
