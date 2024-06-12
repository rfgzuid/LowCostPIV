import torch
from torch.nn.functional import grid_sample, interpolate
from torchvision.transforms import Resize, InterpolationMode

from src.SIV_library.lib import OpticalFlow, SIV


class Warp(torch.nn.Module):
    """Custom module that creates a warped images according to the velocity field acquired in a previous pass
    Image is expected to be of shape (N, C, H, W), just like any torchvision transform"""

    def __init__(self, x, y, u, v):
        super().__init__()
        self.x, self.y, self.u, self.v = x, y, u, v
        self.idx = 0

        self.apply_to = ['a']  # apply this transform to ONLY the first image of the pair

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        rows, cols = image.shape[-2:]
        self.interpolate_field(image.shape[-2:])  # interpolate SIV results to image size

        x, y = self.x / ((cols - 1) / 2) - 1, self.y / ((rows - 1) / 2) - 1

        grid = torch.stack((x[self.idx], y[self.idx]), dim=-1).to(x.device)
        v_grid = grid + torch.stack((-self.u[self.idx] / (cols / 2), self.v[self.idx] / (rows / 2)), dim=-1)

        img_new = grid_sample(image.float(), v_grid[None, :, :, :], mode='bicubic').to(torch.uint8)
        self.idx += 1
        return img_new

    def interpolate_field(self, img_shape) -> None:
        if self.u.shape[-2:] == img_shape:
            return

        self.u = interpolate(self.u[None, :, :, :], img_shape, mode='bicubic').squeeze(dim=0)
        self.v = interpolate(self.v[None, :, :, :], img_shape, mode='bicubic').squeeze(dim=0)

        y, x = torch.meshgrid(torch.arange(0, img_shape[0], 1), torch.arange(0, img_shape[1], 1))
        x, y = x.expand(self.x.shape[0], -1, -1), y.expand(self.y.shape[0], -1, -1)
        self.x, self.y = x.to(self.x.device), y.to(self.y.device)


def ctf_optical(optical: OpticalFlow, num_passes: int = 3, scale_factor: float = 1/2):
    """
    runs the optical flow algorithm in a coarse-to-fine pyramidal structure, allowing for larger displacements
    https://www.ipol.im/pub/art/2013/20/article.pdf
    """
    img_shape = optical.dataset.img_shape
    scales = [scale_factor ** (num_passes - p - 1) for p in range(num_passes)]
    sizes = [(round(img_shape[0] * scale), round(img_shape[1] * scale)) for scale in scales]

    u = torch.zeros((len(optical.dataset), *sizes[0]), device=optical.device)
    v = torch.zeros((len(optical.dataset), *sizes[0]), device=optical.device)

    for idx, size in enumerate(sizes):
        y, x = torch.meshgrid(torch.arange(0, size[0], 1), torch.arange(0, size[1], 1))
        x = x.expand(len(optical.dataset), -1, -1).to(optical.device)
        y = y.expand(len(optical.dataset), -1, -1).to(optical.device)

        resize = Resize(size, InterpolationMode.BICUBIC)
        resize.apply_to = ['a', 'b']  # apply the resize transform to both images in the pair
        warp = Warp(x, y, u, v)

        optical.dataset.img_shape = size
        optical.dataset.transforms = [resize, warp]

        _, _, du, dv = optical.run()
        u, v = u + du, v + dv

        if idx < num_passes - 1:
            u = interpolate(u[None, :, :, :], sizes[idx + 1], mode='bicubic').squeeze()
            v = interpolate(v[None, :, :, :], sizes[idx + 1], mode='bicubic').squeeze()

            u, v = u / scale_factor, v / scale_factor
    return x, y, u, v


def match_refine(matching: SIV, optical: OpticalFlow):
    """
    runs the matching algorithm and refines the result with optical flow
    https://link-springer-com.tudelft.idm.oclc.org/article/10.1007/s00348-019-2820-4?fromPaywallRec=false
    """
    img_shape = matching.dataset.img_shape
    x, y, u, v = matching.run()

    warp = Warp(x, y, u, v)

    warp.interpolate_field(img_shape)
    x, y, u, v = warp.x, warp.y, warp.u, warp.v

    optical.dataset.transforms = [warp]

    _, _, du, dv = optical.run()
    u, v = u + du, v + dv
    return x, y, u, v
