import torch

from torch.nn.functional import grid_sample, interpolate
from torchvision.transforms import Resize, InterpolationMode
from src.SIV_library.lib import OpticalFlow, SIVDataset


class Warp(torch.nn.Module):
    """Custom module that creates a warped images according to the velocity field acquired in a previous pass
    Image is expected to be of shape (N, C, H, W), just like any torchvision transform"""

    def __init__(self, x, y, u, v):
        super().__init__()
        self.x, self.y, self.u, self.v = x, y, u, v

    def forward(self, image):
        rows, cols = image.shape[-2:]
        x, y = self.x / ((cols - 1) / 2) - 1, self.y / ((rows - 1) / 2) - 1

        # https://discuss.pytorch.org/t/image-warping-for-backward-flow-using-forward-flow-matrix-optical-flow/99298
        # https://discuss.pytorch.org/t/solved-torch-grid-sample/51662/2
        grid = torch.stack((x, y), dim=2).unsqueeze(0)
        v_grid = grid + torch.stack((-self.u / (cols / 2), self.v / (rows / 2)), dim=2)

        img_new = grid_sample(image.float(), v_grid, mode='bicubic').to(torch.uint8)
        return img_new


def coarse_to_fine_optical(inp: OpticalFlow, num_passes: int = 3, scale_factor: float = 1/2):
    """
    runs the optical flow algorithm in a coarse-to-fine pyramidal structure, allowing for larger displacements
    https://www.ipol.im/pub/art/2013/20/article.pdf
    """
    img_shape = inp.img_shape

    for p in range(num_passes):
        scale = scale_factor ** (num_passes - p - 1)
        new_size = (round(img_shape[0] * scale), round(img_shape[1] * scale))

        resize = Resize(new_size, InterpolationMode.BICUBIC)
        warp = Warp(...)
        inp.dataset.transforms = [resize, warp]

        x, y, u, v = inp.run()


def match_refine():
    """
    runs the matching algorithm and refines the result with optical flow
    https://link-springer-com.tudelft.idm.oclc.org/article/10.1007/s00348-019-2820-4?fromPaywallRec=false
    """

    pass
