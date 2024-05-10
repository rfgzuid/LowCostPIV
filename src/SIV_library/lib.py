from torch.nn.functional import conv2d, pad, unfold, fold
import torch

from tqdm import tqdm


def correlate_conv(images_a: torch.Tensor, images_b: torch.Tensor) -> torch.Tensor:
    """
    Compute cross correlation based on fft method
    Between two torch.Tensors of shape [c, width, height]
    fft performed over last two dimensions of tensors
    """
    inp, ref = images_a.float(), images_b.float()

    num_row, num_column = ref.shape[-2] - inp.shape[-2], ref.shape[-1] - inp.shape[-1]
    res = torch.zeros((inp.shape[0], num_row+1, num_column+1))

    for idx, (window, area) in tqdm(enumerate(zip(inp, ref)), total=inp.shape[0]):
        res[idx] = conv2d(area.unsqueeze(0).unsqueeze(0), window.unsqueeze(0).unsqueeze(0), stride=1)
    return res


def correlate_intensity(images_a: torch.Tensor, images_b: torch.Tensor) -> torch.Tensor:
    height, width = images_a.shape[-2:]
    num_row, num_column = images_b.shape[-2] - images_a.shape[-2], images_b.shape[-1] - images_a.shape[-1]

    res = torch.zeros((images_a.shape[0], num_row + 1, num_column + 1))
    windows, areas = images_a.float(), images_b.float()

    for j in tqdm(range(num_row + 1)):
        for i in range(num_column + 1):
            ref = areas[:, j:j + height, i:i + width]
            res[:, j, i] = torch.sum(torch.abs(windows - ref), dim=(1, 2))

    return res


def moving_reference_array(array: torch.Tensor, window_size, overlap,
                           left: int, right: int, top: int, bottom: int) -> torch.Tensor:
    padded = pad(array, (left, right, top, bottom))
    shape = padded.shape

    strides = (
        shape[-1] * (window_size - overlap),
        (window_size - overlap),
        shape[-1],
        1
    )
    shape = (
        int((shape[-2] - window_size - top - bottom) / (window_size - overlap)) + 1,
        int((shape[-1] - window_size - left - right) / (window_size - overlap)) + 1,
        window_size + top + bottom,
        window_size + left + right,
    )

    return torch.as_strided(
        padded, size=shape, stride=strides
    ).reshape(-1, window_size + top + bottom, window_size + left + right)
