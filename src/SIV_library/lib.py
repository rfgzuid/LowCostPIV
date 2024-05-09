from torch.nn.functional import conv2d, pad, unfold, fold
import torch

from tqdm import tqdm


def correlate_conv(images_a: torch.Tensor, images_b: torch.Tensor) -> torch.Tensor:
    """
    Compute cross correlation based on fft method
    Between two torch.Tensors of shape [c, width, height]
    fft performed over last two dimensions of tensors
    """
    inputs = images_a.float()
    reference = images_b.float()

    num_row, num_column = reference.shape[-2] - inputs.shape[-2], reference.shape[-1] - inputs.shape[-1]
    res = torch.zeros((inputs.shape[0], num_row+1, num_column+1))

    for idx, (inp, ref) in tqdm(enumerate(zip(inputs, reference)), total=inputs.shape[0]):
        res[idx] = conv2d(ref.unsqueeze(0).unsqueeze(0), inp.unsqueeze(0).unsqueeze(0), stride=1)
    return res.squeeze()


def correlate_intensity(images_a: torch.Tensor, images_b: torch.Tensor) -> torch.Tensor:
    height, width = images_a.shape[-2:]
    num_row, num_column = images_b.shape[-2] - images_a.shape[-2], images_b.shape[-1] - images_a.shape[-1]

    res = torch.zeros((images_a.shape[0], num_row + 1, num_column + 1))
    windows = images_a.float()

    for j in tqdm(range(num_row + 1)):
        for i in range(num_column + 1):
            ref = images_b[:, j:j+height, i:i+width].float()
            res[:, j, i] = torch.sum(torch.abs(windows-ref))
    return res.squeeze()


def correlate_intensity_optim(images_a: torch.Tensor, images_b: torch.Tensor) -> torch.Tensor:
    height, width = images_a.shape[-2:]
    num_row, num_column = images_b.shape[-2] - images_a.shape[-2], images_b.shape[-1] - images_a.shape[-1]

    inp, ref = images_a.unsqueeze(0).float(), images_b.unsqueeze(0).float()

    unfolded = torch.nn.functional.unfold(ref, (height, width))
    conv_out = unfolded.transpose(1, 2) - inp.view(inp.size(0), -1)

    sad = torch.sum(torch.abs(conv_out.transpose(1, 2)), dim=1)
    out = torch.nn.functional.fold(sad, (num_row+1, num_column+1), (1,1))

    return out.squeeze()


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
