import torch.nn.functional as F
import torch


def correlate_conv(images_a: torch.Tensor, images_b: torch.Tensor, idx: int) -> torch.Tensor:
    """
    Compute cross correlation based on fft method
    Between two torch.Tensors of shape [c, width, height]
    fft performed over last two dimensions of tensors
    """
    res = torch.zeros(128, 128, dtype=torch.float32)

    inp = torch.zeros((256, 256), dtype=torch.float32)
    inp[64:192, 64:192] = images_a[idx]
    outp = images_b[idx].float()

    for i in range(128):
        for j in range(128):
            res[128 - 1 - i, 128 - 1 - j] = torch.sum(inp[i:i + 128, j:j + 128] * outp)
    return res


def correlate_intensity(images_a: torch.Tensor, images_b: torch.Tensor, idx: int) -> torch.Tensor:
    res = torch.zeros(128, 128, dtype=torch.float32)

    inp = torch.zeros(255, 255, dtype=torch.float32)
    inp[64:192, 64:192] = images_a[idx]
    outp = images_b[idx].float()

    for i in range(128):
        for j in range(128):
            res[128-1-i, 128-1-j] = torch.sum(torch.abs(inp[i:i+128, j:j+128] - outp))

    return res
