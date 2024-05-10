from torch.nn.functional import conv2d, pad
import torch

from tqdm import tqdm


def block_match(windows: torch.Tensor, areas: torch.Tensor, mode: int) -> torch.Tensor:
    windows, areas = windows.float(), areas.float()
    (count, window_rows, window_cols), (area_rows, area_cols) = windows.shape, areas.shape[-2:]

    res = torch.zeros((count, area_rows - window_rows + 1, area_cols - window_cols + 1))

    if mode == 0:  # correlation mode
        for idx, (window, area) in tqdm(enumerate(zip(windows, areas)), total=count):
            corr = conv2d(area.unsqueeze(0).unsqueeze(0), window.unsqueeze(0).unsqueeze(0), stride=1)
            res[idx] = corr

        for channel in res:
            if torch.count_nonzero(channel) == 0:
                channel[(area_rows - window_rows) // 2, (area_rows - window_rows) // 2] = 1.

    elif mode == 1:  # SAD mode
        for j in tqdm(range(area_rows - window_rows + 1)):
            for i in range(area_cols - window_cols + 1):
                ref = areas[:, j:j + window_rows, i:i + window_cols]
                res[:, j, i] = torch.sum(torch.abs(windows - ref), dim=(1, 2))

        for channel in res:
            if torch.count_nonzero(channel) == 0:
                channel[(area_rows - window_rows) // 2, (area_rows - window_rows) // 2] = -1.

    else:
        raise ValueError("Only mode 0 (correlation) or 1 (SAD) are supported")

    return res


def match_to_displacement(matches: torch.Tensor):
    count, rows, cols = matches.shape
    res = matches.view(matches.shape[0], -1).argmax(-1, keepdim=True)
    print(res)


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
