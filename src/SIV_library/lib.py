from torch.nn.functional import conv2d, pad
import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm


def block_match(windows: torch.Tensor, areas: torch.Tensor, mode: int) -> torch.Tensor:
    windows, areas = windows.float(), areas.float()
    (count, window_rows, window_cols), (area_rows, area_cols) = windows.shape, areas.shape[-2:]

    res = torch.zeros((count, area_rows - window_rows + 1, area_cols - window_cols + 1))

    if mode == 0:  # correlation mode
        for idx, (window, area) in tqdm(enumerate(zip(windows, areas)), total=count):
            corr = conv2d(area.unsqueeze(0).unsqueeze(0), window.unsqueeze(0).unsqueeze(0), stride=1)
            res[idx] = corr

    elif mode == 1:  # SAD mode
        for j in tqdm(range(area_rows - window_rows + 1)):
            for i in range(area_cols - window_cols + 1):
                ref = areas[:, j:j + window_rows, i:i + window_cols]
                res[:, j, i] = torch.sum(torch.abs(windows - ref), dim=(1, 2))

    else:
        raise ValueError("Only mode 0 (correlation) or 1 (SAD) are supported")

    return res


def match_to_displacement(matches: torch.Tensor):
    count, rows, cols = matches.shape
    res = matches.view(matches.shape[0], -1).argmax(-1, keepdim=True)
    # print(res)


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


def correlation_to_displacement(
        corr: torch.Tensor,
        n_rows, n_cols, correlation_mode: int = 0, interpolation_mode: int = 0):
    """
    Correlation maps are converted to displacement for
    each interrogation window
    Inputs:
        corr : 3D torch.Tensor [channels, :, :]
            contains output of the correlate_fft
        n_rows, n_cols : number of interrogation windows, output of the
            get_field_shape
        validate: bool Flag for validation
        val_ratio: int = 1.2 peak2peak validation coefficient
        validation_window: int = 3 half of peak2peak validation window
    """
    c, rows, cols = corr.shape
    cor = corr.view(c, -1).type(torch.float64)

    if correlation_mode == 0:
        m = corr.view(c, -1).argmax(-1, keepdim=True)  # correlation: argmax
    elif correlation_mode == 1:
        m = corr.view(c, -1).argmin(-1, keepdim=True)  # SAD: argmin
    else:
        raise ValueError("Mode must be either 0 or 1")

    row, col = torch.floor_divide(m, cols), torch.remainder(m, cols)
    neighbors = torch.zeros(c, 3, 3)
    neighbors_valid = np.ones(c)

    for idx, field in enumerate(corr):
        min_idx = torch.argmin(field)
        max_idx = torch.argmax(field)
        if min_idx == max_idx:
            neighbors_valid[idx] = 0.
            continue
        row_idx, col_idx = row[idx].item(), col[idx].item()
        if row_idx in [0, field.shape[0] - 1] or col_idx in [0, field.shape[1] - 1]:
            neighbors_valid[idx] = 0.
            continue
        # print(field[row_idx - 1:row_idx + 2, col_idx - 1:col_idx + 2])
        neighbors[idx] = torch.tensor(field[row_idx - 1:row_idx + 2, col_idx - 1:col_idx + 2])

    print(neighbors)

    nom1, nom2, den1, den2 = 0, 0, 1, 1

    # Gaussian interpolation
    if interpolation_mode == 0:
        ct, cb, cl, cr, cm = (neighbors[:, 0, 1], neighbors[:, 2, 1], neighbors[:, 1, 0],
                              neighbors[:, 1, 2], neighbors[:, 1, 1])

        nom1 = torch.log(cr) - torch.log(cl)
        den1 = 2 * (torch.log(cl) + torch.log(cr)) - 4 * torch.log(cm)
        nom2 = torch.log(cb) - torch.log(ct)
        den2 = 2 * (torch.log(cb) + torch.log(ct)) - 4 * torch.log(cm)

    # SIV interpolation
    elif interpolation_mode == 2:
        # oke stel dat we de 3x3 grid hebben. dan worden X en Y dus de v en u componenten in de grid
        x = np.array((cl, cm, cr))
        y = np.array((ct, cm, cb))
        X, Y = np.meshgrid(x, y, copy=False)
        Z = cm  # ?

        X = X.flatten()
        Y = Y.flatten()
        # A kan hetzelfde blijven omdat in de paper ook dezelfde soort polynomial wordt gebruikt
        A = np.array([X * 0 + 1, X, Y, X ** 2, X ** 2 * Y, X ** 2 * Y ** 2, Y ** 2, X * Y ** 2, X * Y]).T
        B = Z.flatten()

        coeff, r, rank, s = np.linalg.lstsq(A, B)
        den2, den1 = 1, 1
        nom1 = 1  # change to result for x
        nom2 = 1  # change to result for y
    nom1[neighbors_valid], nom2[neighbors_valid] = 0, 0

    m2d = torch.cat((m // rows, m % cols), -1)
    u = m2d[:, 1][:, None] + nom1 / den1
    v = m2d[:, 0][:, None] + nom2 / den2

    default_peak_position = corr.shape[-2:]
    v = v - int(default_peak_position[0] / 2)
    u = u - int(default_peak_position[1] / 2)
    torch.nan_to_num_(v)
    torch.nan_to_num_(u)
    u = u.reshape(n_rows, n_cols).cpu().numpy()
    v = v.reshape(n_rows, n_cols).cpu().numpy()
    return u, v


def get_field_shape(image_size, search_area_size, overlap):
    field_shape = (np.array(image_size) - search_area_size) // (
            search_area_size - overlap
    ) + 1
    return field_shape


def get_x_y(image_size, search_area_size, overlap):
    shape = get_field_shape(image_size, search_area_size, overlap)
    x_single = np.arange(shape[1], dtype=int) * (search_area_size - overlap) + search_area_size // 2
    y_single = np.arange(shape[0], dtype=int) * (search_area_size - overlap) + search_area_size // 2
    x = np.tile(x_single, shape[0])
    y = np.tile(y_single.reshape((shape[0], 1)), shape[1]).flatten()
    return x, y


# TODO: fix flipping
def plot_velocity_single_frame(img, x, y, u, v):
    # vy, vx, y, x = np.reshape(field, (field.shape[0] * field.shape[1], 4)).T

    # new_x = x/piv_gen._scale
    # new_y = y/piv_gen._scale
    # u = np.flip(u, axis=0)
    # v = -v
    plt.imshow(img, cmap='gray')
    plt.quiver(x, y, u, v, color='red')
    plt.show()
