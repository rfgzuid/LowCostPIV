from torch.nn.functional import conv2d, pad, interpolate
import torch


def block_match(windows: torch.Tensor, areas: torch.Tensor, mode: int) -> torch.Tensor:
    windows, areas = windows.float(), areas.float()
    (count, window_rows, window_cols), (area_rows, area_cols) = windows.shape, areas.shape[-2:]

    # [6] - all windows with intensity std < 4 will be ignored (error-prone)
    stds = torch.std(windows, dim=(1, 2))
    err = torch.where(stds < 4., True, False).to(windows.device)
    areas[err] = torch.zeros((area_rows, area_cols)).to(windows.device)

    res = torch.zeros((count, area_rows - window_rows + 1, area_cols - window_cols + 1), device=windows.device)

    if mode == 0:  # correlation mode
        for idx, (window, area) in enumerate(zip(windows, areas)):
            corr = conv2d(area.unsqueeze(0).unsqueeze(0), window.unsqueeze(0).unsqueeze(0), stride=1)
            res[idx] = corr

    elif mode == 1:  # SAD mode
        for j in range(area_rows - window_rows + 1):
            for i in range(area_cols - window_cols + 1):
                ref = areas[:, j:j + window_rows, i:i + window_cols]
                res[:, j, i] = torch.sum(torch.abs(windows - ref), dim=(1, 2))

    else:
        raise ValueError("Only mode 0 (correlation) or 1 (SAD) are supported")

    # normalized output
    return res / (window_rows * window_cols)


def window_array(array: torch.Tensor, window_size, overlap,
                 area: tuple[int, int, int, int] | None = None) -> torch.Tensor:
    if area is None:
        area = (0, 0, 0, 0)

    left, right, top, bottom = area
    padded = pad(array, area)
    shape = padded.shape

    strides = (shape[-1] * (window_size - overlap), (window_size - overlap), shape[-1], 1)

    shape = (
        int((shape[-2] - window_size - top - bottom) / (window_size - overlap)) + 1,
        int((shape[-1] - window_size - left - right) / (window_size - overlap)) + 1,
        window_size + top + bottom,
        window_size + left + right,
    )

    return (torch.as_strided(padded, size=shape, stride=strides)
            .reshape(-1, window_size + top + bottom, window_size + left + right))


def correlation_to_displacement(corr: torch.Tensor, search_area, n_rows, n_cols, mode: int = 0):
    c, rows, cols = corr.shape
    device = corr.device

    eps = 1e-7
    corr += eps

    if mode == 0:
        m = corr.view(c, -1).argmax(-1, keepdim=True)  # correlation: argmax
    elif mode == 1:
        m = corr.view(c, -1).argmin(-1, keepdim=True)  # SAD: argmin
    else:
        raise ValueError("Mode must be either 0 or 1")

    row, col = torch.floor_divide(m, cols), torch.remainder(m, cols)
    neighbors = torch.zeros(c, 3, 3).to(device)

    no_displacements = torch.zeros(c, dtype=torch.bool, device=device)
    edge_cases = torch.zeros(c, dtype=torch.bool, device=device)

    for idx, field in enumerate(corr):
        row_idx, col_idx = row[idx].item(), col[idx].item()
        peak_val = field.max() if mode == 0 else field.min()

        # if multiple peak vals exist (e.g. flat field) the displacement is undetermined (set to 0)
        if rows * cols - torch.count_nonzero(field - peak_val) > 1:
            no_displacements[idx] = True
            continue

        # if peak is at the edge, mask as edge case (no interpolation will be considered)
        if row_idx in [0, rows-1] or col_idx in [0, cols-1]:
            edge_cases[idx] = True
            continue

        neighbors[idx] = field.clone()[row_idx-1:row_idx+2, col_idx-1:col_idx+2]

    # Gaussian interpolation for correlation
    if mode == 0:
        ct, cb, cl, cr, cm = (neighbors[:, 0, 1], neighbors[:, 2, 1], neighbors[:, 1, 0],
                              neighbors[:, 1, 2], neighbors[:, 1, 1])

        s_x = (torch.log(cl) - torch.log(cr)) / (2 * (torch.log(cl) + torch.log(cr)) - 4 * torch.log(cm))
        s_y = (torch.log(cb) - torch.log(ct)) / (2 * (torch.log(cb) + torch.log(ct)) - 4 * torch.log(cm))

    # Polynomial interpolation for SAD
    if mode == 1:
        xx = torch.tensor([[-1, 0, 1],
                           [-1, 0, 1],
                           [-1, 0, 1]], dtype=torch.float32, device=device)
        yy = torch.tensor([[1, 1, 1],
                           [0, 0, 0],
                           [-1, -1, -1]], dtype=torch.float32, device=device)
        x, y = xx.flatten(), yy.flatten()

        # Least Squares method (https://www.youtube.com/watch?v=9Zve4NFBbSM)
        A = torch.stack((torch.ones_like(x), x, y, x*y, x**2, y**2)).T
        A = A.unsqueeze(0).repeat(c, 1, 1)

        B = neighbors.flatten(start_dim=1)
        res = torch.linalg.lstsq(A, B)
        coefs = res.solution

        a0, a1, a2, a3, a4, a5 = coefs[:, 0], coefs[:, 1], coefs[:, 2], coefs[:, 3], coefs[:, 4], coefs[:, 5]
        a = torch.stack([torch.stack([2*a4, a3], dim=1),
                         torch.stack([a3, 2*a5], dim=1)],
                        dim=2)

        ranks = torch.linalg.matrix_rank(a)
        singular = torch.where(ranks != 2, True, False)

        min_locs = torch.zeros([c, 2], device=device)

        b = torch.stack([-a1, -a2], dim=1)
        min_locs[~singular] = torch.linalg.solve(a[~singular], b[~singular])

        s_x, s_y = min_locs[:, 0], min_locs[:, 1]

        # cases where interpolation fails: override with s = 0.
        s_x = torch.where(torch.abs(s_x) >= 1., 0., s_x)
        s_y = torch.where(torch.abs(s_y) >= 1., 0., s_y)

    s_x[edge_cases], s_y[edge_cases] = 0., 0.
    m2d = torch.cat((m // cols, m % cols), -1)

    u = m2d[:, 1][:, None] + s_x[:, None]
    v = m2d[:, 0][:, None] + s_y[:, None]

    left, right, top, bottom = search_area
    default_peak_position = corr.shape[-2] + (top - bottom), corr.shape[-1] + (left - right)

    v = v - int(default_peak_position[0] / 2)
    u = u - int(default_peak_position[1] / 2)

    u[no_displacements], v[no_displacements] = 0., 0.

    u = u.reshape(n_rows, n_cols)
    v = v.reshape(n_rows, n_cols)

    return u, v


def get_field_shape(image_size, window_size, overlap):
    field_shape = (torch.tensor(image_size) - window_size) // (window_size - overlap) + 1
    return tuple(field_shape.tolist())


def get_x_y(image_size, search_area_size, overlap):
    shape = get_field_shape(image_size, search_area_size, overlap)

    x_single = torch.arange(0, shape[1]) * (search_area_size - overlap) + search_area_size // 2
    y_single = torch.arange(0, shape[0]) * (search_area_size - overlap) + search_area_size // 2

    x = torch.tile(x_single, (shape[0],))
    y = torch.tile(y_single.reshape((shape[0], 1)), (shape[1],)).flatten()

    return x, y


class WindowShift:
    # copies piv_iteration_DWS
    def __init__(self, img_shape, window_size, overlap, search_area, device):
        self.img_shape = img_shape
        self.search_area = search_area

        pixel_idx = torch.arange(0, img_shape[-2] * img_shape[-1], dtype=torch.int64, device=device).reshape(img_shape)
        self.idx_a = window_array(pixel_idx, window_size, overlap)
        self.idx_b = window_array(pixel_idx, window_size, overlap, search_area)

    def run(self, img_a, img_b, x, y, u, v):
        u_interp = interpolate(u[None, None, :, :], self.img_shape, mode='bicubic').squeeze()
        v_interp = interpolate(v[None, None, :, :], self.img_shape, mode='bicubic').squeeze()

        xx, yy = x[...].to(torch.int64), y[...].to(torch.int64)
        u, v = u_interp[yy, xx].round(), v_interp[yy, xx].round()
        u2, v2 = (u/2).view(-1)[..., None, None].to(torch.int64), (v/2).view(-1)[..., None, None].to(torch.int64)

        a_grid = self.idx_a + v2 * self.img_shape[-1] - u2
        a_grid.clamp_(0, img_a.numel() - 1)
        windows = torch.gather(img_a.view(-1), -1, a_grid.view(-1)).reshape(self.idx_a.shape)

        b_grid = self.idx_b - v2 * self.img_shape[-1] + u2
        b_grid.clamp_(0, img_b.numel() - 1)
        areas = torch.gather(img_b.view(-1), -1, b_grid.view(-1)).reshape(self.idx_b.shape)

        return windows, areas, u, v
