from torch.nn.functional import conv2d, pad, grid_sample
from torchvision.transforms import Resize, InterpolationMode
import torch
import numpy as np

from torch.utils.data import Dataset
import os
import cv2

from tqdm import tqdm
import matplotlib.pyplot as plt


def block_match(windows: torch.Tensor, areas: torch.Tensor, mode: int) -> torch.Tensor:
    windows, areas = windows.float(), areas.float()
    (count, window_rows, window_cols), (area_rows, area_cols) = windows.shape, areas.shape[-2:]

    res = torch.zeros((count, area_rows - window_rows + 1, area_cols - window_cols + 1))

    if mode == 0:  # correlation mode
        for idx, (window, area) in tqdm(enumerate(zip(windows, areas)), total=count, desc='Correlation'):
            corr = conv2d(area.unsqueeze(0).unsqueeze(0), window.unsqueeze(0).unsqueeze(0), stride=1)
            res[idx] = corr

    elif mode == 1:  # SAD mode
        for j in tqdm(range(area_rows - window_rows + 1), desc='SAD'):
            for i in range(area_cols - window_cols + 1):
                ref = areas[:, j:j + window_rows, i:i + window_cols]
                res[:, j, i] = torch.sum(torch.abs(windows - ref), dim=(1, 2))

    else:
        raise ValueError("Only mode 0 (correlation) or 1 (SAD) are supported")

    # normalized output
    return res / (window_rows * window_cols)


def window_array(array: torch.Tensor, window_size, overlap) -> torch.Tensor:
    shape = array.shape
    strides = (shape[-1] * (window_size - overlap), (window_size - overlap), shape[-1], 1)

    shape = (int((shape[-2] - window_size) / (window_size - overlap)) + 1,
             int((shape[-1] - window_size) / (window_size - overlap)) + 1,
             window_size, window_size)

    return (torch.as_strided(array, size=shape, stride=strides)
            .reshape(-1, window_size, window_size))


def search_array(array: torch.Tensor, window_size, overlap,
                 area: tuple[int, int, int, int] | None = None,
                 offsets: torch.Tensor | None = None) -> torch.Tensor:
    iters = get_field_shape(array.shape, window_size, overlap)

    dx, dy = torch.round(offsets[0]).int(), torch.round(offsets[1]).int()
    dx_max, dy_max = torch.max(torch.abs(dx)).item(), torch.max(torch.abs(dy)).item()

    left, right, top, bottom = area
    padding = (left + dx_max, right + dx_max, top + dy_max, bottom + dy_max)
    array = pad(array, padding)

    areas = torch.zeros((iters[0]*iters[1], window_size+top+bottom, window_size+left+right), dtype=torch.uint8)

    for j in range(iters[0]):
        for i in range(iters[1]):
            offset_x, offset_y = int(offsets[0, i, j]), int(offsets[1, i, j])
            area = array[j*overlap+offset_y+dy_max:j*overlap+window_size+top+bottom+offset_y+dy_max,
                         i*overlap+offset_x+dx_max:i*overlap+window_size+left+right+offset_x+dx_max]
            areas[i+j*iters[1], :, :] = area

            if i + j*iters[1] == 3550:
                start = (i * overlap + offset_x + dx_max,
                         j*overlap+offset_y+dy_max)
                end = (i*overlap+window_size+left+right+offset_x+dx_max,
                       j*overlap+window_size+top+bottom+offset_y+dy_max)

                img = cv2.rectangle(array.numpy(), start, end, (255, 0, 0), 1)
                img = cv2.resize(img, (500, 500), cv2.INTER_AREA)
                cv2.imshow('img', img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    return areas


def correlation_to_displacement(corr: torch.Tensor, n_rows, n_cols, mode: int = 0):
    c, rows, cols = corr.shape

    eps = 1e-7
    corr += eps

    if mode == 0:
        m = corr.view(c, -1).argmax(-1, keepdim=True)  # correlation: argmax
    elif mode == 1:
        m = corr.view(c, -1).argmin(-1, keepdim=True)  # SAD: argmin
    else:
        raise ValueError("Mode must be either 0 or 1")

    row, col = torch.floor_divide(m, cols), torch.remainder(m, cols)
    neighbors = torch.zeros(c, 3, 3)

    no_displacements = torch.zeros(c, dtype=torch.bool)
    edge_cases = torch.zeros(c, dtype=torch.bool)

    for idx, field in enumerate(corr):
        row_idx, col_idx = row[idx].item(), col[idx].item()
        peak_val = torch.max(field) if mode == 0 else torch.min(field)

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

        s_x[edge_cases], s_y[edge_cases] = 0., 0.

    # Polynomial interpolation for SAD
    if mode == 1:
        xx = torch.tensor([[-1, 0, 1],
                           [-1, 0, 1],
                           [-1, 0, 1]], dtype=torch.float32)
        yy = torch.tensor([[1, 1, 1],
                           [0, 0, 0],
                           [-1, -1, -1]], dtype=torch.float32)
        x, y = xx.flatten(), yy.flatten()

        # design matrix (https://www.youtube.com/watch?v=9Zve4NFBbSM)
        A = torch.stack((torch.ones_like(x), x, y, x*y, x**2, y**2)).T
        A = A.unsqueeze(0).repeat(c, 1, 1)

        B = neighbors.flatten(start_dim=1)
        res = torch.linalg.lstsq(A, B)
        coefs = res.solution

        a0, a1, a2, a3, a4, a5 = coefs[:, 0], coefs[:, 1], coefs[:, 2], coefs[:, 3], coefs[:, 4], coefs[:, 5]
        a = torch.stack([torch.stack([2*a4, a3], dim=1), torch.stack([a3, 2*a5], dim=1)], dim=2)

        ranks = torch.linalg.matrix_rank(a)
        singular = torch.where(ranks != 2, True, False)

        min_locs = torch.zeros([c, 2])

        b = torch.stack([-a1, -a2], dim=1)
        min_locs[~singular] = torch.linalg.solve(a[~singular], b[~singular])

        s_x, s_y = min_locs[:, 0], min_locs[:, 1]

        # cases where interpolation fails: override with s = 0.
        s_x = torch.where(torch.abs(s_x) >= 1., 0., s_x)
        s_y = torch.where(torch.abs(s_y) >= 1., 0., s_y)
        s_x[edge_cases], s_y[edge_cases] = 0., 0.

        ############################################################################

        # idx = 3550
        #
        # xs, ys = np.linspace(-1, 1, 51), np.linspace(-1, 1, 51)
        # a0, a1, a2, a3, a4, a5 = a0[idx], a1[idx], a2[idx], a3[idx], a4[idx], a5[idx]
        # min_loc = min_locs[idx]
        #
        # def surface(xi, yi):
        #     return a0 + a1 * xi + a2 * yi + a3 * xi * yi + a4 * xi ** 2 + a5 * yi ** 2
        #
        # heightmap = surface(xs[None, :], ys[:, None])
        # xs, ys = np.meshgrid(xs, ys)
        #
        # min_height = surface(*min_loc)
        # print('u', min_loc, min_height)
        #
        # x, y = np.meshgrid(np.arange(-1, 2, 1), np.arange(-1, 2, 1))
        #
        # fig, ax = plt.subplots()
        # ax = fig.add_subplot(projection='3d')
        #
        # ax.plot_surface(x, -y, neighbors[idx], color='r', alpha=0.5, label='SAD')
        # ax.plot_surface(xs, ys, heightmap.reshape(51, 51), color='b', alpha=0.5, label='interp')
        # ax.scatter(*min_loc, min_height, color='g')
        #
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # plt.show()

    m2d = torch.cat((m // rows, m % cols), -1)

    u = m2d[:, 1][:, None] + s_x[:, None]
    v = m2d[:, 0][:, None] + s_y[:, None]

    default_peak_position = corr.shape[-2:]
    v = v - int(default_peak_position[0] / 2)
    u = u - int(default_peak_position[1] / 2)

    u[no_displacements], v[no_displacements] = torch.nan, torch.nan

    torch.nan_to_num_(v)
    torch.nan_to_num_(u)

    u = u.reshape(n_rows, n_cols)
    v = v.reshape(n_rows, n_cols)

    return u, v


def get_field_shape(image_size, search_area_size, overlap):
    field_shape = (np.array(image_size) - search_area_size) // (search_area_size - overlap) + 1
    return field_shape


def get_x_y(image_size, search_area_size, overlap):
    shape = get_field_shape(image_size, search_area_size, overlap)
    x_single = np.arange(shape[1], dtype=int) * (search_area_size - overlap) + search_area_size // 2
    y_single = np.arange(shape[0], dtype=int) * (search_area_size - overlap) + search_area_size // 2
    x = np.tile(x_single, shape[0])
    y = np.tile(y_single.reshape((shape[0], 1)), shape[1]).flatten()
    return x, y


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
    def __init__(self, folder: str, device: torch.device="cpu",
                 window_size: int=128, overlap: int=64, search_area: tuple[int, int, int, int]=(0, 0, 0, 0),
                 multipass: int=1, multipass_scale: float=2.,
                 dt: float=1/240) -> None:
        self.dataset = SIVDataset(folder=folder)
        self.device = device
        self.window_size, self.overlap, self.search_area = window_size, overlap, search_area
        self.multipass, self.multipass_scale = multipass, multipass_scale
        self.dt = dt

    def run(self, mode: int):
        for idx, data in enumerate(self.dataset):
            img_a, img_b = data
            img_a, img_b = img_a.to(self.device), img_b.to(self.device)

            n_rows, n_cols = get_field_shape(img_a.shape, self.window_size, self.overlap)

            u, v = torch.zeros((n_rows, n_cols)), torch.zeros((n_rows, n_cols))

            for k in range(self.multipass):
                scale = self.multipass_scale ** (k - self.multipass + 1)
                window_size, overlap = int(self.window_size * scale), int(self.overlap * scale)

                new_size = (round(img_a.shape[1] * scale), round(img_a.shape[0] * scale))
                resize = Resize(new_size, InterpolationMode.BILINEAR)
                a, b = resize(img_a[None, :, :]).squeeze(), resize(img_b[None, :, :]).squeeze()

                offset = torch.stack((u, v))
                print(offset[:, 3550//n_rows, 3550%n_cols])

                x, y = get_x_y(a.shape, window_size, overlap)
                x, y = x.reshape(n_rows, n_cols), y.reshape(n_rows, n_cols)

                window = window_array(a, window_size, overlap)
                area = search_array(b, window_size, overlap, area=self.search_area, offsets=offset)

                match = block_match(window, area, mode)
                du, dv = correlation_to_displacement(match, n_rows, n_cols, mode)

                print(du[3550//n_rows, 3550%n_cols], dv[3550//n_rows, 3550%n_cols])

                u, v = u + du*self.multipass_scale, v + dv*self.multipass_scale

                fig, ax = plt.subplots(1, 1)

                frame = a.cpu().numpy()
                ax.imshow(frame, cmap='gray')
                ax.quiver(x, y, u, -v, color='red', scale=1, scale_units='xy')

                ax.set_title('Correlation' if mode == 0 else 'SAD')
                ax.set_axis_off()

                fig.tight_layout()
                plt.show()
        return x, y, u, -v


class OpticalFlow:
    def __init__(self, folder: str=None, device: torch.device="cpu",
                 multipass: int=1, multipass_scale: float=2., dt: float=1/240) -> None:
        self.dataset = SIVDataset(folder=folder)
        self.device = device
        self.multipass, self.multipass_scale = multipass, multipass_scale
        self.dt = dt

    def run(self, mode: int):
        for idx, data in enumerate(self.dataset):
            if idx == 0:
                img_a, img_b = data
                n_rows, n_cols = get_field_shape(img_a.shape, self.window_size, self.overlap)
                u, v = torch.zeros((n_rows, n_cols)), torch.zeros((n_rows, n_cols))

                for k in range(self.multipass):
                    scale = self.multipass_scale ** (k - self.multipass + 1)
                    new_size = (round(img_a.shape[1] * scale), round(img_a.shape[0] * scale))

                    a, b = cv2.resize(img_a, new_size, cv2.INTER_AREA), cv2.resize(img_b, new_size, cv2.INTER_AREA)
                    a = torch.tensor(a, dtype=torch.uint8).to(self.device)
                    b = torch.tensor(b, dtype=torch.uint8).to(self.device)

                    x, y = get_x_y(a.shape, window_size, overlap)
                    x, y = x.reshape(n_rows, n_cols), y.reshape(n_rows, n_cols)

                    grid = torch.zeros(1, *a.shape, 2)

                    # https://discuss.pytorch.org/t/cv2-remap-in-pytorch/99354/10
                    du_interp = grid_sample(x, y, du, mode='bilinear')

                    # du_interp = interpolate(du[None, None, :, :], a.shape, mode='bilinear').squeeze()
                    # dv_interp = interpolate(dv[None, None, :, :], a.shape, mode='bilinear').squeeze()

                    u, v = u + du, v + dv
        return x, y, u, v
