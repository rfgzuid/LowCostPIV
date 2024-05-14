from torchPIV.PIVbackend import ToTensor, PIVDataset, moving_window_array
from SIV_library.lib import block_match, moving_reference_array, match_to_displacement, correlation_to_displacement, get_field_shape, get_x_y

import cv2
import torch
import numpy as np

import matplotlib.pyplot as plt


device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

dataset = PIVDataset("../Test Data/plume simulation_PROCESSED", ".jpg",
                     "sequential", transform=ToTensor(dtype=torch.uint8))

img_a, img_b = dataset[1]
img_a, img_b = img_a.to(device), img_b.to(device)

window_size, overlap = 128, 64
aa = moving_window_array(img_a, window_size, overlap)

window_amount = aa.shape[0]
print(window_amount, 'windows')
bb = moving_reference_array(img_b, window_size, overlap, left=5, right=5, top=5, bottom=5)

idx = 566
window, area = aa, bb
# idx = 5

#4.09458182656812
#-3.4124938088181693

cv2.imshow('Interrogation window', window[idx].cpu().numpy())
cv2.waitKey(0)
cv2.imshow('Search area', area[idx].cpu().numpy())
cv2.waitKey(0)
cv2.destroyAllWindows()
corr = block_match(window, area, mode=0)[idx]
intensity = block_match(window, area, mode=1)[idx]

corr_min, corr_max = torch.min(corr), torch.max(corr)
corr_img = (corr-corr_min)/(corr_max-corr_min)

intensity_min, intensity_max = torch.min(intensity), torch.max(intensity)
intensity_img = 1 - (intensity-intensity_min)/(intensity_max-intensity_min)

match_to_displacement(block_match(window, area, 1))

corr = block_match(window, area, mode=0)
nrows, ncols = get_field_shape(img_a.shape, window_size, overlap)
u, v = correlation_to_displacement(corr, nrows, ncols)
row, col = idx // 31, idx % 31
print(f"u shape = {u.shape}")
print(u[row, col])
print(f"v shape = {v.shape}")
print(v[row, col])
get_x_y(img_a.shape, window_size, overlap)

plts = np.hstack((corr_img.cpu().numpy(), intensity_img.cpu().numpy()))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
x, y = np.meshgrid(range(area.shape[2] - window.shape[2] + 1), range(area.shape[1] - window.shape[1] + 1))


ax.plot_surface(x, y, corr_img, color='b', alpha=0.5, label='correlation')
ax.plot_surface(x, y, intensity_img, color='r', alpha=0.5, label='SAD')

plt.legend()
plt.show()
