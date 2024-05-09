from torchPIV.PIVbackend import ToTensor, PIVDataset, moving_window_array, correalte_fft
from SIV_library.lib import correlate_conv, correlate_intensity, moving_reference_array, correlate_intensity_optim

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
bb = moving_reference_array(img_b, window_size, overlap, left=20, right=20, top=20, bottom=20)

idx = 567
window, area = aa[idx], bb[idx]

cv2.imshow('Interrogation window', window.cpu().numpy())
cv2.waitKey(0)
cv2.imshow('Search area', area.cpu().numpy())
cv2.waitKey(0)
cv2.destroyAllWindows()

fast = correlate_intensity_optim(window.unsqueeze(0), area.unsqueeze(0))
corr = correlate_conv(window.unsqueeze(0), area.unsqueeze(0))
intensity = correlate_intensity(window.unsqueeze(0), area.unsqueeze(0))

corr_min, corr_max = torch.min(corr), torch.max(corr)
corr_img = (corr-corr_min)/(corr_max-corr_min)

intensity_min, intensity_max = torch.min(intensity), torch.max(intensity)
intensity_img = 1 - (intensity-intensity_min)/(intensity_max-intensity_min)

fast_min, fast_max = torch.min(fast), torch.max(fast)
fast_img = 1 - (fast-fast_min)/(fast_max-fast_min)

plts = np.hstack((corr_img.cpu().numpy(), intensity_img.cpu().numpy(), fast_img.cpu().numpy()))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
x, y = np.meshgrid(range(area.shape[1] - window.shape[1] + 1), range(area.shape[0] - window.shape[0] + 1))

ax.plot_surface(x, y, corr_img, color='b', alpha=0.5, label='correlation')
ax.plot_surface(x, y, intensity_img, color='r', alpha=0.5, label='SAD')
ax.plot_surface(x, y, fast_img, color='g', alpha=0.5, label='SAD fast')
plt.legend()
plt.show()
