from torchPIV.PIVbackend import ToTensor, PIVDataset, moving_window_array, correalte_fft
from SIV_library.lib import correlate_conv, correlate_intensity, moving_reference_array

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

idx = 566
window, area = aa, bb

# cv2.imshow('Interrogation window', window[idx].cpu().numpy())
# cv2.waitKey(0)
# cv2.imshow('Search area', area[idx].cpu().numpy())
# cv2.waitKey(0)
# cv2.destroyAllWindows()

corr = correlate_conv(window, area)[idx]
intensity = correlate_intensity(window, area)[idx]

corr_min, corr_max = torch.min(corr), torch.max(corr)
corr_img = (corr-corr_min)/(corr_max-corr_min)

intensity_min, intensity_max = torch.min(intensity), torch.max(intensity)
intensity_img = 1 - (intensity-intensity_min)/(intensity_max-intensity_min)

plts = np.hstack((corr_img.cpu().numpy(), intensity_img.cpu().numpy()))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
x, y = np.meshgrid(range(area.shape[1] - window.shape[1] + 1), range(area.shape[2] - window.shape[2] + 1))

ax.plot_surface(x, y, corr_img, color='b', alpha=0.5, label='correlation')
ax.plot_surface(x, y, intensity_img, color='r', alpha=0.5, label='SAD')
plt.legend()
plt.show()
