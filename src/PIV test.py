from torchPIV.PIVbackend import ToTensor, PIVDataset, moving_window_array, correalte_fft
from SIV_library.lib import correlate_conv, correlate_intensity

import torch
import cv2
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

dataset = PIVDataset("../Test Data/plume simulation_PROCESSED", ".jpg",
                     "sequential", transform=ToTensor(dtype=torch.uint8))

img_a, img_b = dataset[1]

window_size, overlap = 128, 64
aa = moving_window_array(img_a, window_size, overlap)
bb = moving_window_array(img_b, window_size, overlap)

idx = 448
corr1 = correalte_fft(aa, bb)
corr2 = correlate_conv(aa, bb, idx)
corr3 = correlate_intensity(aa, bb, idx)

corr_wdw1 = corr1[idx]
min1, max1 = torch.min(corr_wdw1), torch.max(corr_wdw1)
corr_wdw1 = 255 * (corr_wdw1-min1)/(max1-min1)

corr_wdw2 = corr2
min2, max2 = torch.min(corr_wdw2), torch.max(corr_wdw2)
corr_wdw2 = 255 * (corr_wdw2-min2)/(max2-min2)

corr_wdw3 = corr3
min3, max3 = torch.min(corr_wdw3), torch.max(corr_wdw3)
corr_wdw3 = 255 * (corr_wdw3-min3)/(max3-min3)
corr_wdw3 = 255 - corr_wdw3


windows = np.hstack((aa[idx].numpy(), bb[idx].numpy(),
                     corr_wdw1.numpy().astype(np.uint8), corr_wdw2.numpy().astype(np.uint8),
                     corr_wdw3.numpy().astype(np.uint8)))

cv2.imshow("windows", windows)
cv2.waitKey(0)
cv2.destroyAllWindows()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
x, y = np.meshgrid(range(window_size), range(window_size))

ax.plot_surface(x, y, corr_wdw1, color='b', alpha=0.5)
ax.plot_surface(x, y, corr_wdw2, color='r', alpha=0.5)
ax.plot_surface(x, y, corr_wdw3, color='g', alpha=0.5)
plt.show()
