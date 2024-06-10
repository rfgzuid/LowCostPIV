from processing import *
import cv2
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import RegularGridInterpolator

from tqdm import tqdm
import os
video_file = "kutding.MOV"
fn = video_file.split(".")[0]

# reference frame specified first, then the range we want to analyse with SIV
frames = [*(i for i in range(317, 338)), *(i for i in range(557, 578)), *(i for i in range(797, 818)), *(i for i in range(1037, 1058)), *(i for i in range(1277, 1298))]

#vid = Video(rf"../Test Data/{video_file}", df='.png', indices=frames)
#vid.create_frames()

processor = Processor(rf"../Test Data/IMG_2371", df='.png')
processor.postprocess()