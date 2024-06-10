from processing import *
import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

video_file = "C:/Users/jortd/PycharmProjects/kloten/Test Data/IMG_2371.MOV"

# reference frame specified first, then the range we want to analyse with SIV
frames = [*(i for i in range(516, 537)), *(i for i in range(756, 777)), *(i for i in range(996, 1017)), *(i for i in range(1236, 1257)), *(i for i in range(1476, 1497))]
#frames = [*(i for i in range(0, 240))]
vid = Video(rf"{video_file}", df='.jpg', indices=frames)
vid.create_frames()
