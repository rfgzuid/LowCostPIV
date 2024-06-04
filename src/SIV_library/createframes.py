from SIV_library.processing import *
import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

video_file = "C:/Users/jortd/PycharmProjects/kloten/Test Data/smokechessboard30-05.mp4"

# reference frame specified first, then the range we want to analyse with SIV
frames = [0, *(i for i in range(1000, 1100))]
vid = Video(rf"{video_file}", df='.jpg', indices=frames)
vid.create_frames()
