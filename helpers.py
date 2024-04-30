import os
import glob
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import matplotlib.pyplot as plt
import numpy as np
import threading
import warnings
import time
import glm

default_fig_size = (9.5, 3.5)
dataset_folder = "datasets"
force_show_plots = False

class force_show_plots():
    def __enter__(self):
        global force_show_plots
        self._old_force_show_plots = force_show_plots
        force_show_plots = True

    def __exit__(self, a, b, c):
        global force_show_plots
        force_show_plots = self._old_force_show_plots

def hide_plots():
    return not force_show_plots and "NBGRADER_EXECUTION" in os.environ and (os.environ["NBGRADER_EXECUTION"] == "autograde" or os.environ["NBGRADER_EXECUTION"] == "validate")


def SSD(lhs, rhs):
    return np.sum((lhs - rhs)**2)


def SSD_per_pixel(lhs, rhs):
    return SSD(lhs, rhs) / np.product(np.array(lhs).shape)

def _image_to_rgb(image):
    if image.dtype == np.float64:
        image = image.astype(np.float32)
        
    if len(image.shape) == 3:
        image_rgb = image
    else:
        if image.dtype == bool:
            image = image * np.ones(image.shape, np.float32)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    return image_rgb


def show_image(image, title="Title"):
    if hide_plots():
        return

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        image_rgb = _image_to_rgb(image)
        image_rgb = np.clip(image_rgb, 0, 1)

        plt.figure()
        if type(image_rgb) == cv2.UMat:
            plt.imshow(image_rgb.get())
        else:
            plt.imshow(image_rgb)
        plt.axis("off")
        plt.title(title)
        plt.tight_layout()
        plt.show()


# https://stackoverflow.com/questions/46615554/how-to-display-multiple-images-in-one-figure-correctly/46616645
def show_images(figures, nrows=1, ncols=1, col_height=3):
    if hide_plots():
        return

    """Plot a dictionary of figures.

    Parameters
    ----------
    figures : <title, figure> dictionary
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """
    assert(len(figures) <= nrows * ncols)
    if (len(figures) == 1):
        for title, image in figures.items():
            show_image(image, title)
        return

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # col_height = 3  # if nrows <= 2 else 2.5
        fig, axeslist = plt.subplots(
            ncols=ncols, nrows=nrows, figsize=(9.5, nrows * col_height))
        #fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows)
        for ind, title in enumerate(figures):
            if title is None or figures[title] is None:
                axeslist.ravel()[ind].set_axis_off()
                continue

            image = figures[title]
            image_rgb = _image_to_rgb(image)
            image_rgb = np.clip(image_rgb, 0, 1)

            if type(image_rgb) == cv2.UMat:
                axeslist.ravel()[ind].imshow(image_rgb.get())
            else:
                axeslist.ravel()[ind].imshow(image_rgb)

            axeslist.ravel()[ind].set_title(title)
            axeslist.ravel()[ind].set_axis_off()
        for ind in range(len(figures), nrows * ncols):
            axeslist.ravel()[ind].set_axis_off()
        plt.tight_layout()  # optional


def resize_image(image, scale, nn_interpolation=False):
    if nn_interpolation:
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation = cv2.INTER_NEAREST)
    else:
        return cv2.resize(image, None, fx=scale, fy=scale)


def bgr2rgb(image):
    if image.dtype == np.float64:
        image = image.astype(np.float32)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rgb2bgr(image):
    if image.dtype == np.float64:
        image = image.astype(np.float32)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

def rgb2gray(image):
    if image.dtype == np.float64:
        image = image.astype(np.float32)
    return cv2.cvtColor(rgb2bgr(image), cv2.COLOR_BGR2GRAY)

def check_file_exists(file):
    if not os.path.exists(file):
        print(f"Cannot find file \"{file}\"!")

def imread_hdr(file, scale=1.0, nn_interpolation=False):
    check_file_exists(file)
    image = bgr2rgb(cv2.imread(file, cv2.IMREAD_UNCHANGED))
    if scale != 1.0:
        image = resize_image(image, scale, nn_interpolation)
    if "normal" in file:
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                N = image[y, x] 
                if glm.dot(N, N) > 0:
                    image[y, x] = glm.normalize(image[y, x])
    return image

def imread_normalized_float(file, scale=1.0, nn_interpolation=False):
    # Check if a *.npy version of this image exists.
    # Only here so we can more easily fix up assignment 3 (2021/2022).
    # Remove this in the future
    file_npy = os.path.splitext(file)[0] + ".npy"
    if os.path.exists(file_npy):
        file = file_npy

    if not os.path.exists(file):
        print(f"ERROR: trying to read non-existant image file \"{file}\"")

    if os.path.splitext(file)[1] == ".npy":
        return np.load(file) * scale
    else:
        image = bgr2rgb(cv2.imread(file))
        if scale != 1.0:
            image = resize_image(image, scale, nn_interpolation)
        return np.array(image / 255.0, dtype=np.float32)

def imread_normalized_float_grayscale(file, scale=1.0, nn_interpolation=False):
    # Check if a *.npy version of this image exists.
    # Only here so we can more easily fix up assignment 3 (2021/2022).
    # Remove this in the future
    file_npy = os.path.splitext(file)[0] + ".npy"
    if os.path.exists(file_npy):
        file = file_npy
        
    if os.path.splitext(file)[1] == ".npy":
        return np.load(file) * scale
    else:
        check_file_exists(file)
        image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        if scale != 1.0:
            image = resize_image(image, scale, nn_interpolation)
        return np.array(image / 255.0, dtype=np.float32)

def imwrite(file, image):
    if os.path.splitext(file)[1] == ".npy":
        np.save(file, image)
    else:
        cv2.imwrite(file, cv2.cvtColor(np.uint8(np.clip(image*255, 0, 255)), cv2.COLOR_RGB2BGR))


def normalize_image(gray_image):
    return (gray_image - np.min(gray_image)) / (np.max(gray_image) - np.min(gray_image))

def list_files_in_folder(folder, filter="*"):
    x = glob.glob(os.path.join(folder, filter), recursive=True)
    rel_paths = glob.glob(os.path.join(folder, filter))
    #abs_paths = [os.path.abspath(f) for f in rel_paths]
    return [f for f in rel_paths if os.path.isfile(f)]

# List of (key, value) tuples to a dict of {key: [values]} (thus a key may appear multiple times in the input list)


def list_to_multi_dict(l):
    # https://stackoverflow.com/questions/261655/converting-a-list-of-tuples-into-a-dict
    d = dict()
    for k, v in l:
        d.setdefault(k, []).append(v)
    return d


# Flatten from list of lists (i.e. [[1, 2], [3, 4, 5]]) to a single list ([1, 2, 3, 4, 5])
def flatten_list(l):
    # https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists
    return [item for sublist in l for item in sublist]


class Timer():
    def __enter__(self):
        self.__start = time.time()

    def __exit__(self, type, value, traceback):
        self.__end = time.time()

    def elapsed(self):
        return self.__end - self.__start


class ThreadWithReturnValue(threading.Thread):
        # https://stackoverflow.com/questions/6893968/how-to-get-the-return-value-from-a-thread-in-python

    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        super().__init__(group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args,
                                        **self._kwargs)

    def join(self, *args, **kwargs):
        threading.Thread.join(self, *args, **kwargs)
        return self._return


def timed_execution(timeout, function, args):
    s = time.time()
    t = ThreadWithReturnValue(target=function, args=args)
    t.start()
    return t.join(timeout=timeout)


# Float to int
def round_to_int(f):
    return int(f + 0.5)