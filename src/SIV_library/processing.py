import cv2
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import RegularGridInterpolator

from tqdm import tqdm
import os

from collections.abc import Collection


class Video:
    """
    Load video frames and save them in a directory
    """
    def __init__(self, path: str, df: str, indices: Collection[int, ...] | None = None) -> None:
        dirs = path.split('/')
        self.root = '/'.join(dirs[:-1])

        self.fn = dirs[-1]
        self.df = df  # data format for frame images

        self.indices = indices

    def show_frame(self, frame_number: int) -> None:
        cap = cv2.VideoCapture(f"{self.root}/{self.fn}")
        image = None

        idx = 0
        while cap.isOpened():
            ret, frame = cap.read()

            # if all frames have been loaded, break out of loop
            if not ret:
                break

            if idx == frame_number:
                image = frame
                break

            idx += 1
        cap.release()

        cv2.imshow(f"Frame {frame_number}", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def create_frames(self) -> None:
        folder_name = self.fn.split(".")[0]

        try:
            os.mkdir(f"{self.root}/{folder_name}")
        except FileExistsError:
            print(f"Directory '{self.root}/{folder_name}' already exists")
            return

        cap = cv2.VideoCapture(f"{self.root}/{self.fn}")

        idx = 0
        while cap.isOpened():
            ret, frame = cap.read()

            # if all frames have been loaded, break out of loop
            if not ret:
                break

            # if no indices are specified, all frames are processed
            if self.indices is not None:
                if idx in self.indices:
                    cv2.imwrite(f"{self.root}/{folder_name}/{idx}{self.df}", frame)

            else:
                cv2.imwrite(f"{self.root}/{folder_name}/{idx}{self.df}", frame)

            idx += 1
        cap.release()


class Processor:
    """
    Apply post-processing to sequential video frames
        - Specify the folder containing video frames
    """

    def __init__(self, path: str, df: str,
                 denoise: bool = False, rescale: float | None = None, crop: bool = False) -> None:
        dirs = path.split('/')
        self.root = '/'.join(dirs[:-1])

        self.dir = dirs[-1]
        self.df = df  # data format for frame images

        self.denoise_enabled = denoise
        self.crop_enabled = crop
        self.rescale_factor = rescale

        self.reference = None  # for smoke masking

    @staticmethod
    def rescale(image: np.ndarray, factor: float) -> np.ndarray:
        height, width = image.shape[:2]
        scaled_img = cv2.resize(image, (round(width * factor), round(height * factor)), cv2.INTER_AREA)
        return scaled_img

    # remove black border from image
    @staticmethod
    def crop_border(image: np.ndarray) -> np.ndarray:
        y_nonzero, x_nonzero = np.nonzero(image)
        return image[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]

    @staticmethod
    def denoise(image: np.ndarray) -> np.ndarray:
        clean_img = cv2.fastNlMeansDenoising(image, None, h=3, templateWindowSize=7, searchWindowSize=21)
        return clean_img

    # isolate smoke by subtracting reference image (see Jonas paper)
    def mask(self, image: np.ndarray) -> np.ndarray:
        frame = image.copy()  # IDK why but without copying the masking goes wrong
        diff = cv2.absdiff(image, self.reference)
        _, mask = cv2.threshold(diff, 8, 255, cv2.THRESH_BINARY_INV)
        frame[mask.astype(bool)] = 0
        return frame

    def postprocess(self) -> None:
        try:
            os.mkdir(f"{self.root}/{self.dir}_PROCESSED")
        except FileExistsError:
            print(f"Directory '{self.root}/{self.dir}_PROCESSED' already exists")
            return

        for file in tqdm(os.listdir(f"{self.root}/{self.dir}"), desc="Processing"):
            idx = file.split('.')[0]

            frame = cv2.imread(f"{self.root}/{self.dir}/{file}", cv2.IMREAD_GRAYSCALE)

            if self.crop_enabled:
                new_frame = self.crop_border(frame)
            else:
                new_frame = frame

            if self.denoise_enabled:
                new_frame = self.denoise(new_frame)

            # set first frame as reference
            if self.reference is None:
                self.reference = new_frame

            new_frame = self.mask(new_frame)

            if self.rescale_factor is not None:
                new_frame = self.rescale(new_frame, self.rescale_factor)

            cv2.imwrite(f"{self.root}/{self.dir}_PROCESSED/{idx}{self.df}", new_frame)


class Viewer:
    def __init__(self, path: str, capture_fps: float = 240., playback_fps: float = 30.) -> None:
        self.capture_fps, self.playback_fps = capture_fps, playback_fps

        files = [f"{path}/{fn}" for fn in os.listdir(path)]

        # sort files by index (to ensure the dataset is sequential)
        self.files = sorted(files, key=lambda x: int(x.split("/")[-1].split(".")[0]))

    def read_frame(self, fn: str, show_time: bool = True) -> np.ndarray:
        frame = cv2.imread(fn, cv2.IMREAD_UNCHANGED)

        if show_time:
            idx = int(fn.split("/")[-1].split(".")[0])
            frame = cv2.putText(frame, f't = {(idx / self.capture_fps):.3f} s', (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                                2, cv2.LINE_AA)
        return frame

    def play_video(self) -> None:
        for fn in self.files:
            frame = self.read_frame(fn)
            cv2.imshow('PROCESSED VIDEO', frame)

            if cv2.waitKey(round(1000 / self.playback_fps)) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

    def vector_field(self, results: np.ndarray, scale: float) -> None:
        fig, ax = plt.subplots()

        velocities = results[:, 2:][1:]  # exclude reference frame (only nan/zero values)
        abs_velocities = np.sqrt(velocities[:, 0]**2 + velocities[:, 1]**2)
        min_abs, max_abs = np.min(abs_velocities, axis=0), np.max(abs_velocities, axis=0)

        frame = self.read_frame(self.files[0])
        image = ax.imshow(frame, cmap='gray')

        # Note: velocity scaling is done for correct color mapping and quiver length
        x0, y0, vx0, vy0 = results[0]
        new_x0, new_y0 = x0/scale, y0/scale
        vx0, vy0 = np.flip(vx0, axis=0), np.flip(vy0, axis=0)  # flip velocities for correct IMAGE coords
        vectors = ax.quiver(new_x0, new_y0, vx0, vy0, max_abs-min_abs, scale=.2, cmap='jet')

        def update(idx):
            frame = self.read_frame(self.files[idx])
            image.set_data(frame)

            # https://stackoverflow.com/questions/19329039/plotting-animated-quivers-in-python
            vx, vy = results[idx][2], results[idx][3]
            vx, vy = np.flip(vx, axis=0), np.flip(vy, axis=0)
            scaling = np.sqrt(vx ** 2 + vy ** 2) - min_abs
            vectors.set_UVC(vx, vy, scaling)

            return image, vectors

        ani = animation.FuncAnimation(fig=fig, func=update, frames=len(self.files)-1, interval=1000/self.playback_fps)

        # writer = animation.PillowWriter(fps=15,
        #                                 metadata=dict(artist='Me'),
        #                                 bitrate=1800)
        # ani.save('Test Data/plume.gif', writer=writer)

        plt.show()

    def velocity_field(self, results: np.ndarray, scale: float,
                       resolution: int, interpolation_mode: str) -> None:
        fig, ax = plt.subplots()

        velocities = results[:, 2:][1:]  # exclude reference frame (only nan/zero values)
        abs_velocities = np.sqrt(velocities[:, 0] ** 2 + velocities[:, 1] ** 2)
        min_abs, max_abs = np.min(abs_velocities), np.max(abs_velocities)

        frame = self.read_frame(self.files[0])
        height, width = frame.shape[:2]

        # grid points where to interpolate the velocity field to (num = desired resolution)
        xg, yg = np.meshgrid(np.linspace(0, width, resolution),
                             np.linspace(0, height, resolution))

        x0, y0, vx0, vy0 = results[0]
        new_x0, new_y0 = x0 / scale, y0 / scale
        vx0, vy0 = np.flip(vx0, axis=0), np.flip(vy0, axis=0)

        field = RegularGridInterpolator((new_x0[0, :], new_y0[:, 0]),
                                        np.sqrt(vx0 ** 2 + vy0 ** 2), method=interpolation_mode,
                                        bounds_error=False, fill_value=0)
        values = field((xg, yg))
        image = ax.imshow(values.T, vmin=min_abs, vmax=max_abs, cmap='jet')

        def update(idx):
            vx, vy = results[idx][2], results[idx][3]
            vx, vy = np.flip(vx, axis=0), np.flip(vy, axis=0)

            field = RegularGridInterpolator((new_x0[0, :], new_y0[:, 0]),
                                            np.sqrt(vx ** 2 + vy ** 2), method=interpolation_mode,
                                            bounds_error=False, fill_value=0)
            values = field((xg, yg))
            image.set_data(values.T)

            return image,

        ani = animation.FuncAnimation(fig=fig, func=update, frames=len(self.files)-1, interval=1000/self.playback_fps)

        # writer = animation.PillowWriter(fps=15,
        #                                 metadata=dict(artist='Me'),
        #                                 bitrate=1800)
        # ani.save('Test Data/plume.gif', writer=writer)

        plt.show()
