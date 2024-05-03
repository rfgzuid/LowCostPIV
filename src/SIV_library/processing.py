import cv2
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

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

    def __init__(self, path: str, df: str, denoise: bool = False, rescale: float | None = None) -> None:
        dirs = path.split('/')
        self.root = '/'.join(dirs[:-1])

        self.dir = dirs[-1]
        self.df = df  # data format for frame images

        self.denoise_enabled = denoise
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
            new_frame = self.crop_border(frame)

            if self.denoise_enabled:
                new_frame = self.denoise(new_frame)

            # set first frame as reference
            if self.reference is None:
                self.reference = new_frame

            new_frame = self.mask(new_frame)

            if self.rescale is not None:
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
        info = list(zip(self.files, results))
        fig, ax = plt.subplots()

        frame = self.read_frame(info[0][0])
        image = ax.imshow(frame, cmap='gray')

        # Don't use very first frame for initialization - velocities are zero. This leads to poor vector length scaling
        x0, y0, vx0, vy0 = info[3][1]
        vx0, vy0 = np.ones_like(vx0) * 500, np.ones_like(vy0) * 500  # overwrite scaling if still buggy
        new_x0, new_y0 = x0/scale, np.flip(y0/scale, axis=0)  # y flipped for correct image row coords
        vectors = ax.quiver(new_x0, new_y0, vx0, vy0, np.sqrt(vx0 * vx0 + vy0 * vy0), cmap='jet')

        def update(idx):
            frame = self.read_frame(info[idx][0])
            image.set_data(frame)

            # https://stackoverflow.com/questions/19329039/plotting-animated-quivers-in-python
            vx, vy = info[idx][1][2], info[idx][1][3]
            vectors.set_UVC(vx, vy, np.sqrt(vx * vx + vy * vy))

            return image, vectors

        _ = animation.FuncAnimation(fig=fig, func=update, frames=50, interval=1000/self.playback_fps)
        plt.show()
