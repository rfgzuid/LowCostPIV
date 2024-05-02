import cv2
import numpy as np

import torch
from torch.utils.data import Dataset

from tqdm import tqdm
import os

from collections.abc import Collection


def process_video(path: str, fps: tuple[float, float], folder=None) -> None:
    cap = cv2.VideoCapture(path)
    frames = []

    while (cap.isOpened()):
        ret, frame = cap.read()

        # if all frames have been loaded, break out of loop
        if not ret:
            break

        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame)

    new_frames = []
    reference = frames[10]

    # Pre-processing: black out background
    for idx, frame in tqdm(enumerate(frames), total=len(frames), desc='Denoising'):
        diff = cv2.absdiff(frame, reference)
        _, mask = cv2.threshold(diff, 8, 255, cv2.THRESH_TOZERO)
        thing = mask.astype(bool)
        new_frame = frame[~mask]
        new_frames.append(new_frame)

    # If a folder is specified, save the processed frames. Else show a video of the processed frames
    for idx, frame in enumerate(new_frames):
        if folder is not None:
            cv2.imwrite(f'{folder}/{idx}.bmp', frame)
        else:
            img = cv2.putText(frame, f't = {(idx / fps[0]):.3f} s', (20, 50),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            video_frame = np.concatenate((frames[idx], img), axis=1)
            height, width = video_frame.shape[:2]

            video_frame = cv2.resize(video_frame, (round(width/3), round(height/3)), interpolation=cv2.INTER_AREA)
            cv2.imshow('PROCESSED VIDEO', video_frame)

            if cv2.waitKey(round(1000 / fps[1])) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


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
        while (cap.isOpened()):
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

    def __init__(self, path: str, df: str, settings: dict) -> None:
        dirs = path.split('/')
        self.root = '/'.join(dirs[:-1])

        self.dir = dirs[-1]
        self.df = df  # data format for frame images

        self.settings = settings
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
            new_frame = self.denoise(new_frame)

            # set first frame as reference
            if self.reference is None:
                self.reference = new_frame

            new_frame = self.mask(new_frame)
            # new_frame = self.rescale(new_frame, self.settings["rescale"])

            cv2.imwrite(f"{self.root}/{self.dir}_PROCESSED/{idx}{self.df}", new_frame)


class SIVDataset(Dataset):
    """
    Create an SIV Dataset
    -   Creates a PyTorch Dataloader from a specified folder
    -   Allows for videos to be played
    """

    def __init__(self, path: str) -> None:
        fns = os.listdir(path)
        files = [f"{path}/{fn}" for fn in fns]

        # sort files by index (to ensure the dataset is sequential)
        self.files = sorted(files, key=lambda x: int(x.split("/")[-1].split(".")[0]))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        img_a = cv2.imread(self.files[idx], cv2.IMREAD_UNCHANGED)
        img_b = cv2.imread(self.files[idx+1], cv2.IMREAD_UNCHANGED)

        return img_a, img_b

    def play_video(self, playback_fps: float):
        for fn in self.files:
            frame = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
            cv2.imshow('PROCESSED VIDEO', frame)

            if cv2.waitKey(round(1000 / playback_fps)) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
