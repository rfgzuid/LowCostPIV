from SIV_library.processing import Video, Processor, Viewer
from torchPIV import OfflinePIV

import numpy as np
import torch

import cv2
import os


if __name__ == "__main__":
    video_file = "plume simulation.mp4"
    fn = video_file.split(".")[0]

    # reference frame specified first, then the range we want to analyse with SIV
    frames = [0, *(i for i in range(300, 400))]

    vid = Video(rf"Test Data/{video_file}", df='.jpg', indices=frames)
    # vid.show_frame(500)
    vid.create_frames()

    processor = Processor(rf"Test Data/{fn}", df='.jpg', denoise=False, rescale=None, crop=False)
    processor.postprocess()

    device = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
    print(device)

    capture_fps = 240.
    scale = 0.02
    t = OfflinePIV(
        folder=rf"Test Data/{fn}_PROCESSED",  # Path to experiment
        device=device,  # Device name
        file_fmt=".jpg",
        wind_size=64,
        overlap=32,
        dt=int(1_000_000/capture_fps),  # Time between frames, mcs
        scale=scale,  # mm/pix
        multipass=1,
        multipass_mode="DWS",  # CWS or DWS
        multipass_scale=2.0,  # Window downscale on each pass
        folder_mode="sequential"  # Pairs or sequential frames
    )

    if f"{fn}_RESULTS.npy" not in os.listdir(f"Test Data"):
        results = []
        for out in t():
            x, y, vx, vy = out

            # intercept nan values to prevent plotting bugs (false vectors)
            vx, vy = np.nan_to_num(vx), np.nan_to_num(vy)
            results.append((x, y, vx, vy))

        res = np.array(results)
        np.save(rf"Test Data/{fn}_RESULTS", res)
    else:
        print("Loading results...")
        res = np.load(rf"Test Data/{fn}_RESULTS.npy")

    viewer = Viewer(rf"Test Data/{fn}_PROCESSED", playback_fps=30., capture_fps=capture_fps)

    # viewer.play_video()
    viewer.vector_field(res, scale)
    # viewer.velocity_field(res, scale, 30, 'cubic')
