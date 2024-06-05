from src.SIV_library.processing import Video, Processor, Viewer
from src.SIV_library.lib import SIV

import numpy as np
import torch

import os


if __name__ == "__main__":
    video_file = "Cilinder.MOV"
    fn = video_file.split(".")[0]

    # reference frame specified first, then the range we want to analyse with SIV
    frames = [0, *(i for i in range(225, 325))]

    vid = Video(rf"Test Data/{video_file}", df='.png', indices=frames)
    vid.create_frames()

    processor = Processor(rf"Test Data/{fn}", df='.png')
    processor.postprocess()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    capture_fps = 240.
    scale = 0.02

    siv = SIV(
        folder=rf"Test Data/{fn}_PROCESSED",
        device=device,
        window_size=64,
        overlap=32,
        search_area=(20, 20, 20, 20)
    )

    if f"{fn}_RESULTS.npy" not in os.listdir(f"Test Data"):
        x, y, vx, vy = siv.run(mode=1)
        res = np.array((x.cpu(), y.cpu(), vx.cpu(), vy.cpu()))
        np.save(rf"Test Data/{fn}_RESULTS", res)
    else:
        print("Loading results...")
        res = np.load(rf"Test Data/{fn}_RESULTS.npy")

    viewer = Viewer(rf"Test Data/{fn}_PROCESSED", playback_fps=30., capture_fps=capture_fps)

    # viewer.play_video()
    viewer.vector_field(res, scale)
    # viewer.velocity_field(res, scale, 30, 'cubic')
