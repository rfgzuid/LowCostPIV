from SIV_library.processing import Video, Processor, Viewer
from SIV_library.lib import OfflinePIV

import numpy as np
import os


if __name__ == "__main__":
    video_file = "IMG_4255.MOV"

    fn = video_file.split(".")[0]
    frames = [0, *(i for i in range(300, 400))]

    vid = Video(rf"Test Data/{video_file}", df='.jpg', indices=frames)
    vid.create_frames()

    processor = Processor(rf"Test Data/{fn}", df='.jpg', denoise=True, rescale=None)
    processor.postprocess()

    t = OfflinePIV(
        folder=rf"Test Data/{fn}_PROCESSED",  # Path to experiment
        device="cpu",  # Device name
        file_fmt=".jpg",
        wind_size=64,
        overlap=32,
        dt=1/30,  # Time between frames, mcs
        scale=0.02,  # mm/pix
        multipass=2,
        multipass_mode="CWS",  # CWS or DWS
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

    viewer = Viewer(rf"Test Data/{fn}_PROCESSED", playback_fps=30., capture_fps=30.)
    # viewer.play_video()
    viewer.vector_field(res, 0.02)
