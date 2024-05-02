from SIV_library.processing import Video, Processor, SIVDataset


if __name__ == "__main__":
    video_file = "SmokeVideo.mp4"
    fn = video_file.split(".")[0]
    frames = [0, *(i for i in range(300, 350))]

    vid = Video(rf"Test Data/{video_file}", df='.jpg', indices=frames)
    vid.create_frames()

    processor = Processor(rf"Test Data/{fn}", df='.jpg', denoise=True, rescale=None)
    processor.postprocess()

    dataset = SIVDataset(rf"Test Data/{fn}_PROCESSED")
    dataset.play_video(playback_fps=3., capture_fps=30.)
