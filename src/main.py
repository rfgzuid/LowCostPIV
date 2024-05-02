from SIV_library.processing import Video, Processor, SIVDataset

data_settings = {
    'rescale': 1.,
    'capture fps': 240,
    'playback fps': 30
}

if __name__ == "__main__":
    video_file = "SmokeVideo.mp4"
    fn = video_file.split(".")[0]
    frames = [0, *(i for i in range(300, 350))]

    vid = Video(rf"Test Data/{video_file}", df='.jpg', indices=frames)
    vid.create_frames()

    processor = Processor(rf"Test Data/{fn}", df='.jpg', settings=data_settings)
    processor.postprocess()

    dataset = SIVDataset(rf"Test Data/{fn}_PROCESSED")
    dataset.play_video(playback_fps=30.)

    # capture_rate = 240  # Hz
    # playback_rate = 30  # Hz - Iphone standard
    # fps_settings = (capture_rate, playback_rate)
    #
    # for f in os.listdir('test video'):
    #     os.remove(f'test video/{f}')
    # process_video('IMG_4255.MOV', fps_settings)  # , folder='test video')

    # idx_pair = (150, 151)
    #
    # file1, file2 = f'test video/{idx_pair[0]}.bmp', f'test video/{idx_pair[1]}.bmp'
    # img1, img2 = cv2.imread(file1, cv2.IMREAD_GRAYSCALE), cv2.imread(file2, cv2.IMREAD_GRAYSCALE)
    # img1, img2 = rescaled_img(img1, 0.5), rescaled_img(img2, 0.5)
    #
    # cv2.imshow('window', np.concatenate((img1, img2), axis=1))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #
    # img1 = torch.tensor(img1)
    # res = lib.moving_window_array(img1, 64, 32)
    #
    # wndw = res[150]
