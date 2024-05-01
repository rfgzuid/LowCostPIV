import cv2
import numpy as np
import lib


def process_video(path: str, fps: tuple[float, float], folder=None) -> None:
    cap = cv2.VideoCapture(path)
    frames = []

    while (cap.isOpened()):
        ret, frame = cap.read()

        # if all frames have been loaded, break out of loop
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)

    new_frames = []
    new_frames.append(np.zeros_like(frames[0]))

    # Pre-processing: black out background
    for idx, frame in enumerate(frames[1:]):
        reference = frames[idx - 1]
        difference = cv2.absdiff(frame, reference)
        _, mask = cv2.threshold(difference, 50, 255, cv2.THRESH_BINARY)
        new_frames.append(mask)

    # If a folder is specified, save the processed frames. Else show a video of the processed frames
    for idx, frame in enumerate(new_frames):
        if folder is not None:
            cv2.imwrite(f'folder/{idx}.jpg', frame)
        else:
            img = cv2.putText(frame, f't = {(idx / fps[0]):.3f} s', (20, 50),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            video_frame = np.concatenate((frames[idx], img), axis=1)
            height, width = video_frame.shape

            video_frame = cv2.resize(video_frame, (round(width/3), round(height/3)), interpolation=cv2.INTER_AREA)
            cv2.imshow('PROCESSED VIDEO', video_frame)

            if cv2.waitKey(round(1000 / fps[1])) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    capture_rate = 240  # Hz
    playback_rate = 30  # Hz - Iphone standard
    fps_settings = (capture_rate, playback_rate)

    process_video('IMG_4255.MOV', fps_settings)
