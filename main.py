import cv2
import numpy as np

if __name__ == "__main__":
    cap = cv2.VideoCapture('IMG_4255.MOV')
    frames = []

    while (cap.isOpened()):
        ret, frame = cap.read()

        # if all frames have been loaded, break out of loop
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)

    reference = frames[0]
    for idx, frame in enumerate(frames):
        frames[idx] = np.where(np.abs(frame - reference) > 10, 255, 0).astype(np.uint8)

    for frame in frames:
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
