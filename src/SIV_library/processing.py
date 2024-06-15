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
    def __init__(self, path: str, df: str, indices: Collection[int, ...] | None = None, prepped_frames_path: Collection[np.ndarray] | None = None) -> None:
        dirs = path.split('/')
        self.root = '/'.join(dirs[:-1])

        self.fn = dirs[-1]
        self.df = df  # data format for frame images

        self.indices = indices
        self.prepped_frames_path = prepped_frames_path

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

            if idx in self.indices:
                cv2.imwrite(f"{self.root}/{folder_name}/{idx}{self.df}", frame)

            elif not self.indices or idx in self.indices:
                cv2.imwrite(f"{self.root}/{folder_name}/{idx}{self.df}", frame)
            idx += 1
        cap.release()


class Processor:
    """
    Apply post-processing to sequential video frames
        - Specify the folder containing video frames
    """

    def __init__(self, path: str, df: str) -> None:
        dirs = path.split('/')
        self.root = '/'.join(dirs[:-1])

        self.dir = dirs[-1]
        self.df = df  # data format for frame images
        self.reference = None  # for smoke masking

    @staticmethod
    def denoise(image: np.ndarray) -> np.ndarray:
        # clean_img = cv2.fastNlMeansDenoising(image, None, h=3, templateWindowSize=7, searchWindowSize=21)
        clean_img = cv2.medianBlur(image, 3)
        return clean_img

    # isolate smoke by subtracting reference image (see Jonas paper)
    def mask(self, image: np.ndarray) -> np.ndarray:
        diff = image.astype(np.float32) - self.reference.astype(np.float32)
        diff = np.where(diff < 0., 0., diff)
        diff *= 255./diff.max()
        return diff.astype(np.uint8)

    def postprocess(self) -> None:
        try:
            os.mkdir(f"{self.root}/{self.dir}_PROCESSED")
        except FileExistsError:
            print(f"Directory '{self.root}/{self.dir}_PROCESSED' already exists")
            return

        for file in tqdm(os.listdir(f"{self.root}/{self.dir}"), desc="Processing"):
            idx = file.split('.')[0]

            frame = cv2.imread(f"{self.root}/{self.dir}/{file}", cv2.IMREAD_GRAYSCALE)
            new_frame = self.denoise(frame)

            # set first frame as reference - don't save this one for SIV
            if self.reference is None:
                self.reference = new_frame
            else:
                new_frame = self.mask(new_frame)
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
        vectors = ax.quiver(new_x0, new_y0, vx0, vy0, max_abs-min_abs, scale=.5, cmap='jet')

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

        # writer = animation.PillowWriter(fps=15)
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

        # writer = animation.PillowWriter(fps=15)
        # ani.save('Test Data/plume.gif', writer=writer)

        plt.show()


class WarpVideo:
    def __init__(self, frames_path: str, chessboard_vid_path: str, destination_frames_folder: str | None = None,
                 chessboard_shape: tuple[int, int] = (6, 9)) -> None:
        self.frames_path: str = frames_path
        self.chessboard_vid_path: str = chessboard_vid_path
        self.chessboard_shape: tuple[int, int] = chessboard_shape
        self.destination_frames_folder: str | None = destination_frames_folder
        self.chessboard_images: list[np.ndarray] | None = None
        self.camera_matrix: np.ndarray | None = None
        self.dist_coeff: np.ndarray | None = None
        self.undistorted_frames: np.ndarray | list[np.ndarray] | None = None
        self.adjusted_matrix: np.ndarray | None = None
        self.adjusted_width: int | None = None
        self.adjusted_height: int | None = None
        self.corresp_imgnames: list[str] | None = None
        self.saved_img_names: list[str] | None = None

    @staticmethod
    def display_video(frames):
        for frame in frames:
            cv2.imshow("Frame", frame)
            cv2.waitKey(30)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

    def __load_chessboard_images(self, every_x_frames: int = 20, overwrite: bool = False) -> list[np.ndarray]:
        char_frames = []
        for i, file in enumerate(os.listdir(self.chessboard_vid_path)):
            current_frame = -1
            charuco_vid_file_1 = f"{self.chessboard_vid_path}/{file}"
            video = cv2.VideoCapture(charuco_vid_file_1)
            grabbed, frame = video.read()
            while grabbed:
                current_frame += 1
                if not current_frame % every_x_frames:
                    char_frames.append(frame)
                grabbed, frame = video.read()
        print("number of pics loaded: ", len(char_frames))
        if overwrite:
            self.chessboard_images = char_frames
        return char_frames

    def find_intrinsic(self, overwrite: bool = False) -> tuple[np.ndarray, np.ndarray]:
        self.chessboard_images = self.__load_chessboard_images(
            overwrite=overwrite) if self.chessboard_images is None else self.chessboard_images

        copiedframes = copy.deepcopy(self.chessboard_images)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(4,7,0)
        objp = np.zeros((self.chessboard_shape[0] * self.chessboard_shape[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0: self.chessboard_shape[0], 0: self.chessboard_shape[1]].T.reshape(-1, 2)

        size_of_chessboard_squares_mm = 30
        objp = objp * size_of_chessboard_squares_mm

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane

        # Use the first frame to determine the frame size
        if len(copiedframes) > 0:
            frameHeight, frameWidth = copiedframes[0].shape[:2]
            frameSize = (frameWidth, frameHeight)
        else:
            raise ValueError("No frames in charucoFrames list")

        found = 0
        for idx, frame in enumerate(copiedframes):
            if frame is None:
                continue

            # Downscale the image by a factor of 5
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, self.chessboard_shape, None)

            if ret:
                found += 1
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)

        if len(objpoints) == 0 or len(imgpoints) == 0:
            raise ValueError("No chessboard corners found in any of the frames.")
        print(f"Found chessboard corners in {found} frames.")

        ret, cameraMatrix, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

        if not ret:
            raise ValueError("Camera calibration failed.")
        print(f"Camera calibration succeeded. RME: {ret} pixels")
        self.camera_matrix, self.dist_coeff = cameraMatrix, dist if overwrite or (
        self.camera_matrix, self.dist_coeff) == (None, None) else None
        return cameraMatrix, dist

    def undistort_frames(self, cameraMatrix: np.ndarray | None = None, dist: np.ndarray | None = None,
                         pathToFolder: str | None = None, return_imgs: bool = False,
                         overwrite_existing: bool = True) -> None | list[np.ndarray]:
        if pathToFolder is None:
            pathToFolder = self.frames_path
            if pathToFolder is None:
                raise ValueError("No frames path specified.")
        if cameraMatrix is None:
            cameraMatrix = self.camera_matrix
            if cameraMatrix is None:
                raise ValueError("Camera matrix is empty.")
        if dist is None:
            dist = self.dist_coeff
            if dist is None:
                raise ValueError("Distance coefficient is empty.")
        undistorted_frames = []
        if self.undistorted_frames is None:
            self.undistorted_frames = []
        if self.saved_img_names is None:
            self.saved_img_names = []
        for filename in os.listdir(pathToFolder):
            frame_path = os.path.join(pathToFolder, filename)
            frame = cv2.imread(frame_path)
            h, w = frame.shape[:2]
            newCameraMatrix, _ = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, (w, h), 1, (w, h))

            # Undistort the image
            undistorted = cv2.undistort(frame, cameraMatrix, dist, None, newCameraMatrix)
            if return_imgs:
                undistorted_frames.append(undistorted)
            if overwrite_existing:
                self.undistorted_frames.append(undistorted)
                self.saved_img_names.append(filename)
        if return_imgs:
            return undistorted_frames
        return None

    def findWarpPerspective(self, image_path, image_points, cornerpoints, image: np.ndarray | None = None):

        # Load the image
        if image is None:
            image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image at {image_path}")
            exit()
        # Create a figure and axis
        # Display the input image
        # ax.imshow(image)
        # plt.close()

        # Add dots for each coordinate
        # for point in image_points:
        #     ax.scatter(point[0], point[1], color='red', s=40)  # s is the size of the dot
        if len(image_points) != 4:
            print("You need to select exactly 4 points.")
            exit()
        # Convert points to numpy float32 format
        pts1 = np.float32(image_points)
        # Compute the width and height of the quadrilateral
        width_top = np.linalg.norm(pts1[0] - pts1[1])
        width_bottom = np.linalg.norm(pts1[2] - pts1[3])
        height_left = np.linalg.norm(pts1[0] - pts1[3])
        height_right = np.linalg.norm(pts1[1] - pts1[2])

        # Use the maximum of the widths and heights to define the square size
        max_width = max(int(width_top), int(width_bottom))
        max_height = max(int(height_left), int(height_right))
        square_size = max(max_width, max_height)

        # Define the destination points as a square with the calculated size
        pts2 = np.float32([
            [0, 0],
            [square_size - 1, 0],
            [square_size - 1, square_size - 1],
            [0, square_size - 1]
        ])
        # Get the perspective transform matrix
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        # Warp the entire image using the perspective transform matrix
        # To keep the whole image visible, let's compute the output bounds
        h, w = image.shape[:2]

        # Transform the four corners of the original image
        if cornerpoints is None or len(cornerpoints) == 0:
            corners_points = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        else:
            corners_points = np.float32(cornerpoints)

        transformed_corners = cv2.perspectiveTransform(corners_points[None, :, :], matrix)[0]
        # Find the bounding box of the transformed corners
        x_min, y_min = np.min(transformed_corners, axis=0).astype(int)
        x_max, y_max = np.max(transformed_corners, axis=0).astype(int)

        # Calculate the size of the new image
        new_width = x_max - x_min
        new_height = y_max - y_min

        # Create the translation matrix to shift the image to the positive coordinates
        translation_matrix = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])

        # Adjust the perspective transform matrix with the translation
        adjusted_matrix = translation_matrix @ matrix

        # Perform the warp with the adjusted matrix
        result = cv2.warpPerspective(image, adjusted_matrix, (new_width, new_height), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

        # Display the transformed image using matplotlib
        # plt.figure(figsize=(10, 10))
        # plt.imshow(result)
        # cv2.imwrite("../../Test Data/charuco_temp_folder/measurement_frames_formatted/inverted.jpg", cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        # plt.title("Perspective Transform Applied to Entire Image")
        # plt.show()
        impth = "../../Test Data/charuco_temp_folder/abcdef.jpg"
        abc, defg, _ = result.shape
        # print(result.shape)

        self.adjusted_matrix, self.adjusted_width, self.adjusted_height = adjusted_matrix, new_width, new_height

        # result_resized = cv2.resize(result, (defg // 4, abc// 4), interpolation=cv2.INTER_LINEAR)
        # cv2.imwrite(impth, result)
        # cv2.imshow("res", result_resized)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

    def __warp_image(self, image):
        return cv2.warpPerspective(image, self.adjusted_matrix, (self.adjusted_width, self.adjusted_height),
                                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    def generate_warped_images(self, camera_matrix: np.ndarray | None = None, dist_coeff: np.ndarray | None = None,
                               path_to_write_vids: str | None = None, path_to_distorted_pics: str | None = None,
                               path_to_undistorted_pics: str | None = None,
                               filenames: list[str] | np.ndarray[str] | None = None):
        if camera_matrix is None:
            camera_matrix = self.camera_matrix
        if dist_coeff is None:
            dist_coeff = self.dist_coeff
        if dist_coeff is None or camera_matrix is None:
            self.dist_coeff = None
            self.camera_matrix = None
            self.find_intrinsic(overwrite=True)

        if path_to_write_vids is None:
            raise ValueError("No destination for warped videos found.")
        if path_to_distorted_pics is not None:
            undistorted_imgs = self.undistort_frames(pathToFolder=path_to_distorted_pics, return_imgs=True,
                                                     overwrite_existing=False)
        if self.undistorted_frames is None:
            self.undistort_frames()
        else:
            undistorted_imgs = self.undistorted_frames
        if self.adjusted_matrix is None or self.adjusted_width is None or self.adjusted_height is None:
            raise ValueError("You need to find the warp matrix. Run self.findWarpPerspective first.")

        for i, frame in enumerate(self.undistorted_frames):
            img_name = f"{i}.jpg" if filenames is None else filenames[i]
            warped_image = self.__warp_image(frame)
            cv2.imwrite(f"{path_to_write_vids}/{img_name}", warped_image)
