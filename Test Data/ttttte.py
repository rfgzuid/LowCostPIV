import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# Load the image
image_path = "charuco_temp_folder/chessboard_pic/middle_frame.jpg"
image = Image.open(image_path)
image = np.array(image)

image = cv2.circle(image, (0, 0), radius=50, color=(255, 0, 0), thickness=-1)
image = cv2.circle(image, (image.shape[1]-1, image.shape[0]-1), radius=50, color=(255, 0, 0), thickness=-1)
image = cv2.circle(image, (image.shape[1]-1, 0), radius=50, color=(255, 0, 0), thickness=-1)
image = cv2.circle(image, (0, image.shape[0]-1), radius=50, color=(255, 0, 0), thickness=-1)
if image is None:
    print(f"Failed to load image at {image_path}")
    exit()

# Create a figure and axis
fig, ax = plt.subplots()
plt.figure(figsize=(10, 10))
# Display the input image
ax.imshow(image)

# Define points (you can modify these as needed)
image_points = [
    [1070, 315],
    [1557, 395],
    [1677, 654],
    [997, 532]
]

# Add dots for each coordinate
for point in image_points:
    ax.scatter(point[0], point[1], color='red', s=40)  # s is the size of the dot

# Show the plot
# plt.title("Original Image with Points")
# plt.show()

if len(image_points) != 4:
    print("You need to select exactly 4 points.")
    exit()

# Convert points to numpy float32 format
pts1 = np.float32(image_points)

# Compute the width and height of the quadrilateral
width_top = math.hypot(pts1[0][0] - pts1[1][0], pts1[0][1] - pts1[1][1])
width_bottom = math.hypot(pts1[2][0] - pts1[3][0], pts1[2][1] - pts1[3][1])
height_left = math.hypot(pts1[0][0] - pts1[3][0], pts1[0][1] - pts1[3][1])
height_right = math.hypot(pts1[1][0] - pts1[2][0], pts1[1][1] - pts1[2][1])

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
corners = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
corners = np.float32([
    [276,64],
    [1836,438],
    [1769,1038],
    [100,974]
])
transformed_corners = cv2.perspectiveTransform(corners[None, :, :], matrix)[0]

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
cv2.imwrite("charuco_temp_folder/testing/0.png", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
# Display the transformed image using matplotlib
plt.figure(figsize=(10, 10))
plt.imshow(result)
plt.title("Perspective Transform Applied to Entire Image")
plt.show()
