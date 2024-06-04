import os
import cv2
import numpy as np


def crop_and_isolate_green_pixels(input_dir, output_dir, crop_width, crop_height):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = [f for f in os.listdir(input_dir)]

    for filename in files:
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)

        # Calculate the coordinates for the cropping box
        left = int((img.shape[1] - crop_width) / 2)
        top = int((img.shape[0] - crop_height) / 2)
        right = left + crop_width
        bottom = top + crop_height

        # Crop the image
        cropped_img = img[top:bottom, left:right]

        # Convert the cropped image to HSV color space
        hsv_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)

        # Define the range for green color and create a mask
        lower_green = np.array([35, 100, 200])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv_img, lower_green, upper_green)

        # Create an output image that keeps only the green pixels
        green_pixels = cv2.bitwise_and(cropped_img, cropped_img, mask=mask)

        # Convert non-green pixels to black
        non_green_mask = cv2.bitwise_not(mask)
        green_pixels[non_green_mask == 255] = [0, 0, 0]

        # Save the processed image
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, green_pixels)

fn = "4302_warped"
input_directory = rf"Test Data/{fn}"
output_directory = rf"Test Data/{fn}_cropped"
crop_width = 400  # Desired crop width
crop_height = 400  # Desired crop height
crop_and_isolate_green_pixels(input_directory, output_directory, crop_width, crop_height)
