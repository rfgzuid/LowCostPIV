import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk

class ImageProcessor:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processor")

        self.input_dir_button = ttk.Button(root, text="Select Input Directory", command=self.select_input_directory)
        self.input_dir_button.pack()

        self.img_label = ttk.Label(root)
        self.img_label.pack()

        # Contrast Slider
        self.contrast_label = ttk.Label(root, text="Contrast")
        self.contrast_label.pack()
        self.contrast_slider = ttk.Scale(root, from_=1.0, to=10.0, orient='horizontal', command=self.update_image)
        self.contrast_slider.set(3.0)

        # Lower Green Slider
        self.lower_green_label = ttk.Label(root, text="Lower Green Threshold")
        self.lower_green_label.pack()
        self.lower_green_slider = ttk.Scale(root, from_=0, to=255, orient='horizontal', command=self.update_image)
        self.lower_green_slider.set(35)

        self.save_button = ttk.Button(root, text="Save and Apply to All", command=self.save_and_apply_to_all)

        self.input_dir = ""
        self.output_dir = ""
        self.files = []
        self.current_img = None
        self.crop_width = 800
        self.crop_height = 800

    def select_input_directory(self):
        self.input_dir = filedialog.askdirectory(title="Select Input Directory")
        if self.input_dir:
            self.output_dir = os.path.join(self.input_dir, "output")
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            self.files = [f for f in os.listdir(self.input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
            self.current_img = cv2.imread(os.path.join(self.input_dir, self.files[0]))

            self.crop_image()
            self.show_image(self.current_img)

            self.contrast_slider.pack()
            self.lower_green_slider.pack()
            self.save_button.pack()

    def crop_image(self):
        left = int((self.current_img.shape[1] - self.crop_width) / 2)
        top = int((self.current_img.shape[0] - self.crop_height) / 2)
        right = left + self.crop_width
        bottom = top + self.crop_height
        self.current_img = self.current_img[top:bottom, left:right]

    def show_image(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        # Resize the image for preview
        img_pil.thumbnail((400, 400))  # Resize to fit in a 400x400 box while maintaining aspect ratio

        img_tk = ImageTk.PhotoImage(img_pil)
        self.img_label.img_tk = img_tk
        self.img_label.config(image=img_tk)

    def update_image(self, event=None):
        clahe = cv2.createCLAHE(clipLimit=self.contrast_slider.get(), tileGridSize=(8, 8))
        lab = cv2.cvtColor(self.current_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        hsv_img = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2HSV)
        lower_green = np.array([int(self.lower_green_slider.get()), 50, 50])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv_img, lower_green, upper_green)

        green_pixels = cv2.bitwise_and(enhanced_img, enhanced_img, mask=mask)
        non_green_mask = cv2.bitwise_not(mask)
        green_pixels[non_green_mask == 255] = [0, 0, 0]

        self.show_image(green_pixels)

    def save_and_apply_to_all(self):
        contrast = self.contrast_slider.get()
        lower_green_value = int(self.lower_green_slider.get())
        for filename in self.files:
            img_path = os.path.join(self.input_dir, filename)
            img = cv2.imread(img_path)

            # Crop the image
            left = int((img.shape[1] - self.crop_width) / 2)
            top = int((img.shape[0] - self.crop_height) / 2)
            right = left + self.crop_width
            bottom = top + self.crop_height
            img = img[top:bottom, left:right]

            clahe = cv2.createCLAHE(clipLimit=contrast, tileGridSize=(8, 8))
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

            hsv_img = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2HSV)
            lower_green = np.array([lower_green_value, 50, 50])
            upper_green = np.array([85, 255, 255])
            mask = cv2.inRange(hsv_img, lower_green, upper_green)

            green_pixels = cv2.bitwise_and(enhanced_img, enhanced_img, mask=mask)
            non_green_mask = cv2.bitwise_not(mask)
            green_pixels[non_green_mask == 255] = [0, 0, 0]

            output_path = os.path.join(self.output_dir, filename)
            cv2.imwrite(output_path, green_pixels)

        print("Processing complete. Images saved to", self.output_dir)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessor(root)
    root.mainloop()
