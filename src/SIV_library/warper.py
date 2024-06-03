import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw
import cv2
import numpy as np
import math
import os
import PIL
from io import BytesIO
# Create the main application class
class WarperApp(tk.Tk):
    def __init__(self):
        super().__init__()
        
        # Set the window title
        self.title("Warper")
        
        # Make the window full screen
        self.attributes('-fullscreen', True)
        
        # Create a top frame for the buttons and slider
        top_frame = tk.Frame(self, bg='black')
        top_frame.pack(side=tk.TOP, fill=tk.X)
        
        # Create a close button
        close_button = tk.Button(top_frame, text='X', fg='white', bg='red', command=self.close_app, font=('Helvetica', 12, 'bold'), bd=0)
        close_button.pack(side=tk.LEFT, padx=10, pady=10)
        
        # Create a frame to center the upload button
        center_frame = tk.Frame(top_frame, bg='black')
        center_frame.pack(side=tk.LEFT, padx=10, pady=10)
        
        # Create an upload video button
        upload_button = tk.Button(center_frame, text="Upload Video", fg='black', bg='white', font=('Helvetica', 12), bd=0, command=self.upload_video)
        upload_button.pack()

        # Create a frame for the slider and labels on the right
        right_frame = tk.Frame(top_frame, bg='black')
        right_frame.pack(side=tk.RIGHT, padx=10, pady=10)

        # Create a wider slider to adjust the image scale
        self.scale_var = tk.DoubleVar(value=1.0)
        self.scale_slider = tk.Scale(right_frame, from_=0.1, to=2.0, resolution=0.01, orient=tk.HORIZONTAL, variable=self.scale_var, command=self.update_image_scale, bg='black', fg='white', length=400)
        self.scale_slider.pack(side=tk.LEFT)

        # Label to display the current scale value
        self.scale_label = tk.Label(right_frame, text="Scale: 1.0", bg='black', fg='white')
        self.scale_label.pack(side=tk.LEFT, padx=10)

        # Labels to display the current width and height
        self.dimensions_label = tk.Label(right_frame, text="Width: N/A, Height: N/A", bg='black', fg='white')
        self.dimensions_label.pack(side=tk.LEFT, padx=10)
        
        # Create a canvas to display the video frame
        self.video_canvas = tk.Canvas(self, bg='white', highlightthickness=0)
        self.video_canvas.pack(side=tk.TOP, pady=20, expand=True, fill=tk.BOTH)

        # Bind mouse motion event to the canvas for magnifier
        self.video_canvas.bind("<Motion>", self.on_mouse_move)
        # Bind mouse click event to the canvas
        self.video_canvas.bind("<Button-1>", self.on_image_click)

        # Create a bottom frame for undo button and navigation buttons
        bottom_frame = tk.Frame(self, bg='black')
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Create frame navigation buttons and other widgets
        self.create_navigation_widgets(bottom_frame)

        # Create undo button
        self.undo_button = tk.Button(bottom_frame, text="Undo", fg='white', bg='red', font=('Helvetica', 12), bd=0, command=self.undo_last_point, state=tk.DISABLED)
        self.undo_button.pack(side=tk.RIGHT, pady=10)

        # Create warp button
        self.warp_button = tk.Button(bottom_frame, text="Warp Video", fg='white', bg='blue', font=('Helvetica', 12), bd=0, command=self.perform_perspective_transform, state=tk.DISABLED)
        self.warp_button.pack(side=tk.RIGHT, pady=10, padx=5)

        # Create magnifier window
        self.magnifier = tk.Toplevel(self)
        self.magnifier.withdraw()
        self.magnifier_canvas = tk.Canvas(self.magnifier, width=100, height=100, highlightthickness=0)
        self.magnifier_canvas.pack()

        self.original_image = None  # To store the original image
        self.points = []  # To store the points (x, y)
        self.scaled_points = [] # To store the scaled points (x, y)
        self.max_points = 4  # Maximum number of points allowed
        self.video_path = None  # To store the video file path
        self.current_frame_index = 0  # Current frame index
        self.total_frames = 0  # Total number of frames in the video

    # Method to create frame navigation widgets
    def create_navigation_widgets(self, parent):
        navigation_frame = tk.Frame(parent, bg='black')
        navigation_frame.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.prev_10_button = tk.Button(navigation_frame, text="<<", fg='black', bg='white', font=('Helvetica', 12), bd=0, command=lambda: self.navigate_frames(-10))
        self.prev_10_button.pack(side=tk.LEFT, padx=5)

        self.prev_1_button = tk.Button(navigation_frame, text="<", fg='black', bg='white', font=('Helvetica', 12), bd=0, command=lambda: self.navigate_frames(-1))
        self.prev_1_button.pack(side=tk.LEFT, padx=5)

        self.next_1_button = tk.Button(navigation_frame, text=">", fg='black', bg='white', font=('Helvetica', 12), bd=0, command=lambda: self.navigate_frames(1))
        self.next_1_button.pack(side=tk.LEFT, padx=5)

        self.next_10_button = tk.Button(navigation_frame, text=">>", fg='black', bg='white', font=('Helvetica', 12), bd=0, command=lambda: self.navigate_frames(10))
        self.next_10_button.pack(side=tk.LEFT, padx=5)
        
        # Frame label to display current frame number and total frames
        self.frame_label = tk.Label(navigation_frame, text="Frame: 0/0", bg='black', fg='white', font=('Helvetica', 12))
        self.frame_label.pack(side=tk.LEFT, padx=10)
        
        # Entry to input frame number to jump to
        self.frame_entry = tk.Entry(navigation_frame, width=5, font=('Helvetica', 12))
        self.frame_entry.pack(side=tk.LEFT, padx=5)
        self.frame_entry.bind("<Return>", self.go_to_frame)
        
        # Go button to jump to entered frame number
        self.go_button = tk.Button(navigation_frame, text="Go", fg='black', bg='white', font=('Helvetica', 12), bd=0, command=self.go_to_frame)
        self.go_button.pack(side=tk.LEFT, padx=5)

    # Method to close the application
    def close_app(self):
        self.destroy()
    
    # Method to upload a video
    def upload_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov;*.mkv")])
        if file_path:
            self.video_path = file_path
            self.show_first_frame(file_path)
    
    # Method to show the first frame of the video
    def show_first_frame(self, file_path):
        cap = cv2.VideoCapture(file_path)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ret, frame = cap.read()
        if ret:
            # Convert the frame to an image
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.original_image = Image.fromarray(frame)
            self.update_image_scale()
        cap.release()
        self.update_navigation_buttons()

    # Method to navigate through frames
    def navigate_frames(self, step):
        new_frame_index = self.current_frame_index + step
        if 0 <= new_frame_index < self.total_frames:
            self.current_frame_index = new_frame_index
            cap = cv2.VideoCapture(self.video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_index)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.original_image = Image.fromarray(frame)
                self.update_image_scale()
            cap.release()
        self.update_navigation_buttons()

    # Method to go to a specific frame
    def go_to_frame(self, event=None):
        try:
            frame_index = int(self.frame_entry.get())
            if 0 <= frame_index < self.total_frames:
                self.current_frame_index = frame_index
                cap = cv2.VideoCapture(self.video_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_index)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.original_image = Image.fromarray(frame)
                    self.update_image_scale()
                cap.release()
            self.update_navigation_buttons()
        except ValueError:
            pass  # Ignore invalid input

    # Method to update navigation buttons and frame label
    def update_navigation_buttons(self):
        self.prev_10_button.config(state=tk.NORMAL if self.current_frame_index >= 10 else tk.DISABLED)
        self.prev_1_button.config(state=tk.NORMAL if self.current_frame_index >= 1 else tk.DISABLED)
        self.next_1_button.config(state=tk.NORMAL if self.current_frame_index < self.total_frames - 1 else tk.DISABLED)
        self.next_10_button.config(state=tk.NORMAL if self.current_frame_index < self.total_frames - 10 else tk.DISABLED)
        self.frame_label.config(text=f"Frame: {self.current_frame_index}/{self.total_frames}")

    def update_image_scale(self, *args):
        if self.original_image:
            scale = self.scale_var.get()
            new_width = int(self.original_image.width * scale)
            new_height = int(self.original_image.height * scale)

            # Resize the image while maintaining the aspect ratio
            resized_image = self.original_image.resize((new_width, new_height), Image.LANCZOS)
            self.image_tk = ImageTk.PhotoImage(resized_image)

            # Clear the canvas and draw the resized image
            self.video_canvas.delete("all")
            self.video_canvas.create_image(0, 0, anchor=tk.NW, image=self.image_tk)

            # Update the scale and dimensions labels
            self.scale_label.config(text=f"Scale: {scale:.2f}")
            self.dimensions_label.config(text=f"Width: {new_width}, Height: {new_height}")

            # Clear the scaled points list
            self.scaled_points = []
            self.scaled_points = [(int(round(x * scale)), int(round(y * scale))) for x, y in self.points]

            # Redraw the scaled points
            for point in self.scaled_points:
                self.draw_point(point[0], point[1])

    # Method to handle mouse click event on the image canvas
    def on_image_click(self, event):
        if len(self.points) < self.max_points:
            # Check if the click occurred within the bounds of the scaled image
            if self.on_image(event.x, event.y):
                scale = self.scale_var.get()
                # Add a point (x, y) to the list
                self.points.append((event.x/scale, event.y/scale))
                self.draw_point(event.x, event.y)

                # Update the scaled points list
                self.scaled_points = [(int(round(x * scale)), int(round(y * scale))) for x, y in self.points]
                self.draw_point(self.scaled_points[-1][0], self.scaled_points[-1][1])

                # Enable the undo button
                self.undo_button.config(state=tk.NORMAL)

                if len(self.points) == self.max_points:
                    self.warp_button.config(state=tk.NORMAL)

    # Method to draw a red dot at the given coordinates on the image canvas
    def draw_point(self, x, y):
        # Create a red dot
        dot_size = 3
        self.video_canvas.create_oval(x - dot_size, y - dot_size, x + dot_size, y + dot_size, fill='red', outline='')

    # Method to undo the last point
    def undo_last_point(self):
        if self.points:
            scale = self.scale_var.get()
            # Remove the last point from the list
            self.points.pop()
            self.scaled_points = [(int(round(x * scale)), int(round(y * scale))) for x, y in self.points]
            # Clear the canvas and redraw all points except the removed one
            self.video_canvas.delete("all")
            if self.image_tk:
                self.video_canvas.create_image(0, 0, anchor=tk.NW, image=self.image_tk)
            for point in self.scaled_points:
                self.draw_point(point[0], point[1])

            # Disable the undo button if no points are left
            if not self.points:
                self.undo_button.config(state=tk.DISABLED)
            
            if len(self.points) < self.max_points:
                self.warp_button.config(state=tk.DISABLED)

    def on_image(self, x, y):
        if self.original_image is None:
            return False  # If no image is loaded, return False
        scale = self.scale_var.get()
        new_width = int(self.original_image.width * scale)
        new_height = int(self.original_image.height * scale)
        if 0 <= x < new_width and 0 <= y < new_height:
            return True
        return False

    def perform_perspective_transform(self):
        if len(self.points) == 4:
            pts1 = np.float32(self.points)
            width = round(math.hypot(pts1[0, 0] - pts1[1, 0], pts1[0, 1] - pts1[1, 1]))
            height = round(math.hypot(pts1[0, 0] - pts1[3, 0], pts1[0, 1] - pts1[3, 1]))

            x = pts1[0, 0]
            y = pts1[0, 1]

            pts2 = np.float32([[x, y], [x + width - 1, y], [x + width - 1, y + height - 1], [x, y + height - 1]])

            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            scale = self.scale_var.get()
            cap = cv2.VideoCapture(self.video_path)

            # Directory to save the frames
            output_dir = 'C:/Users/jortd/PycharmProjects/kloten/.venv/trampezium/Test Data/4302_warped'

            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count == 0 or (600 <= frame_count <= 700):
                    hh, ww = frame.shape[:2]
                    result = cv2.warpPerspective(frame, matrix, (ww, hh), cv2.INTER_LINEAR,
                                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
                    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                    result = cv2.resize(result, (int(result.shape[1] * scale), int(result.shape[0] * scale)),
                                        cv2.INTER_AREA)
                    result_image = Image.fromarray(result)

                    # Save the scaled frame
                    frame_path = os.path.join(output_dir, f"{frame_count:04d}.png")
                    result_image.save(frame_path)

                    result_image_tk = ImageTk.PhotoImage(result_image)
                    self.video_canvas.delete("all")
                    self.video_canvas.create_image(0, 0, anchor=tk.NW, image=result_image_tk)
                    self.update_idletasks()
                    self.update()

                frame_count += 1

            cap.release()

    # Method to handle mouse move event for magnifier
    def on_mouse_move(self, event):
        if self.on_image(event.x, event.y):
            self.show_magnifier(event.x, event.y)
        else:
            self.magnifier.withdraw()

    # Method to show the magnifier window
    def show_magnifier(self, x, y):
        if self.original_image is None:
            return  # If no image is loaded, do nothing
        scale = self.scale_var.get()
        magnify_scale = 2  # Magnification scale
        magnify_size = 100  # Size of the magnifier window

        # Calculate the region to be magnified
        start_x = int(max(x/scale - magnify_size/(2*magnify_scale), 0))
        start_y = int(max(y/scale - magnify_size/(2*magnify_scale), 0))
        end_x = int(min(start_x + magnify_size/magnify_scale, self.original_image.width))
        end_y = int(min(start_y + magnify_size/magnify_scale, self.original_image.height))

        region = self.original_image.crop((start_x, start_y, end_x, end_y))
        magnified_region = region.resize((magnify_size, magnify_size), Image.LANCZOS)
        
        # Add crosshair to the magnified region
        draw = ImageDraw.Draw(magnified_region)
        crosshair_color = "red"
        crosshair_size = 10
        draw.line((magnify_size/2 - crosshair_size, magnify_size/2, magnify_size/2 + crosshair_size, magnify_size/2), fill=crosshair_color)
        draw.line((magnify_size/2, magnify_size/2 - crosshair_size, magnify_size/2, magnify_size/2 + crosshair_size), fill=crosshair_color)

        magnified_image_tk = ImageTk.PhotoImage(magnified_region)
        self.magnifier_canvas.create_image(0, 0, anchor=tk.NW, image=magnified_image_tk)
        self.magnifier_canvas.image = magnified_image_tk

        # Position the magnifier window
        self.magnifier.geometry(f"+{self.winfo_rootx() + x + 20}+{self.winfo_rooty() + y + 20}")
        self.magnifier.deiconify()

# Create an instance of the application
if __name__ == "__main__":
    app = WarperApp()
    app.mainloop()
