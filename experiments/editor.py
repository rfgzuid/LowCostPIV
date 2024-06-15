import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageEnhance, ImageOps


class PhotoEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("Batch Foto Editor")

        self.img = None
        self.original_img = None
        self.img_display = None

        self.load_button = tk.Button(root, text="Selecteer Invoer Map", command=self.select_input_folder)
        self.load_button.pack()

        self.process_button = tk.Button(root, text="Verwerk Afbeeldingen", command=self.process_images)
        self.process_button.pack()

        self.brightness_scale = tk.Scale(root, from_=-100, to=100, resolution=1, label="Helderheid",
                                         orient=tk.HORIZONTAL, command=self.update_image)
        self.brightness_scale.set(0)
        self.brightness_scale.pack()

        self.contrast_scale = tk.Scale(root, from_=-100, to=100, resolution=1, label="Contrast", orient=tk.HORIZONTAL,
                                       command=self.update_image)
        self.contrast_scale.set(0)
        self.contrast_scale.pack()

        self.shadows_scale = tk.Scale(root, from_=-100, to=100, resolution=1, label="Schaduw", orient=tk.HORIZONTAL,
                                      command=self.update_image)
        self.shadows_scale.set(0)
        self.shadows_scale.pack()

        self.highlights_scale = tk.Scale(root, from_=-100, to=100, resolution=1, label="Hoogtepunten",
                                         orient=tk.HORIZONTAL, command=self.update_image)
        self.highlights_scale.set(0)
        self.highlights_scale.pack()

        self.black_scale = tk.Scale(root, from_=-100, to=100, resolution=1, label="Zwart", orient=tk.HORIZONTAL,
                                    command=self.update_image)
        self.black_scale.set(0)
        self.black_scale.pack()

        self.green_scale = tk.Scale(root, from_=-100, to=100, resolution=1, label="Groen", orient=tk.HORIZONTAL,
                                    command=self.update_image)
        self.green_scale.set(0)
        self.green_scale.pack()

        self.canvas = tk.Canvas(root, width=600, height=400, bg='grey')
        self.canvas.pack()

        self.input_folder = None
        self.output_folder = None

    def select_input_folder(self):
        self.input_folder = filedialog.askdirectory()
        if self.input_folder:
            self.output_folder = os.path.join(self.input_folder, 'output')
            if not os.path.exists(self.output_folder):
                os.makedirs(self.output_folder)
            messagebox.showinfo("Invoer Map", f"Geselecteerde Invoer Map: {self.input_folder}")
            self.load_first_image()

    def load_first_image(self):
        for filename in os.listdir(self.input_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                file_path = os.path.join(self.input_folder, filename)
                self.original_img = Image.open(file_path)
                self.img = self.original_img.copy()
                self.display_image(self.img)
                break

    def display_image(self, img):
        self.img_display = ImageTk.PhotoImage(img.resize((600, 400)))
        self.canvas.create_image(300, 200, image=self.img_display)

    def update_image(self, value=None):
        if self.original_img:
            img = self.original_img.copy()
            img = ImageEnhance.Brightness(img).enhance(1 + self.brightness_scale.get() / 100)
            img = ImageEnhance.Contrast(img).enhance(1 + self.contrast_scale.get() / 100)
            img = self.adjust_shadows(img, self.shadows_scale.get())
            img = self.adjust_highlights(img, self.highlights_scale.get())
            img = self.adjust_black(img, self.black_scale.get())
            img = self.adjust_green(img, self.green_scale.get())
            self.img = img
            self.display_image(img)

    def adjust_shadows(self, img, value):
        if value == 0:
            return img
        factor = value / 100
        img = img.convert('RGBA')
        r, g, b, a = img.split()
        r = r.point(lambda i: i * (1 + factor) if i < 128 else i)
        g = g.point(lambda i: i * (1 + factor) if i < 128 else i)
        b = b.point(lambda i: i * (1 + factor) if i < 128 else i)
        img = Image.merge('RGBA', (r, g, b, a))
        return img

    def adjust_highlights(self, img, value):
        if value == 0:
            return img
        factor = value / 100
        img = img.convert('RGBA')
        r, g, b, a = img.split()
        r = r.point(lambda i: i + (255 - i) * factor if i > 128 else i)
        g = g.point(lambda i: i + (255 - i) * factor if i > 128 else i)
        b = b.point(lambda i: i + (255 - i) * factor if i > 128 else i)
        img = Image.merge('RGBA', (r, g, b, a))
        return img

    def adjust_black(self, img, value):
        if value == 0:
            return img
        factor = value / 100
        img = img.convert('RGBA')
        r, g, b, a = img.split()
        r = r.point(lambda i: i * (1 + factor))
        g = g.point(lambda i: i * (1 + factor))
        b = b.point(lambda i: i * (1 + factor))
        img = Image.merge('RGBA', (r, g, b, a))
        return img

    def adjust_green(self, img, threshold):
        img = img.convert('RGBA')
        r, g, b, a = img.split()
        r = r.point(lambda i: 0 if g.getpixel((i % img.width, i // img.width)) < threshold else i)
        b = b.point(lambda i: 0 if g.getpixel((i % img.width, i // img.width)) < threshold else i)
        g = g.point(lambda i: 0 if i < threshold else i)
        img = Image.merge('RGBA', (r, g, b, a))
        return img

    def process_images(self):
        if not self.input_folder or not self.output_folder:
            messagebox.showerror("Fout", "Selecteer een invoermap.")
            return

        for filename in os.listdir(self.input_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                file_path = os.path.join(self.input_folder, filename)
                img = Image.open(file_path)
                img = ImageEnhance.Brightness(img).enhance(1 + self.brightness_scale.get() / 100)
                img = ImageEnhance.Contrast(img).enhance(1 + self.contrast_scale.get() / 100)
                img = self.adjust_shadows(img, self.shadows_scale.get())
                img = self.adjust_highlights(img, self.highlights_scale.get())
                img = self.adjust_black(img, self.black_scale.get())
                img = self.adjust_green(img, self.green_scale.get())
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                output_path = os.path.join(self.output_folder, filename)
                img.save(output_path)

        messagebox.showinfo("Klaar", "Alle afbeeldingen zijn verwerkt en opgeslagen in de uitvoermap.")


if __name__ == "__main__":
    root = tk.Tk()
    editor = PhotoEditor(root)
    root.mainloop()
