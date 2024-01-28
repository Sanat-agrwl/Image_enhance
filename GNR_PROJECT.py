import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np


class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processor")

        # Variables
        self.x = None  # Original image
        self.y = None  # Processed image
        self.transforms_sequence = []

        # Widgets
        self.load_button = tk.Button(
            root, text="Load Image", command=self.load_image)
        self.load_button.pack(pady=10)

        self.grayscale_button = tk.Button(
            root, text="Grayscale", command=self.apply_grayscale_transform)
        self.grayscale_button.pack(pady=5)

        self.logarithmic_button = tk.Button(
            root, text="Logarithmic", command=self.apply_logarithmic_transform)
        self.logarithmic_button.pack(pady=5)

        self.exponential_button = tk.Button(
            root, text="Exponential", command=self.apply_exponential_transform)
        self.exponential_button.pack(pady=5)

        self.exponent_slider = tk.Scale(
            root, from_=1, to=10, orient="horizontal", label="Exponent", length=200)
        self.exponent_slider.set(5)  # Default value
        self.exponent_slider.pack(pady=5)

        self.linear_button = tk.Button(
            root, text="Linear", command=self.apply_linear_transform)
        self.linear_button.pack(pady=5)

        self.original_button = tk.Button(
            root, text="Revert to Original", command=self.revert_to_original)
        self.original_button.pack(pady=10)

        self.save_button = tk.Button(
            root, text="Save Image", command=self.save_image)
        self.save_button.pack(pady=10)

        self.image_label = tk.Label(root)
        self.image_label.pack()

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", ".png; .jpg; .jpeg; .gif")])
        if file_path:
            self.x = cv2.imread(file_path)
            self.y = self.x.copy()
            self.transforms_sequence = [self.y]
            self.display_image(self.y)

    def apply_grayscale_transform(self):
        if self.x is not None:
            self.y = self.x.copy()
            self.y = cv2.cvtColor(self.y, cv2.COLOR_BGR2GRAY)
            self.transforms_sequence.append(self.y)
            self.display_image(self.y)

    def apply_logarithmic_transform(self):
        if self.x is not None:
            self.y = self.x.copy()
            temp_image = cv2.cvtColor(self.y, cv2.COLOR_BGR2GRAY)
            temp_image = np.log1p(temp_image.astype(np.float32))
            temp_image = ((temp_image / (np.max(temp_image)-np.min(temp_image))) * 255) - \
                ((255*np.min(temp_image))/(np.max(temp_image)-np.min(temp_image)))
            self.y = temp_image.astype(np.uint8)
            self.transforms_sequence.append(self.y)
            self.display_image(self.y)

    def apply_exponential_transform(self):
        if self.x is not None:
            self.y = self.x.copy()
            temp_image = cv2.cvtColor(self.y, cv2.COLOR_BGR2GRAY)

            # Get exponent value from the slider
            exponent_value = self.exponent_slider.get()

            # Apply exponential transformation with custom exponent
            temp_image = (temp_image.astype(np.float32)) ** exponent_value
            temp_image = ((temp_image / (np.max(temp_image) - np.min(temp_image))) * 255) - \
                ((255 * np.min(temp_image)) /
                 (np.max(temp_image) - np.min(temp_image)))

            self.y = temp_image.astype(np.uint8)
            self.transforms_sequence.append(self.y)
            self.display_image(self.y)

    def apply_linear_transform(self):
        if self.x is not None:
            self.y = self.x.copy()
            self.y = cv2.cvtColor(self.y, cv2.COLOR_BGR2GRAY)
            self.y = cv2.normalize(self.y, None, 0, 255, cv2.NORM_MINMAX)
            self.transforms_sequence.append(self.y)
            self.display_image(self.y)

    def revert_to_original(self):
        if self.x is not None:
            self.y = self.x.copy()
            self.transforms_sequence = [self.y]
            self.display_image(self.y)

    def display_image(self, image):
        # Resize the image to fit within a fixed-size box
        max_width = 600  # Adjust this value based on your preferred size
        max_height = 400
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        # Resize while maintaining aspect ratio
        width, height = pil_image.size
        aspect_ratio = width / height
        if width > max_width or height > max_height:
            if aspect_ratio > 1:
                new_width = max_width
                new_height = int(max_width / aspect_ratio)
            else:
                new_height = max_height
                new_width = int(max_height * aspect_ratio)
            pil_image = pil_image.resize(
                (new_width, new_height), Image.ANTIALIAS)

        tk_image = ImageTk.PhotoImage(pil_image)
        self.image_label.config(image=tk_image)
        self.image_label.image = tk_image

    def save_image(self):
        if self.y is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                     filetypes=[("PNG files", "*.png"),
                                                                ("JPEG files", ".jpg;.jpeg")])
            if file_path:
                cv2.imwrite(file_path, self.y)


def main():

    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
