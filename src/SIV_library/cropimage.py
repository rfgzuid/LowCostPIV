import os
from PIL import Image

def crop_images(input_folder, output_folder, crop_size=(400, 400)):    # Maak de output folder als deze nog niet bestaat
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop door alle bestanden in de input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Open de afbeelding
            with Image.open(input_path) as img:
                # Bereken de crop coördinaten
                width, height = img.size
                left = ((width - crop_size[0]) / 2)
                top = (height - crop_size[1]) / 2
                right = left + crop_size[0]
                bottom = (height + crop_size[1]) / 2

                # Crop en sla de afbeelding op
                cropped_img = img.crop((left, top, right, bottom))
                cropped_img.save(output_path)

    print(f'Alle afbeeldingen zijn bijgesneden en opgeslagen in {output_folder}')

# Gebruik de functie
input_folder = 'C:/Users/jortd/PycharmProjects/kloten/Test Data/2371_PROCESSED_warped'
output_folder = 'C:/Users/jortd/PycharmProjects/kloten/Test Data/2371_crop'

crop_images(input_folder, output_folder)
