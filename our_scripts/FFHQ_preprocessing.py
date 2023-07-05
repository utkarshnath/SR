import os
import random
from PIL import Image
import argparse

# Function to process an image


def process_image(img_path, output_dir):
    # Open the image
    img = Image.open(img_path)

    # Randomly resize the image
    scale_factor = random.uniform(0.5, 1.0)
    new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
    img = img.resize(new_size, Image.ANTIALIAS)

    # Randomly crop a 512x512 patch from the image if large enough
    if img.width < 512 or img.height < 512:
        print(f'Image too small after resize: {img_path}')
        return

    left = random.randint(0, img.width - 512)
    top = random.randint(0, img.height - 512)
    img = img.crop((left, top, left + 512, top + 512))

    # Determine output path
    img_dir = os.path.dirname(img_path)
    relative_dir = os.path.relpath(img_dir, args.input_directory)
    output_folder = os.path.join(output_dir, relative_dir)
    output_path = os.path.join(output_folder, os.path.basename(img_path))

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Save the image
    img.save(output_path)
    print(f'Image saved: {output_path}')

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Randomly resize and crop images.')
parser.add_argument('input_directory', type=str, help='The directory containing the input images.')
parser.add_argument('output_directory', type=str, help='The directory to save the output images.')
args = parser.parse_args()

# Traverse the input directory and process each image
for dirpath, dirnames, filenames in os.walk(args.input_directory):
    for filename in filenames:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            process_image(os.path.join(dirpath, filename), args.output_directory)
            