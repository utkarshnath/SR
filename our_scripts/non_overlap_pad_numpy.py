import os
import numpy as np
from PIL import Image
import time
import argparse

def extract_sub_images(img, width, height):
    # Convert image to numpy array
    img_array = np.array(img)

    # Get image size
    img_height, img_width, _ = img_array.shape

    # Calculate padding if necessary
    pad_width = 0 if img_width % width == 0 else width - (img_width % width)
    pad_height = 0 if img_height % height == 0 else height - (img_height % height)

    # Add padding if necessary
    if pad_width > 0 or pad_height > 0:
        pad_width_left = pad_width // 2
        pad_width_right = pad_width - pad_width_left
        pad_height_top = pad_height // 2
        pad_height_bottom = pad_height - pad_height_top
        img_array = np.pad(img_array, ((pad_height_top, pad_height_bottom), (pad_width_left, pad_width_right), (0, 0)), mode='constant')

    # Calculate the number of 512x512 sub-images
    rows = img_array.shape[0] // height
    cols = img_array.shape[1] // width

    # Create sub-images
    sub_images = []
    for i in range(rows):
        for j in range(cols):
            sub_img_array = img_array[i*height:(i+1)*height, j*width:(j+1)*width]
            
            # If the majority of pixels in the sub-image are non-padding, add to sub-images list
            if np.count_nonzero(sub_img_array) > width * height / 2:
                sub_images.append(Image.fromarray(sub_img_array))

    return sub_images

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Extract 512x512 sub-images from images in a directory.')
parser.add_argument('input_directory', type=str, help='The directory containing the input images.')
parser.add_argument('output_directory', type=str, help='The directory to save the output images.')
args = parser.parse_args()

# Define sub-image size
width, height = 512, 512

# Start time
start_time = time.time()

# Traverse directory
for foldername, subfolders, filenames in os.walk(args.input_directory):
    for filename in filenames:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Open image
            img = Image.open(os.path.join(foldername, filename))
            
            # Extract sub-images
            sub_images = extract_sub_images(img, width, height)
            
            # Create output folder
            relative_path = os.path.relpath(foldername, args.input_directory)
            output_folder = os.path.join(args.output_directory, relative_path)
            os.makedirs(output_folder, exist_ok=True)
            
            # Save sub-images
            for idx, sub_img in enumerate(sub_images):
                sub_img.save(os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_subimage_{idx}.png"))

# End time
end_time = time.time()

# Print the time taken
print(f'Time taken: {end_time - start_time} seconds')