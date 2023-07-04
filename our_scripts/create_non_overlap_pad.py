import os
import numpy as np
from PIL import Image

def extract_sub_images(img, width, height):
    # Get image size
    img_width, img_height = img.size

    # Calculate padding if necessary
    pad_width = 0 if img_width % width == 0 else width - (img_width % width)
    pad_height = 0 if img_height % height == 0 else height - (img_height % height)

    # Add padding if necessary
    if pad_width > 0 or pad_height > 0:
        new_size = (img_width + pad_width, img_height + pad_height)
        new_img = Image.new("RGB", new_size)
        new_img.paste(img, ((new_size[0]-img_width)//2, (new_size[1]-img_height)//2))
        img = new_img

    # Calculate the number of 512x512 sub-images
    cols = new_size[0] // width
    rows = new_size[1] // height

    # Create sub-images
    sub_images = []
    for i in range(rows):
        for j in range(cols):
            left = j * width
            upper = i * height
            right = left + width
            lower = upper + height
            
            sub_img = img.crop((left, upper, right, lower))
            sub_images.append(sub_img)

    return sub_images, pad_width, pad_height

# Define your directory path
input_directory = '/scratch/vgunda8/degraded_images/DIV2K/DIV2K_train_HR'
output_directory = '/scratch/vgunda8/non_overlap_pad/DIV2K_train_patches'

# Define sub-image size
width, height = 512, 512

# Traverse directory
for foldername, subfolders, filenames in os.walk(input_directory):
    for filename in filenames:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Open image
            img = Image.open(os.path.join(foldername, filename))
            
            # Extract sub-images
            sub_images, pad_width, pad_height = extract_sub_images(img, width, height)
            
            # Create output folder
            relative_path = os.path.relpath(foldername, input_directory)
            output_folder = os.path.join(output_directory, relative_path)
            os.makedirs(output_folder, exist_ok=True)
            
            # Save sub-images
            for idx, sub_img in enumerate(sub_images):
                # Convert the sub-image to a numpy array
                sub_img_array = np.array(sub_img)

                # Calculate the number of non-padding pixels
                num_non_padding_pixels = np.count_nonzero(sub_img_array)

                # If the majority of pixels in the sub-image are non-padding, save the sub-image
                if num_non_padding_pixels > width * height / 2:
                    sub_img.save(os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_subimage_{idx}.png"))
