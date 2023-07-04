import os
from PIL import Image

def extract_sub_images(img, width, height):
    # Get image size
    img_width, img_height = img.size
    
    # Calculate the number of 512x512 sub-images
    cols = img_width // width
    rows = img_height // height

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

    return sub_images

# Define your directory path
input_directory = '/scratch/vgunda8/degraded_images/DIV2K/DIV2K_train_HR'
output_directory = '/scratch/vgunda8/non_overlap_pad'

# Define sub-image size
width, height = 512, 512

# Traverse directory
for foldername, subfolders, filenames in os.walk(input_directory):
    for filename in filenames:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Open image
            img = Image.open(os.path.join(foldername, filename))
            
            # Extract sub-images
            sub_images = extract_sub_images(img, width, height)
            
            # Create output folder
            relative_path = os.path.relpath(foldername, input_directory)
            output_folder = os.path.join(output_directory, relative_path)
            os.makedirs(output_folder, exist_ok=True)
            
            # Save sub-images
            for idx, sub_img in enumerate(sub_images):
                sub_img.save(os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_subimage_{idx}.png"))
