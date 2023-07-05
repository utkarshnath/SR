import os
import numpy as np
from PIL import Image
from scipy import ndimage
import time
import argparse

def filter_image(img):
    # Convert image to grayscale
    img_gray = img.convert('L')

    # Convert image to numpy array
    img_array = np.array(img_gray)

    # Normalize pixel intensities
    # img_array = img_array / 255.0
    
    # Calculate Sobel edge map
    edge_map = np.hypot(ndimage.sobel(img_array, 0), ndimage.sobel(img_array, 1))

    # Calculate mean and variance of edge map
    mu = np.mean(edge_map)
    sigma_squared = np.var(edge_map)

    # Return whether patch should be kept
    return sigma_squared >= 10

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Filter images based on Sobel edge detection.')
parser.add_argument('input_directory', type=str, help='The directory containing the input images.')
parser.add_argument('output_directory', type=str, help='The directory to save the output images.')
args = parser.parse_args()

# Start time
start_time = time.time()

# Traverse directory
for foldername, subfolders, filenames in os.walk(args.input_directory):
    for filename in filenames:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Open image
            img = Image.open(os.path.join(foldername, filename))
            
            # Filter image
            if filter_image(img):
                # Create output folder
                relative_path = os.path.relpath(foldername, args.input_directory)
                output_folder = os.path.join(args.output_directory, relative_path)
                os.makedirs(output_folder, exist_ok=True)
                
                # Save image
                img.save(os.path.join(output_folder, filename))

# End time
end_time = time.time()

# Print the time taken
print(f'Time taken: {end_time - start_time} seconds')
