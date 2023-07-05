import os
import dask.array as da
from PIL import Image
import time

def create_patches(img, width, height):
    # Convert PIL image to Dask array
    img_da = da.from_array(np.array(img), chunks=(height, width, 3))

    # Zero pad the image
    img_padded = da.pad(img_da, pad_width=((0, height - img_da.shape[0] % height), 
                                           (0, width - img_da.shape[1] % width), 
                                           (0, 0)), 
                        mode='constant')

    # Split the image into patches
    patches = [img_padded[i:i+height, j:j+width].compute() for i in range(0, img_padded.shape[0], height) 
                                                             for j in range(0, img_padded.shape[1], width)]

    return patches

# Define your directory path
input_directory = '/scratch/vgunda8/degraded_images/Flickr2K/Flickr2K_HR'
output_directory = '/scratch/vgunda8/non_overlap_pad_numpy/Flickr2K_train_patches'

# Define sub-image size
width, height = 512, 512

# Start time
start_time = time.time()

# Traverse directory
for foldername, subfolders, filenames in os.walk(input_directory):
    for filename in filenames:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Open image
            img = Image.open(os.path.join(foldername, filename))

            # Create patches
            patches = create_patches(img, width, height)

            # Create output folder
            relative_path = os.path.relpath(foldername, input_directory)
            output_folder = os.path.join(output_directory, relative_path)
            os.makedirs(output_folder, exist_ok=True)

            # Save patches
            for i, patch in enumerate(patches):
                patch_img = Image.fromarray(patch.astype('uint8'))
                patch_img.save(os.path.join(output_folder, f'{os.path.splitext(filename)[0]}_patch_{i}.png'))

# End time
end_time = time.time()

# Print the time taken
print(f'Time taken: {end_time - start_time} seconds')
