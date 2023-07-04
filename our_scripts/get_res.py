import os
from PIL import Image

# Path to the main directory
main_dir = "/scratch/vgunda8/degraded_images/DIV2K/DIV2K_train_HR"

# Counter to keep track of the number of images printed per sub-folder
counter = 0

# Walk through each directory in the main directory
for dirpath, dirnames, filenames in os.walk(main_dir):
    
    # If the directory has image files
    if any(filename.lower().endswith(('.png', '.jpg', '.jpeg')) for filename in filenames):
        
        # Reset the counter
        counter = 0
        
        # For each file in the directory
        for filename in filenames:
            
            # If the file is an image and we haven't already printed 5 images
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')) and counter < 2650:
                
                # Open the image and get its size
                img = Image.open(os.path.join(dirpath, filename))
                width, height = img.size
                
                # Print the details
                print(f"Folder: {dirpath[25:]}, Image: {filename}, Width: {width}, Height: {height}")
                
                # Increment the counter
                counter += 1
        print("-------------------------------------------------------")
    print('======================================================================')
    
