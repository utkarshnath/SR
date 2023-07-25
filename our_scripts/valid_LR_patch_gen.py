import os
import sys
import shutil
from basicsr.data.bsrgan_util import degradation_bsrgan_plus
import numpy as np
import random
import cv2

def process_images(parent_dir, scale_factor):
    # scale_factors = [2, 4]

    # Process images in the parent directory
    for root, dirs, files in os.walk(parent_dir):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                image_path = os.path.join(root, file)

                # Iterate over the scale factors
                # for scale_factor in scale_factors:
                # Call func1 to process the image
                img_gt = cv2.imread(image_path).astype(np.float32) / 255.
                img_gt = img_gt[:, :, [2, 1, 0]] # BGR to RGB
                
                img_lq, img_hq = degradation_bsrgan_plus(img_gt, sf = scale_factor, crop = False)
                img_lq = (img_lq[:, :, [2, 1, 0]] * 255).astype(np.uint8)

                # Create the new directory path for the processed image
                new_dir_path = root.replace(parent_dir, f'{parent_dir}_sf{scale_factor}')

                # Create the new directory if it doesn't exist
                os.makedirs(new_dir_path, exist_ok=True)

                # Save the processed image in the new directory with the same filename
                new_image_path = os.path.join(new_dir_path, file)
                print(f'Save {new_image_path}')
                cv2.imwrite(new_image_path, img_lq)

seed = 123
random.seed(seed)
np.random.seed(seed)
process_images(sys.argv[1], int(sys.argv[2]))
