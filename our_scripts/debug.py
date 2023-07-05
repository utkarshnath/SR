import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
from scipy import ndimage
import time
import argparse

def filter_and_show_edge_map(img_path):
    # Open the image
    img = Image.open(img_path)
    
    # Convert image to grayscale
    img_gray = img.convert('L')

    # Convert image to numpy array
    img_array = np.array(img_gray)

    # Calculate Sobel edge map
    edge_map = np.hypot(ndimage.sobel(img_array, 0), ndimage.sobel(img_array, 1))

    # Display the edge map
    plt.imshow(edge_map, cmap='gray')
    plt.title('Edge Map')
    plt.show()
    plt.imsave('/home/vgunda8/FeMaSR/our_scripts/11.png', edge_map, cmap='gray')
    # Calculate mean and variance of edge map
    mu = np.mean(edge_map)
    sigma_squared = np.var(edge_map)

    # Return whether patch should be kept
    return sigma_squared >= 10

result = filter_and_show_edge_map('/scratch/vgunda8/non_overlap_pad_numpy/DIV2K_train_patches/0001_subimage_11.png')
print('Keep image:', result)
