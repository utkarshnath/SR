import os
import shutil
from natsort import natsorted
import sys

def filter_images(parent_dir, x):
    filtered_dir = parent_dir + '_filtered'
    os.makedirs(filtered_dir, exist_ok=True)

    files = [f for f in os.listdir(parent_dir) if os.path.isfile(os.path.join(parent_dir, f))]
    files = natsorted(files, key=lambda x: float(x.split('_')[-1].replace('.png', '')), reverse=True)

    for i in range(x, len(files)):
        shutil.move(os.path.join(parent_dir, files[i]), os.path.join(filtered_dir, files[i]))

# usage
filter_images(sys.argv[1], int(sys.argv[2]))
