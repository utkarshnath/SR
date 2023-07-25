import cv2
import numpy as np
import os
import sys


def generatePatches(path):
    print(path)
    valid_count = 0
    total_count = 0
    crop_size = 512
    for index, subpath in enumerate(os.listdir(path)):
        print(index)
        img = cv2.imread(path+'/'+subpath)
        imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sobelxy = cv2.Sobel(src=imgray, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
        h, w = img.shape[:2]
        for i in range(0, h, crop_size):
            for j in range(0, w, crop_size):
                if i + crop_size >= h:
                   i = h-crop_size
                if j + crop_size >= w:
                   j = w - crop_size
                patch = sobelxy[i:i+crop_size, j:j+crop_size]
                mean = np.mean(patch)
                var = np.mean( (patch-mean)**2)
                var = np.var(patch)
                if np.var(patch) < 10:
                   valid_count += 1
                total_count += 1
                #print(subpath)
                #print(subpath.split('.')[0] + '_' + str(int(i/crop_size)) + '_' + str(int(j/crop_size)) + '_' + str(int(var)))
                writePath = path + '_patches/' + subpath.split('.')[0] + '_' + str(int(i/crop_size)) + '_' + str(int(j/crop_size)) + '_' + str(int(var)) + '.png'
                cv2.imwrite(writePath, img[i:i+crop_size, j:j+crop_size])
                #subpath noext = os.path.splitext (subpath) [0]
    print(valid_count, total_count)

# dir = '/scratch/vgunda8/FeMaSR_Datasets/original_dataset/Flickr2K_HR'
# filelist = ['trainHR_001to200', 'trainHR_1001to1200', 'trainHR_1201to1400', 'trainHR_1401to1500', 'trainHR_201to400', 'trainHR_401to600', 'trainHR_601to800', 'trainHR_801to1000']
# for name in filelist:
    # generatePatches(dir + name)
generatePatches(sys.argv[1])