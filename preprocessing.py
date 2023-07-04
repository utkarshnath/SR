import os
import cv2
import cv2
import os
from os import listdir
from PIL import Image
import numpy

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn
import torchvision.models as models
import torchaudio

import torchvision
import random
 
path = "/scratch/unath/TrainingSet_NewGT/"
save_path = "/scratch/unath/class_project/New_Train_Data/"

videos = os.listdir(path)
video_list = []
gt_list = []

image_list = []
label_list = []

# shortVD_wp_4,ShortVD_wp_61,ShortVD_np_12
# ShortVD_wp_2,
# "ShortVD_np_14","ShortVD_np_5",
# "ShortVD_np_13","ShortVD_wp_24","ShortVD_np_7",
# "ShortVD_np_10","ShortVD_np_6",
# "ShortVD_wp_49","ShortVD_np_8","ShortVD_wp_52","ShortVD_np_9","ShortVD_np_11"
def criteria(item):
  return int(item.split('_')[4])

videos = ["ShortVD_wp_4"]

for v in videos:
    print(v)
    video_list.append(path+v+"/"+v+".wmv")
    gt_list.append(path+v+"/GT")
for i, video in enumerate(video_list):
    captured_video = cv2.VideoCapture(video)
    success, image = captured_video.read()
    frame_num = 0
    print(success)
    while success:
      print(success)
      name = video.split('/')[-1]
      name = name[:-4]+"_"+str(frame_num)
      im = Image.fromarray(image)
      print(save_path+name+'.jpeg')
      im.save(save_path+name+'.jpeg')
      frame_num+=1
      image_list.append(image)
      success, image = captured_video.read()
      
    gt = os.listdir(gt_list[i])
    gt.sort(key=criteria)
    for label in gt:
      im = Image.open(gt_list[i]+"/"+label)
      imarray = numpy.array(im)
      label_list.append(imarray)
      im = Image.fromarray(imarray)
      im.save(save_path+"/"+label[:-5]+'.jpeg')
