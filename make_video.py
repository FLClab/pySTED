# This is a template for video generation

import cv2
import numpy as np
import glob
import os

# for i in range(0, 20):
#     os.mkdir(f"C:/Users/benoi/Documents/School/Automne 2020/Research_stuff/week_of_28_09/acq_plus_bleach/video/{i}")
# exit()

test = "C:/Users/benoi/Documents/School/Automne 2020/Research_stuff/week_of_28_09/acq_plus_bleach/video/0/0.png"
img_test = cv2.imread(test)
height_test, width_test, layers_test = img_test.shape
size_test = (width_test, height_test)

base_folder_path = "C:/Users/benoi/Documents/School/Automne 2020/Research_stuff/week_of_28_09/" \
                   "acq_plus_bleach/video/"
img_array = []
for i in range(0, 20):
    current_folder = base_folder_path + f"{i}/"
    all_images = sorted(glob.glob(current_folder + "*.png"), key=os.path.getmtime)
    for filename in all_images:
        # input(filename)
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)
    print(f"{i + 1} rows completed")

fps = 1
out = cv2.VideoWriter(base_folder_path + f'project_{fps}fps.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, size_test)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
