import numpy as np
import cv2
import scipy.misc as misc
import glob
import os
import matplotlib.pyplot as plt

def stain_norm(source):

    source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
    img = np.zeros((source.shape[0], source.shape[1], 3))
    for x in range(source.shape[0]):
        for y in range(source.shape[1]):
            img[x, y] = source[x, y]

    source_mean = np.zeros(3)
    source_std = np.zeros(3)
    target_mean = [205.30114805, 153.23791293, 117.87448323]
    target_std = [25.92254658, 8.20056325, 4.47438932]
    for i in range(3):
        source_mean[i] = np.mean(img[:, :, i])
        source_std[i] = np.std(img[:, :, i])

    for i in range(3):
        img[:, :, i] = target_std[i] / source_std[i] * (img[:, :, i] - source_mean[i]) + target_mean[i]
        if i == 0:
            img[:, :, i] = np.clip(img[:, :, i], 0 ,255)
        elif i == 1:
            img[:, :, i] = np.clip(img[:, :, i], 42, 226)
        else:
            img[:, :, i] = np.clip(img[:, :, i], 20, 223)

    img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)

    return img


path = 'F:/breaKHis_random/train/100x/tubular_adenoma'
save_path = './train'
img_list = glob.glob(os.path.join(path, '*'))
for i in img_list:
    img = cv2.imread(i)
    img_name = i.split('/')[-1]
    save_name = os.path.join(save_path, img_name)
    output = stain_norm(img)
    print(save_name)
    cv2.imwrite(save_name, output)


