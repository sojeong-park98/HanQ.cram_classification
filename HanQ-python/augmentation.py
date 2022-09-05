import torchvision.transforms.functional as F
import random
import cv2
import numpy as np

def find_table(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5, 0)
    box_binary = 255-cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)
    box_binary = cv2.dilate(box_binary, None, iterations=2)

    left_list = []
    right_list = []
    top_list = []
    buttom_list = []
    for idx, w in enumerate(box_binary):
        if len(w[w==255]) > len(w[w==0]):
            if idx < box_binary.shape[0]*0.3:
                top_list.append(idx)
            elif idx > box_binary.shape[0]*0.7:
                buttom_list.append(idx)
    for idx, h in enumerate(box_binary.transpose(1,0)):
        if len(h[h==255]) > len(h[h==0]):
            if idx < box_binary.shape[1] * 0.3:
                left_list.append(idx)
            elif idx > box_binary.shape[1] * 0.7:
                right_list.append(idx)
    left_list = np.array(left_list)
    right_list = np.array(right_list)
    top_list = np.array(top_list)
    buttom_list = np.array(buttom_list)
    top = top_list.max()
    buttom = buttom_list.min()
    left = left_list.max()
    right = right_list.min()

    img_cropped = img[top:buttom, left:right]
    img_cropped = cv2.resize(img_cropped, (1000, 500), cv2.INTER_LINEAR)

    return img_cropped

def resize(image):
    image = F.resize(image ,size=(1360, 769))

    return image

def random_contrast(image):
    if random.random() > 0.5:
        angle = float(random.randint(7, 13)/10)
        image = F.adjust_contrast(image, angle)
    # more transforms ...
    return image

def random_gamma(image):
    if random.random() > 0.5:
        angle = random.randint(1, 5)
        image = F.adjust_gamma(image, angle)
    # more transforms ...
    return image

def random_saturation(image):
    if random.random() > 0.5:
        angle = float(random.randint(7, 15)/10)
        image = F.adjust_saturation(image, angle)
    # more transforms ...
    return image