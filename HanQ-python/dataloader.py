# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import numpy as np
from PIL import Image, ImageEnhance
import cv2
from skimage import color
import torchvision.transforms as transforms
from skimage import io, color

def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)  # os.walk: traversal all files in rootdir and its subfolders
        for filename in filenames
        if filename.endswith(suffix)
    ]

class Image_loader():
    def __init__(self, source):

        self.root = source
        #self.img_size = (1280, 680)
        self.img_size = (1360, 769)

        self.files = recursive_glob(rootdir=self.root, suffix=".bmp")
        self.mean = np.array([48.3, 38.5, 47.8])
        self.std = np.array([26.2, 28.5, 37.0])

    def __len__(self):
        """__len__"""
        return len(self.files)

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        img_path = self.files[index].rstrip()
        img = cv2.imread(img_path, 0)

        img_color = cv2.imread(img_path, 1)
        if np.sum(img_color[:,:,0]-img_color[:,:,1])==0 and np.sum(img_color[:,:,0]-img_color[:,:,2])==0:
            img_color = cv2.cvtColor(img, cv2.COLOR_BayerRG2RGB)
            img_color = cv2.convertScaleAbs(img_color, alpha=3.4, beta=18)
            img_color = cv2.resize(img_color, self.img_size, interpolation=cv2.INTER_CUBIC)
            lab = cv2.cvtColor(img_color, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge((l, a, b))
            img_color = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            img_color = cv2.medianBlur(img_color, 3)
            img_color[:,:,1]-=45
            # img_color[:, :, 2] += 5
            img_color[img_color<0]=0
            img_color[img_color>255]=255
        img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_CUBIC)


        # cv2.imshow("img", img_color)
        # cv2.waitKey(0)
        return img, img_color, img_path.split('/')[-1]

    '''def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        img_path = self.files[index].rstrip()
        img = Image.open(img_path)
        img = img.resize(self.img_size, Image.BILINEAR)
        #Contrast_enhancer = ImageEnhance.Contrast(img)
        #img = img.Contrast_enhancer(2)

        img = np.array(img, dtype=np.uint8)
        img_color = cv2.cvtColor(img, cv2.COLOR_BayerRG2RGB)

        return img'''

class Videoloader():
    def __init__(self, source):

        self.root = source
        #self.img_size = (1280, 680)
        self.img_size = (1360, 769)
        self.video_len = 0
        self.cap = cv2.VideoCapture(self.root)

        self.tgt_mean = [69.065254, 14.45949796, 52.52039253]
        self.tgt_std = [2.24307508, 1.34125175, 4.41439993]

    def __len__(self):
        """__len__"""
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_fps(self):
        """__len__"""
        return int(self.cap.get(cv2.CAP_PROP_FPS))

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        if self.cap.isOpened():
            _, _ = self.cap.read()
            _, _ = self.cap.read()
            ret, frame = self.cap.read()
            if ret:
                img_color = cv2.resize(frame, dsize=self.img_size, interpolation=cv2.INTER_LINEAR)
                img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
                img_color = np.float64(img_color)
                #img_color *= np.float64([4.0, 2.5, 2.5])
                img_color[img_color > 255] = 255
                img_color[img_color < 0] = 0
                img_color = np.uint8(img_color)

                return img, img_color, index
            else:
                self.cap.release()
