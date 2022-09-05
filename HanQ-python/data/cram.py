"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""
from .config import HOME
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import pandas as pd
import os
from augmentation import *
from PIL import Image

CRAM_CLASSES = (  # always index 0
    'red', 'yellow', 'white')

# note: if you used our download scripts, this should be right
CRAM_ROOT = osp.join("../../data/cram_detection/")


class CRAMAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(CRAM_CLASSES, range(len(CRAM_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for obj in target:
            name = obj[0]
            bbox = obj[1:5]

            bndbox = []
            for i, pt in enumerate(bbox):
                # scale height or width
                cur_pt = pt / width if i % 2 == 0 else pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]

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

def getxyxy(data):
    return torch.tensor([[(data['red_x'].item()-data['radius'].item())/1000,
                          (data['red_y'].item()-data['radius'].item())/500,
                          (data['red_x'].item()+data['radius'].item())/1000,
                          (data['red_y'].item()+data['radius'].item())/500,1],
                         [(data['yellow_x'].item() - data['radius'].item())/1000,
                          (data['yellow_y'].item() - data['radius'].item())/500,
                          (data['yellow_x'].item() + data['radius'].item())/1000,
                          (data['yellow_y'].item() + data['radius'].item())/500,2],
                         [(data['white_x'].item() - data['radius'].item())/1000,
                          (data['white_y'].item() - data['radius'].item())/500,
                          (data['white_x'].item() + data['radius'].item())/1000,
                          (data['white_y'].item() + data['radius'].item())/500, 3]
                         ])

class CRAMDetection(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root,
                 transform=None, target_transform=CRAMAnnotationTransform(),
                 dataset_name='CRAM', split = 'train'):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name

        self.img_size = (1360, 769)
        self.img_path = '../../data/cram_detection/images/'
        self.label_path = '../../data/cram_detection/labels_rd/'
        self.split = split
        self.labels = recursive_glob(rootdir=self.label_path, suffix=".csv")
        self.name = 'cram'

        if split == 'train':
            train_list = pd.read_csv('../../data/cram_detection/train.txt', sep='\t')
            self.file_idx = train_list['idx']
            self.file_shoot = train_list['shoot']
        if split == 'test':
            test_list = pd.read_csv('../../data/cram_detection/test.txt', sep='\t')
            self.file_idx = test_list['idx']
            self.file_shoot = test_list['shoot']

        self.ids = list()
        self.gts = list()
        for file in self.file_idx:
            f = pd.read_csv(self.label_path + str(file) + '.csv', sep=',')
            time = f['time']
            for row in time[1:-1]:
                self.ids.append(f[f['time'] == row]['img'].item())
                self.gts.append([['red', f[f['time'] == row]['red_x'].item()-f[f['time'] == row]['radius'].item(), f[f['time'] == row]['red_y'].item()-f[f['time'] == row]['radius'].item(), f[f['time'] == row]['red_x'].item()+f[f['time'] == row]['radius'].item(), f[f['time'] == row]['red_y'].item()+f[f['time'] == row]['radius'].item()],
                                ['yellow', f[f['time'] == row]['yellow_x'].item()-f[f['time'] == row]['radius'].item(), f[f['time'] == row]['yellow_y'].item()-f[f['time'] == row]['radius'].item(), f[f['time'] == row]['yellow_x'].item()+f[f['time'] == row]['radius'].item(), f[f['time'] == row]['yellow_y'].item()+f[f['time'] == row]['radius'].item()],
                                ['white', f[f['time'] == row]['white_x'].item()-f[f['time'] == row]['radius'].item(), f[f['time'] == row]['white_y'].item()-f[f['time'] == row]['radius'].item(), f[f['time'] == row]['white_x'].item()+f[f['time'] == row]['radius'].item(), f[f['time'] == row]['white_y'].item()+f[f['time'] == row]['radius'].item()]])
    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)
        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]
        target = self.gts[index]

        #img = cv2.imread(self.img_path+img_id)
        img = find_table(np.uint8(resize(Image.open(self.img_path+img_id))))
        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)
        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width
        # return torch.from_numpy(img), target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(self.img_path+img_id, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        gt = self.target_transform(self.gts[index], 1, 1)
        return img_id, gt

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)
