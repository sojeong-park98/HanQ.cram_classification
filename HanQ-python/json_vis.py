# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from dataloader import Image_loader, Videoloader
import cv2
import numpy as np
import math
import argparse
import os
import json
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import torch
from torch.autograd import Variable
from utils import BaseTransform
from ssd import build_ssd
net = build_ssd('test', 300, 4)    # initialize SSD
net.load_state_dict(torch.load('CRAM.pth'))
net.eval()
transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0))

# 길게치기 SHORTANGLE
# 앞돌리기 LONGANGLE
# 옆돌리기 SIDEANGLE
# 빗겨치기 BIASANGLE
# 뒤돌리기 OUTSIDEANGLE
# 횡단샷 CROSSTABLE
# 리버스 REVERSESHOT
# 더블쿠션 DOUBLECUSSIONSHOT
# 되돌아오기 PLATESHOT
# 1뱅크 ONEBANKSHOT
# 2뱅크 TWOBANKSHOT
# 3뱅크 THREEBANKSHOT
# 대회전 GRANDROTATION

path_dict={
    'SHORTANGLE':0,
    'LONGANGLE':1,
    'SIDEANGLE':2,
    'BIASANGLE':3,
    'OUTSIDEANGLE':4,
    'CROSSTABLE':5,
    'REVERSESHOT':6,
    'DOUBLECUSSIONSHOT':7,
    'PLATESHOT':8,
    'ONEBANKSHOT':9,
    'TWOBANKSHOT':10,
    'THREEBANKSHOT':11,
    'GRANDROTATION':12
}

path_to_name={
    0:'SHORTANGLE',
    1:'LONGANGLE',
    2:'SIDEANGLE',
    3:'BIASANGLE',
    4:'OUTSIDEANGLE',
    5:'CROSSTABLE',
    6:'REVERSESHOT',
    7:'DOUBLECUSSIONSHOT',
    8:'PLATESHOT',
    9:'ONEBANKSHOT',
    10:'TWOBANKSHOT',
    11:'THREEBANKSHOT',
    12:'GRANDROTATION'
}

parser = argparse.ArgumentParser(description='HanQ')
parser.add_argument('--input', default='../data/billiards_youtube/1', type=str)
parser.add_argument('--data_type', default='image', type=str)
parser.add_argument('--predict', action='store_true')
args = parser.parse_args()

WIDTH = 1000
HEIGHT = 500
BALL_CRASH_THRES = 60
TOP_CRASH_THRES = 30
BUTTOM_CRASH_THRES = 30
LEFT_CRASH_THRES = 30
RIGHT_CRASH_THRES = 30
START_VALOCITY_THRES = 10
SHOOTING_DISTANCE_THRES = 1000
START_TRACKING_THRES = 3
MIN_BALL_AREA = 200
MAX_BALL_AREA = 500
ACT_VALOCITY_THRES = 5


def read_simulation_result(path):
    with open(path, 'r') as f:
        data = json.load(f)
    red_x = np.uint(np.array(data["rBallPositions"][0::30])/2844*1000)
    red_y = np.uint(np.array(data["rBallPositions"][1::30])/1422*500)
    red_x=np.reshape(red_x, (red_x.shape+(1,)))
    red_y = np.reshape(red_y, (red_y.shape+(1,)))
    #self.ball_path_list['red']=np.concatenate((red_x,red_y), axis=1).tolist()

    yellow_x = np.uint(np.array(data["yBallPositions"][0::30]) / 2844 * 1000)
    yellow_y = np.uint(np.array(data["yBallPositions"][1::30]) / 1422 * 500)
    yellow_x = np.reshape(yellow_x, (yellow_x.shape + (1,)))
    yellow_y = np.reshape(yellow_y, (yellow_y.shape + (1,)))
    #self.ball_path_list['yellow'] = np.concatenate((yellow_x, yellow_y), axis=1).tolist()

    white_x = np.uint(np.array(data["wBallPositions"][0::30]) / 2844 * 1000)
    white_y = np.uint(np.array(data["wBallPositions"][1::30]) / 1422 * 500)
    white_x = np.reshape(white_x, (white_x.shape + (1,)))
    white_y = np.reshape(white_y, (white_y.shape + (1,)))
    #self.ball_path_list['white'] = np.concatenate((white_x, white_y), axis=1).tolist()
    return np.concatenate((red_x,red_y), axis=1).tolist(), np.concatenate((yellow_x, yellow_y), axis=1).tolist(), np.concatenate((white_x, white_y), axis=1).tolist()



# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    img = np.full((500,1000,3), 255, np.uint8)


    red, yellow, white = read_simulation_result('../communication/Python2Unity.json')

    #print(np.array(red).shape)

    cv2.polylines(img, np.array([red]), False, [0,0,255], 2)
    cv2.polylines(img, np.array([yellow]), False, [0, 255, 255], 2)
    cv2.polylines(img, np.array([white]), False, [255, 0, 0], 2)

    cv2.imshow("asdf", img)
    cv2.waitKey(0)



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
