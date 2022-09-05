# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from dataloader import Image_loader, Videoloader
import cv2
import numpy as np
import math
import argparse
import os
import pandas as pd
import time
import warnings
warnings.filterwarnings('ignore')

import torch
from torch.autograd import Variable
from utils import BaseTransform
from ssd import build_ssd
from threading import Thread

import json
net = build_ssd('test', 300, 4).cuda()    # initialize SSD
net.load_state_dict(torch.load('CRAM_background2.pth'))
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
    'GRANDROTATION':12,
    'OTHERSHOT': 13
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
    12:'GRANDROTATION',
    13:'OTHERSHOT'
}

parser = argparse.ArgumentParser(description='HanQ')
parser.add_argument('--input', default='../data/billiards_youtube/1', type=str)
parser.add_argument('--data_type', default='video', type=str)
parser.add_argument('--print_log', action='store_true')
args = parser.parse_args()

WIDTH = 1000
HEIGHT = 500
BALL_CRASH_THRES = 100
BALL_CRASH_THRES_B = 100
TOP_CRASH_THRES = 30
BUTTOM_CRASH_THRES = 30
LEFT_CRASH_THRES = 30
RIGHT_CRASH_THRES = 30
START_VALOCITY_THRES = 10
SHOOTING_DISTANCE_THRES = 1000
START_TRACKING_THRES = 3
MIN_BALL_AREA = 200
MAX_BALL_AREA = 500
ACT_VALOCITY_THRES = 3
ACT_VALOCITY_THRES_S = 1

class Event_checker():
    def __init__(self, ball_path_list, radius):

        self.time=0
        self.first_ball = None
        self.second_ball = None
        self.third_ball = None

        self.first_ball_save = None
        self.second_ball_save = None
        self.third_ball_save = None


        self.score = None
        self.cussion_count = 0
        self.check_spin = False
        self.kiss = False
        self.shootpoint = 'center'

        self.ball_radius = radius

        self.dist_white_yellow = 99999
        self.dist_white_red = 99999
        self.dist_yellow_red = 99999

        self.valocity = {'red': -1, 'yellow':-1, 'white':-1}

        self.ball_path_list = ball_path_list

        self.init_ball_pos = None

        self.event = {"event_num":0}
        self.is_event=False

        self.captured_img = None
        self.final_path = None
        self.is_round_table = False

        # 실패시
        self.closest = -1
        self.final_event = None

        self.fail_description = None

        self.direction = None
        self.tmp=None
        self.prev_img = None

        self.pd_data = pd.DataFrame(columns=['img', 'time', 'red_x',
                         'red_y', 'yellow_x', 'yellow_y', 'white_x', 'white_y', 'radius'])


    def tracker(self, img, show_img, ball_list):
        #if len(ball_list) == START_TRACKING_THRES:

        '''red_dist = np.sqrt(np.sum(
            np.square(np.array(self.ball_path_list['red'][-1]) - np.array([ball_list[0]['x'], ball_list[0]['y']]))))
        yellow_dist = np.sqrt(np.sum(
            np.square(np.array(self.ball_path_list['yellow'][-1]) - np.array([ball_list[1]['x'], ball_list[1]['y']]))))
        white_dist = np.sqrt(np.sum(
            np.square(np.array(self.ball_path_list['white'][-1]) - np.array([ball_list[2]['x'], ball_list[2]['y']]))))'''

        if len(ball_list)!=3:
            return show_img

        if 'red' in ball_list:
            self.ball_path_list['red'].append([ball_list['red']['x'], ball_list['red']['y']])
            show_img = cv2.polylines(show_img, [np.array(self.ball_path_list['red'])], False, (0, 0, 1), 1)
        if 'yellow' in ball_list:
            self.ball_path_list['yellow'].append([ball_list['yellow']['x'], ball_list['yellow']['y']])
            show_img = cv2.polylines(show_img, [np.array(self.ball_path_list['yellow'])], False, (0, 1, 1), 1)
        if 'white' in ball_list:
            self.ball_path_list['white'].append([ball_list['white']['x'], ball_list['white']['y']])
            show_img = cv2.polylines(show_img, [np.array(self.ball_path_list['white'])], False, (1, 1, 1), 1)

        return show_img

    def capture_shoot(self, img, img_color):
        mean = np.median(img_color, axis=(0, 1))
        # print(img_color[np.sum(np.abs(img_color-mean), axis=2)<50].shape)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        masked_img = img_color.copy()
        masked_img[np.sum(np.abs(img_color - mean), axis=2) < 200] = 0

        mask = np.array(img_color[:, :, 0] > img_color[:, :, 1], dtype=np.uint8) * 255
        blurred = cv2.medianBlur(masked_img, 11, 0)
        # blurred = cv2.GaussianBlur(blurred, (0, 0), 1.0)
        gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
        _, img_binary = cv2.threshold(gray, 10, 255, 0)
        if len(img_binary[img_binary==255])>len(img_binary[img_binary==0]):
            img_binary = 255 - img_binary
        enum = {
            'red':0,
            'yellow':1,
            'white':2
        }

        # 공 주변 10픽셀 이외에 다 지우고 시각화해서 확인
        cv2.circle(img_binary, (int(self.ball_path_list['red'][-1][0]), int(self.ball_path_list['red'][-1][1])), int(self.ball_radius+10), 0, -1)
        cv2.circle(img_binary, (int(self.ball_path_list['red'][-1][0]), int(self.ball_path_list['red'][-1][1])), int(self.ball_radius+10), 0, 3)

        cv2.circle(img_binary, (int(self.ball_path_list['yellow'][-1][0]), int(self.ball_path_list['yellow'][-1][1])), int(self.ball_radius+10), 0, -1)
        cv2.circle(img_binary, (int(self.ball_path_list['yellow'][-1][0]), int(self.ball_path_list['yellow'][-1][1])), int(self.ball_radius+10), 0, 3)

        cv2.circle(img_binary, (int(self.ball_path_list['white'][-1][0]), int(self.ball_path_list['white'][-1][1])), int(self.ball_radius+10), 0, -1)
        cv2.circle(img_binary, (int(self.ball_path_list['white'][-1][0]), int(self.ball_path_list['white'][-1][1])), int(self.ball_radius+10), 0, 3)


        if self.init_ball_pos != None:
            cv2.circle(img_binary,
                   (self.init_ball_pos[0]['x'], self.init_ball_pos[0]['y']),
                   self.init_ball_pos[0]['radius']+10, 0, -1)
            cv2.circle(img_binary,
                   (self.init_ball_pos[0]['x'], self.init_ball_pos[0]['y']),
                   self.init_ball_pos[0]['radius']+10, 0, 3)

            cv2.circle(img_binary,
                       (self.init_ball_pos[1]['x'], self.init_ball_pos[1]['y']),
                       self.init_ball_pos[1]['radius']+10, 0, -1)
            cv2.circle(img_binary,
                       (self.init_ball_pos[1]['x'], self.init_ball_pos[1]['y']),
                       self.init_ball_pos[1]['radius']+10, 0, 3)

            cv2.circle(img_binary,
                       (self.init_ball_pos[2]['x'], self.init_ball_pos[2]['y']),
                       self.init_ball_pos[2]['radius']+10, 0, -1)
            cv2.circle(img_binary,
                       (self.init_ball_pos[2]['x'], self.init_ball_pos[2]['y']),
                       self.init_ball_pos[2]['radius']+10, 0, 3)

        # img_binary[:,:ball_list[enum[self.first_ball['color']]]['x']-100]=0
        # img_binary[:, ball_list[enum[self.first_ball['color']]]['x'] + 100:] = 0
        # img_binary[:ball_list[enum[self.first_ball['color']]]['y'] - 100, :] = 0
        # img_binary[ball_list[enum[self.first_ball['color']]]['x'] + 100:, :] = 0

        return img_binary

    def calc_distance(self):
        self.dist_white_yellow = np.sqrt(np.sum(np.square(np.array(self.ball_path_list['yellow'][-1]) - np.array(self.ball_path_list['white'][-1]))))
        self.dist_white_red = np.sqrt(np.sum(np.square(np.array(self.ball_path_list['white'][-1]) - np.array(self.ball_path_list['red'][-1]))))
        self.dist_yellow_red = np.sqrt(np.sum(np.square(np.array(self.ball_path_list['yellow'][-1]) - np.array(self.ball_path_list['red'][-1]))))

    def calc_valocity(self):
        if len(self.ball_path_list['white'])>2 and len(self.ball_path_list['yellow'])>2 and len(self.ball_path_list['red'])>2:
            self.valocity['white'] = np.sqrt(np.sum(np.square(np.array(self.ball_path_list['white'][-1]) - np.array(self.ball_path_list['white'][-2]))))
            self.valocity['red'] = np.sqrt(np.sum(np.square(np.array(self.ball_path_list['red'][-1]) - np.array(self.ball_path_list['red'][-2]))))
            self.valocity['yellow'] = np.sqrt(np.sum(np.square(np.array(self.ball_path_list['yellow'][-1]) - np.array(self.ball_path_list['yellow'][-2]))))
        else:
            self.valocity['white'] = [0,0]
            self.valocity['red'] = [0,0]
            self.valocity['yellow'] = [0,0]

    def check_ball_left_right(self):
        v1 = np.array(self.first_ball['shoot_position']) - np.array(self.first_ball['position'])
        v2 = np.array(self.second_ball['position']) - np.array(self.first_ball['position'])
        theta = np.arcsin((v1[0]*v2[1]-v1[1]*v2[0])/(np.sqrt(v1[0]**2+v1[1]**2)*np.sqrt(v2[0]**2+v2[1]**2)))
        if theta < 0:
            self.event[self.event['event_num']]['hit_point'] = 'right'
        else:
            self.event[self.event['event_num']]['hit_point'] = 'left'
        #print(self.event[self.event['event_num']]['hit_point'])
        self.event['event_num'] += 1
        self.is_event = True

    def calc_theta(self, v1, v2):
        theta = np.arcsin(
            (v1[0] * v2[1] - v1[1] * v2[0]) / (np.sqrt(v1[0] ** 2 + v1[1] ** 2) * np.sqrt(v2[0] ** 2 + v2[1] ** 2)))
        return theta

    def check_first_ball(self, ball_path_list):
        if len(ball_path_list['red'])>=START_TRACKING_THRES and len(self.ball_path_list['yellow'])>=START_TRACKING_THRES and len(self.ball_path_list['white'])>=START_TRACKING_THRES:
            if self.valocity['yellow'] >= START_VALOCITY_THRES:
                self.first_ball = {'color': 'yellow', 'position':self.ball_path_list['yellow'][-1]}
                self.event[self.event['event_num']] = {'name': 'shoot', 'discribe': '첫번째 공 노란색', 'pos': self.ball_path_list['yellow'][-1], 'time': self.time}
                if args.print_log:
                    print("타구: ", self.event[self.event['event_num']]['discribe'])
                self.event['event_num'] += 1
            elif self.valocity['white'] >= START_VALOCITY_THRES:
                self.first_ball = {'color': 'white', 'position':self.ball_path_list['white'][-1]}
                self.event[self.event['event_num']] = {'name': 'shoot', 'discribe': '첫번째 공 하얀색', 'pos': self.ball_path_list['white'][-1], 'time': self.time}
                if args.print_log:
                    print("타구: ", self.event[self.event['event_num']]['discribe'])
                self.event['event_num'] += 1
    def check_second_ball(self, crash_thres, act_thres):
        if self.first_ball['color'] == 'yellow':
            if self.dist_yellow_red < crash_thres and self.valocity['red']>act_thres:
                self.second_ball = {'color': 'red', 'position':self.ball_path_list['red'][-1]}
                self.first_ball['shoot_position']=self.ball_path_list['yellow'][-1]
                self.event[self.event['event_num']] = {'name': 'object1', 'discribe': '제1목적구 빨간공', 'pos': self.ball_path_list['yellow'][-1], 'time': self.time}
                if args.print_log:
                    print("충돌: ", self.event[self.event['event_num']]['discribe'])
            elif self.dist_white_yellow < crash_thres and self.valocity['white']>act_thres:
                self.second_ball = {'color': 'white', 'position':self.ball_path_list['white'][-1]}
                self.first_ball['shoot_position'] = self.ball_path_list['yellow'][-1]
                self.event[self.event['event_num']] = {'name': 'object1', 'discribe': '제1목적구 흰공', 'pos': self.ball_path_list['yellow'][-1], 'time': self.time}
                if args.print_log:
                    print("충돌: ", self.event[self.event['event_num']]['discribe'])
        if self.first_ball['color'] == 'white':
            if self.dist_white_red < crash_thres and self.valocity['red']>act_thres:
                self.second_ball = {'color': 'red', 'position':self.ball_path_list['red'][-1]}
                self.first_ball['shoot_position'] = self.ball_path_list['white'][-1]
                self.event[self.event['event_num']] = {'name': 'object1', 'discribe': '제1목적구 빨간공', 'pos': self.ball_path_list['white'][-1], 'time': self.time}
                if args.print_log:
                    print("충돌: ", self.event[self.event['event_num']]['discribe'])
            if self.dist_white_yellow < crash_thres and self.valocity['yellow']>act_thres:
                self.second_ball = {'color': 'yellow', 'position':self.ball_path_list['yellow'][-1]}
                self.first_ball['shoot_position'] = self.ball_path_list['white'][-1]
                self.event[self.event['event_num']] = {'name': 'object1', 'discribe': '제1목적구 노란공', 'pos': self.ball_path_list['white'][-1], 'time': self.time}
                if args.print_log:
                    print("충돌: ", self.event[self.event['event_num']]['discribe'])
    def check_third_ball(self, crash_thres, act_thres):
        if self.first_ball['color'] == 'yellow':
            if self.second_ball['color'] == 'red':
                if self.dist_white_yellow < crash_thres:
                    if self.valocity['white'] < act_thres:
                        return
                    self.third_ball = {'color': 'white', 'position':self.ball_path_list['white'][-1]}
                    self.event[self.event['event_num']] = {'name': 'object2', 'discribe': '제2목적구 흰공', 'pos': self.ball_path_list['yellow'][-1], 'time': self.time}
                    if args.print_log:
                        print("충돌: ", self.event[self.event['event_num']]['discribe'])
                    self.event['event_num'] += 1
                    self.is_event = True
            elif self.second_ball['color'] == 'white':
                if self.dist_yellow_red < crash_thres:
                    if self.valocity['red'] < act_thres:
                        return
                    self.third_ball = {'color': 'red', 'position':self.ball_path_list['red'][-1]}
                    self.event[self.event['event_num']] = {'name': 'object2', 'discribe': '제2목적구 빨간공', 'pos': self.ball_path_list['yellow'][-1], 'time': self.time}
                    if args.print_log:
                        print("충돌: ", self.event[self.event['event_num']]['discribe'])
                    self.event['event_num'] += 1
                    self.is_event = True
        if self.first_ball['color'] == 'white':
            if self.second_ball['color'] == 'red':
                if self.dist_white_yellow < crash_thres:
                    if self.valocity['yellow'] < act_thres:
                        return
                    self.third_ball = {'color': 'yellow', 'position':self.ball_path_list['yellow'][-1]}
                    self.event[self.event['event_num']] = {'name': 'object2', 'discribe': '제2목적구 노란공', 'pos': self.ball_path_list['white'][-1], 'time': self.time}
                    if args.print_log:
                        print("충돌: ", self.event[self.event['event_num']]['discribe'])
                    self.event['event_num'] += 1
                    self.is_event = True
            elif self.second_ball['color'] == 'yellow':
                if self.dist_white_red < crash_thres:
                    if self.valocity['red'] < act_thres:
                        return
                    self.third_ball = {'color': 'red', 'position':self.ball_path_list['red'][-1]}
                    self.event[self.event['event_num']] = {'name': 'object2', 'discribe': '제2목적구 빨간공', 'pos': self.ball_path_list['white'][-1], 'time': self.time}
                    if args.print_log:
                        print("충돌: ", self.event[self.event['event_num']]['discribe'])
                    self.event['event_num'] += 1
                    self.is_event = True

    def check_bend_path(self):
        rad_prev = math.atan2(self.ball_path_list[self.first_ball['color']][-3][1] - self.ball_path_list[self.first_ball['color']][-2][1], self.ball_path_list[self.first_ball['color']][-3][0] - self.ball_path_list[self.first_ball['color']][-2][0])
        rad_cur = math.atan2(self.ball_path_list[self.first_ball['color']][-2][1] - self.ball_path_list[self.first_ball['color']][-1][1], self.ball_path_list[self.first_ball['color']][-2][0] - self.ball_path_list[self.first_ball['color']][-1][0])
        deg_prev = (rad_prev*180)/math.pi
        deg_cur = (rad_cur*180)/math.pi

        return np.abs(deg_prev-deg_cur)

    def check_cussion(self):
        is_cussion = False

        # print("거리,", np.abs(self.ball_path_list[self.first_ball['color']][-1][1]) < TOP_CRASH_THRES, "각도,", (self.ball_path_list[self.first_ball['color']][-1][1]-self.ball_path_list[self.first_ball['color']][-2][1])\
        #        *(self.ball_path_list[self.first_ball['color']][-2][1]-self.ball_path_list[self.first_ball['color']][-3][1]) <= 0, "직선,", self.ball_path_list[self.first_ball['color']][-1][1]!=self.ball_path_list[self.first_ball['color']][-2][1])
        if np.abs(self.ball_path_list[self.first_ball['color']][-2][1]) < TOP_CRASH_THRES \
                and ((self.ball_path_list[self.first_ball['color']][-1][1] -
                      self.ball_path_list[self.first_ball['color']][-2][1]) \
                     * (self.ball_path_list[self.first_ball['color']][-2][1] -
                        self.ball_path_list[self.first_ball['color']][-3][1]) <= 0 and
                     self.ball_path_list[self.first_ball['color']][-2][1] !=
                     self.ball_path_list[self.first_ball['color']][-3][1]):
            if self.event[self.event['event_num'] - 1]['name'] == 'cussion' and self.event[self.event['event_num'] - 1][
                'cussion'] == 'top':
                return
            self.cussion_count += 1
            self.check_spin = 'top'
            self.event[self.event['event_num']] = {'name': 'cussion', 'discribe': '윗쿠션 침', 'cussion': 'top',
                                                   'pos': self.ball_path_list[self.first_ball['color']][-1],
                                                   'time': self.time}
            if args.print_log:
                print("쿠션: ", self.event[self.event['event_num']]['discribe'])
            self.event['event_num'] += 1
            is_cussion = True
        if np.abs(HEIGHT - self.ball_path_list[self.first_ball['color']][-2][1]) < BUTTOM_CRASH_THRES \
                and ((self.ball_path_list[self.first_ball['color']][-1][1] -
                      self.ball_path_list[self.first_ball['color']][-2][1]) \
                     * (self.ball_path_list[self.first_ball['color']][-2][1] -
                        self.ball_path_list[self.first_ball['color']][-3][1]) <= 0 and
                     self.ball_path_list[self.first_ball['color']][-2][1] !=
                     self.ball_path_list[self.first_ball['color']][-3][1]):
            if self.event[self.event['event_num'] - 1]['name'] == 'cussion' and self.event[self.event['event_num'] - 1][
                'cussion'] == 'buttom':
                return
            self.cussion_count += 1
            self.check_spin = 'buttom'
            self.event[self.event['event_num']] = {'name': 'cussion', 'discribe': '아래 쿠션 침', 'cussion': 'buttom',
                                                   'pos': self.ball_path_list[self.first_ball['color']][-1],
                                                   'time': self.time}
            if args.print_log:
                print("쿠션: ", self.event[self.event['event_num']]['discribe'])
            self.event['event_num'] += 1
            is_cussion = True
        if np.abs(self.ball_path_list[self.first_ball['color']][-2][0]) < LEFT_CRASH_THRES \
                and ((self.ball_path_list[self.first_ball['color']][-1][0] -
                      self.ball_path_list[self.first_ball['color']][-2][0]) \
                     * (self.ball_path_list[self.first_ball['color']][-2][0] -
                        self.ball_path_list[self.first_ball['color']][-3][0]) <= 0 and
                     self.ball_path_list[self.first_ball['color']][-2][0] !=
                     self.ball_path_list[self.first_ball['color']][-3][0]):
            if self.event[self.event['event_num'] - 1]['name'] == 'cussion' and self.event[self.event['event_num'] - 1][
                'cussion'] == 'left':
                return
            self.cussion_count += 1
            self.check_spin = 'left'
            self.event[self.event['event_num']] = {'name': 'cussion', 'discribe': '왼쪽 쿠션 침', 'cussion': 'left',
                                                   'pos': self.ball_path_list[self.first_ball['color']][-1],
                                                   'time': self.time}
            if args.print_log:
                print("쿠션: ", self.event[self.event['event_num']]['discribe'])
            self.event['event_num'] += 1
            is_cussion = True
        if np.abs(WIDTH - self.ball_path_list[self.first_ball['color']][-2][0]) < RIGHT_CRASH_THRES \
                and ((self.ball_path_list[self.first_ball['color']][-1][0] -
                      self.ball_path_list[self.first_ball['color']][-2][0]) \
                     * (self.ball_path_list[self.first_ball['color']][-2][0] -
                        self.ball_path_list[self.first_ball['color']][-3][0]) <= 0 and
                     self.ball_path_list[self.first_ball['color']][-2][0] !=
                     self.ball_path_list[self.first_ball['color']][-3][0]):
            if self.event[self.event['event_num'] - 1]['name'] == 'cussion' and self.event[self.event['event_num'] - 1][
                'cussion'] == 'right':
                return
            self.cussion_count += 1
            self.check_spin = 'right'
            self.event[self.event['event_num']] = {'name': 'cussion', 'discribe': '오른 쿠션 침', 'cussion': 'right',
                                                   'pos': self.ball_path_list[self.first_ball['color']][-1],
                                                   'time': self.time}
            if args.print_log:
                print("쿠션: ", self.event[self.event['event_num']]['discribe'])
            self.event['event_num'] += 1
            is_cussion = True

    def check_kiss(self):
        if self.kiss == 'object1' or self.kiss=='object2':
            return
        if self.first_ball['color'] == 'yellow':
            if np.sqrt(np.sum(np.square(np.array(self.ball_path_list['white'][-1]) - np.array(self.ball_path_list['red'][-1])))) < BALL_CRASH_THRES and (self.valocity['red']>ACT_VALOCITY_THRES and self.valocity['white']>ACT_VALOCITY_THRES):
                self.kiss = 'object2'
                self.event[self.event['event_num']] = {'name': 'kiss', 'discribe': 'Kiss-1목적구 2목적구 충돌', 'pos': self.ball_path_list[self.first_ball['color']][-1], 'time': self.time}
                if args.print_log:
                    print(self.event[self.event['event_num']]['discribe'])
                self.event['event_num'] += 1
                self.is_event = True
            if self.second_ball != None:
                if np.sqrt(np.sum(np.square(np.array(self.ball_path_list['yellow'][-1]) - np.array(self.ball_path_list[self.second_ball['color']][-1])))) < BALL_CRASH_THRES and (self.valocity[self.second_ball['color']] > ACT_VALOCITY_THRES) and self.event[self.event['event_num']-1]['name']!='object1':
                    self.kiss = 'object1'
                    self.event[self.event['event_num']] = {'name': 'kiss', 'discribe': 'Kiss-1목적구 재충돌', 'pos': self.ball_path_list[self.first_ball['color']][-1], 'time': self.time}
                    if args.print_log:
                        print(self.event[self.event['event_num']]['discribe'])
                    self.event['event_num'] += 1
                    self.is_event = True
        if self.first_ball['color'] == 'white':
            if np.sqrt(np.sum(np.square(np.array(self.ball_path_list['yellow'][-1]) - np.array(self.ball_path_list['red'][-1])))) < BALL_CRASH_THRES and (self.valocity['red']>ACT_VALOCITY_THRES and self.valocity['yellow']>ACT_VALOCITY_THRES):
                self.kiss = 'object2'
                self.event[self.event['event_num']] = {'name': 'kiss', 'discribe': 'Kiss-1목적구 2목적구 충돌', 'pos': self.ball_path_list[self.first_ball['color']][-1], 'time': self.time}
                if args.print_log:
                    print(self.event[self.event['event_num']]['discribe'])
                self.event['event_num'] += 1
                self.is_event = True
            if self.second_ball != None:
                if np.sqrt(np.sum(np.square(np.array(self.ball_path_list['white'][-1]) - np.array(self.ball_path_list[self.second_ball['color']][-1])))) < BALL_CRASH_THRES and (self.valocity[self.second_ball['color']] > ACT_VALOCITY_THRES) and self.event[self.event['event_num']-1]['name']!='object1':
                    self.kiss = 'object1'
                    self.event[self.event['event_num']] = {'name': 'kiss', 'discribe': 'Kiss-1목적구 재충돌', 'pos': self.ball_path_list[self.first_ball['color']][-1], 'time': self.time}
                    if args.print_log:
                        print(self.event[self.event['event_num']]['discribe'])
                    self.event['event_num'] += 1
                    self.is_event = True

    def check_spin_direction(self):
        if self.check_spin=='top':
            if self.ball_path_list[self.first_ball['color']][-1][0] > self.ball_path_list[self.first_ball['color']][-3][0]:
                self.event[self.event['event_num']-1]['spin'] = 'right'
                #print(self.event[self.event['event_num']-1]['spin'])
                self.is_event = True
            else:
                self.event[self.event['event_num']-1]['spin'] = 'left'
                #print(self.event[self.event['event_num']-1]['spin'])
                self.is_event = True
        elif self.check_spin=='buttom':
            if self.ball_path_list[self.first_ball['color']][-1][0] < self.ball_path_list[self.first_ball['color']][-3][0]:
                self.event[self.event['event_num']-1]['spin'] = 'right'
                #print(self.event[self.event['event_num']-1]['spin'])
                self.is_event = True
            else:
                self.event[self.event['event_num']-1]['spin'] = 'left'
                #print(self.event[self.event['event_num']-1]['spin'])
                self.is_event = True
        elif self.check_spin=='left':
            if self.ball_path_list[self.first_ball['color']][-1][1] < self.ball_path_list[self.first_ball['color']][-3][1]:
                self.event[self.event['event_num']-1]['spin'] = 'right'
                #print(self.event[self.event['event_num']-1]['spin'])
                self.is_event = True
            else:
                self.event[self.event['event_num']-1]['spin'] = 'left'
                #print(self.event[self.event['event_num']-1]['spin'])
                self.is_event = True
        elif self.check_spin=='right':
            if self.ball_path_list[self.first_ball['color']][-1][1] > self.ball_path_list[self.first_ball['color']][-3][1]:
                self.event[self.event['event_num']-1]['spin'] = 'right'
                #print(self.event[self.event['event_num']-1]['spin'])
                self.is_event = True
            else:
                self.event[self.event['event_num']-1]['spin'] = 'left'
                #print(self.event[self.event['event_num']-1]['spin'])
                self.is_event = True
        self.check_spin = False

    def check_result(self):
        if self.cussion_count>=3 and self.first_ball != None and self.second_ball != None and self.third_ball != None:
            self.score = True
        else:
            if self.first_ball == None:
                self.fail_description="수구 안침"
            elif self.second_ball == None:
                self.fail_description ="1 목적구 안침"
            elif self.third_ball == None:
                self.fail_description="2 목적구 안침"
            elif self.cussion_count <3:
                self.fail_description="쓰리쿠 안침"

    def get_valocity(self):
        width = (self.ball_path_list[self.first_ball['color']][-1][0]-self.ball_path_list[self.first_ball['color']][-2][0])/1000 * 2844
        height = (self.ball_path_list[self.first_ball['color']][-1][1]-self.ball_path_list[self.first_ball['color']][-2][1])/500 * 1422
        distance = np.sqrt(width**2 + height**2)

        if args.data_type == 'video':
            return distance/1000/3
        else:
            return distance/1000

    def check_event(self, time, img_color):
        self.time=time
        self.calc_distance()
        self.calc_valocity()

        if self.init_ball_pos == None:
            self.init_ball_pos = inital_ball(img_color)

        # 쪽났는지 확인하자
        if self.first_ball != None:
            self.check_kiss()

        #수구가 무엇인지 알아보자
        if self.first_ball == None:
            self.check_first_ball(self.ball_path_list)
            if self.first_ball != None:
                # self.captured_img = self.capture_shoot(cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY), self.prev_img)
                self.captured_img = self.capture_shoot(cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY), img_color)
                return 'shoot'
        #수구를 쳤다면 1목적구가 무엇인지 알아보자
        if self.first_ball != None and self.second_ball == None:
            self.check_second_ball(BALL_CRASH_THRES, ACT_VALOCITY_THRES)
            if self.second_ball != None:
                self.check_ball_left_right()
                self.find_cuepoint()
                return 'object1'
        if self.first_ball != None and self.second_ball != None and self.third_ball == None:
            self.check_third_ball(BALL_CRASH_THRES, ACT_VALOCITY_THRES)
        elif self.third_ball != None:
            self.check_result()
            return 'object2'
        #쿠션을 쳤는지 확인하자
        if self.first_ball != None:
            self.check_cussion()
        #스핀을 확인하자
        if self.check_spin != False:
            self.check_spin_direction()
        return None

    def write_event(self):
        self.pd_data['event'] = 'NA'
        self.pd_data['direction'] = 'NA'
        for event in self.event:
            try:
                int(event)
                if self.event[event]['name'] == 'shoot':
                    self.pd_data.loc[self.event[event]['time'], 'event'] = self.event[event]['name']
                    direction = self.event[event]['shootpoint']
                elif self.event[event]['name'] == 'object1' or self.event[event]['name'] == 'object2':
                    self.pd_data.loc[self.event[event]['time'], 'event'] = self.event[event]['name']
                    direction = self.event[event]['hit_point']
                elif self.event[event]['name'] == 'cussion':
                    self.pd_data.loc[self.event[event]['time'], 'event'] = self.event[event]['cussion']
                    direction = self.event[event]['spin']
                self.pd_data.loc[self.event[event]['time'], 'direction'] = direction
            except:
                continue

    def find_cuepoint(self):
        y_idx, x_idx = np.where(self.captured_img != 0)
        min_dis = 99999
        cue_point = None
        ball_point = None
        ball_direction = None
        enum={'red':0, 'yellow':1,'white':2}
        ball_point = self.init_ball_pos[enum[self.first_ball['color']]]

        for event in self.event:
            try:
                if self.event[event]['name'] == 'shoot':
                    ball_direction = self.event[event + 1]['pos']
                    break
            except:
                continue
        for x, y in zip(x_idx, y_idx):
            dis = np.sqrt(np.abs(x - ball_point['x']) ** 2 + np.abs(y - ball_point['y']) ** 2)
            if dis < min_dis:
                min_dis = dis
                cue_point = [x, y]

        if cue_point != None and self.init_ball_pos!=None:
            v1 = np.array([ball_point['x'], ball_point['y']]) - np.array(cue_point)
            v2 = np.array(ball_direction) - np.array([ball_point['x'], ball_point['y']])
            theta = math.degrees(np.arcsin(
                (v1[0] * v2[1] - v1[1] * v2[0]) / (np.sqrt(v1[0] ** 2 + v1[1] ** 2) * np.sqrt(v2[0] ** 2 + v2[1] ** 2))))

            if theta < -10:
                shoot_point = 'strong_left'
            elif theta < -3:
                shoot_point = 'left'
            elif theta > 10:
                shoot_point = 'strong_right'
            elif theta > 3:
                shoot_point = 'right'
            else:
                shoot_point = 'center'
            for event in self.event:
                try:
                    if self.event[event]['name'] == 'shoot':
                        self.event[event]['shootpoint'] = shoot_point
                        break
                except:
                    continue
            self.shootpoint = shoot_point

    def path_analysis(self):
        cussion_list = []
        cussion_num = 0

        for event in self.event:
            try:
                int(event)
                if self.event[event]['name'] == 'cussion':
                    cussion_list.append(self.event[event]['cussion'])
                    cussion_num += 1
                if self.event[event]['name'] == 'object2':
                    break
                if cussion_num >= 3:
                    break
            except:
                continue

        #쿠션 수 한개 미달 --> 2 목적구 타격 시 연장선을 그어 3 번째 쿠션으로 사용
        error = None
        if cussion_num == 2:
            cussion_list = []
            error = self.expand_path()
            for event in self.event:
                try:
                    int(event)
                    if self.event[event]['name'] == 'cussion':
                        cussion_list.append(self.event[event]['cussion'])
                    if self.event[event]['name'] == 'object2':
                        break
                    if cussion_num >= 3:
                        break
                except:
                    continue

        # 실패시
        # 2목적구 미스 샷 발생 --> 3쿠션 이후 2목적구에 최근접 할 때까지의 경로를 구종 분류에 사용
        if self.first_ball != None and self.second_ball != None and self.third_ball == None:
            # 쿠션 3개부터 최저 거리 지점 확인
            three_cussion_check=0
            three_cussion_time=0
            three_cussion_event=0
            for event in self.event:
                try:
                    int(event)
                    if self.event[event]['name'] == 'cussion':
                        three_cussion_check += 1
                    if three_cussion_check == 3:
                        three_cussion_time=int(self.event[event]['time'])
                        three_cussion_event = int(event)
                        break
                except:
                    continue
            cur_event = three_cussion_event
            self.final_event = cur_event
            ghost_object2 = {'name':'object2', 'discribe':'error', 'pos':[0,0]}
            for t in range(three_cussion_time, len(self.ball_path_list[self.first_ball['color']])):
                if cur_event+1<self.event['event_num']:
                    #print("시간", t, self.event[cur_event+1]['time'])
                    if self.event[cur_event+1]['time']<t:
                        cur_event+=1
                if self.first_ball['color'] == 'white':
                    if self.second_ball['color'] == 'red':
                        cur_dist = np.sqrt(np.sum(np.square(np.array(self.ball_path_list['yellow'][t]) - np.array(self.ball_path_list['white'][t]))))
                        ghost_object2={'name':'object2', 'discribe':'제2목적구 노란공 (가짜)', 'pos':self.ball_path_list['yellow'][t]}
                    elif self.second_ball['color'] == 'yellow':
                        cur_dist = np.sqrt(np.sum(np.square(np.array(self.ball_path_list['red'][t]) - np.array(self.ball_path_list['white'][t]))))
                        ghost_object2 = {'name': 'object2', 'discribe': '제2목적구 빨간공 (가짜)',
                                         'pos': self.ball_path_list['red'][t]}
                if self.first_ball['color'] == 'yellow':
                    if self.second_ball['color'] == 'red':
                        cur_dist = np.sqrt(np.sum(np.square(np.array(self.ball_path_list['yellow'][t]) - np.array(self.ball_path_list['white'][t]))))
                        ghost_object2 = {'name': 'object2', 'discribe': '제2목적구 흰공 (가짜)',
                                         'pos': self.ball_path_list['white'][t]}
                    elif self.second_ball['color'] == 'white':
                        cur_dist = np.sqrt(np.sum(np.square(np.array(self.ball_path_list['red'][t]) - np.array(self.ball_path_list['yellow'][t]))))
                        ghost_object2 = {'name': 'object2', 'discribe': '제2목적구 빨간공 (가짜)',
                                         'pos': self.ball_path_list['red'][t]}
                if self.first_ball['color'] == 'red':
                    if self.second_ball['color'] == 'yellow':
                        cur_dist = np.sqrt(np.sum(np.square(np.array(self.ball_path_list['red'][t]) - np.array(self.ball_path_list['white'][t]))))
                        ghost_object2 = {'name': 'object2', 'discribe': '제2목적구 흰공 (가짜)',
                                         'pos': self.ball_path_list['white'][t]}
                    elif self.second_ball['color'] == 'white':
                        cur_dist = np.sqrt(np.sum(np.square(np.array(self.ball_path_list['red'][t]) - np.array(self.ball_path_list['yellow'][t]))))
                        ghost_object2 = {'name': 'object2', 'discribe': '제2목적구 빨간공 (가짜)',
                                         'pos': self.ball_path_list['red'][t]}
                if self.closest == -1 or cur_dist < self.closest:
                    self.closest = cur_dist
                    self.final_event = cur_event

            tmp = {'event_num': self.final_event+1}
            for event in self.event:
                try:
                    int(event)
                    if int(event) <= self.final_event:
                        tmp[event] = self.event[event]
                except:
                    continue
            tmp[self.final_event+1]=ghost_object2
            tmp[self.final_event+1]['time']=tmp[self.final_event]['time']+1
            self.event = tmp

        cussion_list = []
        cussion_num = 0

        for event in self.event:
            try:
                int(event)
                if self.event[event]['name'] == 'cussion':
                    cussion_list.append(self.event[event]['cussion'])
                    cussion_num += 1
                if self.event[event]['name'] == 'object2':
                    break
                if cussion_num >= 3:
                    break
            except:
                continue

        #최초충돌 공 vs 쿠션
        first_path = None
        for event in self.event:
            try:
                if self.event[event]['name']=='object1':
                    first_path = 'object1'
                    break
                elif self.event[event]['name']=='cussion':
                    first_path = 'cussion'
                    break
            except:
                continue

        #공
        if first_path == 'object1':
            # 선회진로?
            clock_wise = [["top", "right", "buttom"], ["right", "buttom", "left"], ["buttom",
                          "left", "top"], ["left", "top", "right"], ["left", "top", "buttom"], ["right", "buttom", "top"], ["top", "right", "left"], ["buttom", "left", "right"]]
            counter_clock_wise = [["top", "left", "buttom"], ["right", "top", "left"], ["buttom", "right", "top"], ["left",
                                  "buttom", "right"], ["left", "buttom", "top"], ["right", "top", "buttom"], ["top", "left", "right"], ["buttom", "right", "left"]]

            if cussion_list in clock_wise:
                way = "clock_wise"
            elif cussion_list in counter_clock_wise:
                way = "counter_clock_wise"
            else:
                way = None
                # 1,2쿠션 횡단?
                cussion_num = 0
                cussion1 = None
                cussion2 = None
                for event in self.event:
                    try:
                        int(event)
                        if self.event[event]['name']=='cussion':
                            if cussion_num == 0:
                                cussion1 = self.event[event]['cussion']
                                cussion_num += 1
                            elif cussion_num == 1:
                                cussion2 = self.event[event]['cussion']
                                cussion_num += 1
                            else:
                                break
                    except:
                        continue
                if (cussion1 == 'top' and cussion2 == 'buttom') or (cussion1 == 'buttom' and cussion2 == 'top'):
                    cross = 'long'
                elif (cussion1 == 'left' and cussion2 == 'right') or (cussion1 == 'right' and cussion2 == 'left'):
                    cross = 'short'
                else:
                    cross = None
                    self.final_path=path_dict['PLATESHOT']
                #완전한 횡단?
                cross_cushion_count=0
                if cross == 'long' or cross == 'short':
                    is_pure = True
                    for event in self.event:
                        try:
                            int(event)
                            if self.event[event]['name'] == 'cussion':
                                cross_cushion_count+=1
                                if cross == 'long' and (self.event[event]['cussion']=='left' or self.event[event]['cussion']=='right'):
                                    is_pure = False
                                    break
                                elif cross == 'short' and (self.event[event]['cussion']=='top' or self.event[event]['cussion']=='buttom'):
                                    is_pure = False
                                    break
                                if cross_cushion_count>=3:
                                    break
                        except:
                            continue
                    if is_pure:
                        self.final_path=path_dict['CROSSTABLE']
                    else:
                        shootpoint=None
                        first_cussion_spin=None
                        for event in self.event:
                            try:
                                int(event)
                                if self.event[event]['name'] == 'shoot':
                                    shootpoint = self.event[event]['shootpoint']
                                    break
                            except:
                                continue
                        for event in self.event:
                            try:
                                int(event)
                                if self.event[event]['name'] == 'cussion':
                                    first_cussion_spin = self.event[event]['spin']
                                    break
                            except:
                                continue
                        if shootpoint == first_cussion_spin:
                            self.final_path=path_dict['DOUBLECUSSIONSHOT']
                        else:
                            self.final_path=path_dict['REVERSESHOT']

            #선회 방향?
            if way == "clock_wise" or way == "counter_clock_wise":
                hit_point = None
                for event in self.event:
                    try:
                        int(event)
                        if self.event[event]['name']=='object1':
                            hit_point = self.event[event]['hit_point']
                            break
                    except:
                        continue
                #정방향
                if (way == "clock_wise" and hit_point == "right") or (way == "counter_clock_wise" and hit_point == "left"):
                    rail = None
                    for event in self.event:
                        try:
                            int(event)
                            if self.event[event]['name'] == 'cussion':
                                rail = self.event[event]['cussion']
                                break
                        except:
                            continue
                    if rail == 'left' or rail == 'right':
                        three_cussion_point=None
                        cussion_num=0
                        for event in self.event:
                            try:
                                int(event)
                                if self.event[event]['name'] == 'cussion':
                                    cussion_num += 1
                                    if cussion_num == 3:
                                        three_cussion = self.event[event]['cussion']
                                        three_cussion_point = self.event[event]['pos']
                                        break
                            except:
                                continue
                        if three_cussion == 'top' or three_cussion == 'buttom':
                            self.final_path = path_dict['LONGANGLE']
                        under_point = HEIGHT*0.625
                        up_point = HEIGHT*0.375
                        if way == 'clock_wise' and three_cussion == 'left':
                            if three_cussion_point[1]>=under_point:
                                self.final_path=path_dict['SHORTANGLE']
                            else:
                                self.final_path = path_dict['LONGANGLE']
                        elif way == 'clock_wise' and three_cussion == 'right':
                            if three_cussion_point[1] <= up_point:
                                self.final_path = path_dict['SHORTANGLE']
                            else:
                                self.final_path = path_dict['LONGANGLE']
                        elif way == 'counter_clock_wise' and three_cussion == 'left':
                            if three_cussion_point[1] <= up_point:
                                self.final_path = path_dict['SHORTANGLE']
                            else:
                                self.final_path = path_dict['LONGANGLE']
                        elif way == 'counter_clock_wise' and three_cussion == 'right':
                            if three_cussion_point[1] >= under_point:
                                self.final_path = path_dict['SHORTANGLE']
                            else:
                                self.final_path = path_dict['LONGANGLE']
                        else:
                            print("구현 안됨1")
                    else:
                        self.final_path=path_dict['SIDEANGLE']
                else:
                    first_ball_point=None
                    second_ball_point=None
                    first_cussion_point=None
                    first_cussion=None
                    for event in self.event:
                        try:
                            int(event)
                            if self.event[event]['name'] == 'shoot':
                                first_ball_point = self.event[event]['pos']
                                break
                        except:
                            continue
                    for event in self.event:
                        try:
                            int(event)
                            if self.event[event]['name'] == 'object1':
                                second_ball_point = [0,0]
                                enum = {'red': 0, 'yellow': 1, 'white': 2}
                                second_ball_point[0] = self.init_ball_pos[enum[self.second_ball['color']]]['x']
                                second_ball_point[1] = self.init_ball_pos[enum[self.second_ball['color']]]['y']
                                break
                        except:
                            continue
                    for event in self.event:
                        try:
                            int(event)
                            if self.event[event]['name'] == 'cussion':
                                first_cussion_point = self.event[event]['pos']
                                first_cussion = self.event[event]['cussion']
                                break
                        except:
                            continue
                    if first_cussion == 'top' or first_cussion =='buttom':
                        if self.init_ball_pos[enum[self.first_ball['color']]]['x'] < self.init_ball_pos[enum[self.second_ball['color']]]['x']:

                            if first_cussion_point[0] < second_ball_point[0]:
                                self.final_path=path_dict['BIASANGLE']
                            else:
                                self.final_path=path_dict['OUTSIDEANGLE']
                        else:
                            if first_cussion_point[0] > second_ball_point[0]:
                                self.final_path=path_dict['BIASANGLE']
                            else:
                                self.final_path=path_dict['OUTSIDEANGLE']
                    elif first_cussion == 'left' or 'right':
                        if self.init_ball_pos[enum[self.first_ball['color']]]['x'] < second_ball_point[y]:
                            if first_cussion_point[1] < second_ball_point[1]:
                                self.final_path=path_dict['BIASANGLE']
                            else:
                                self.final_path=path_dict['OUTSIDEANGLE']
                        else:
                            if first_cussion_point[1] > second_ball_point[1]:
                                self.final_path=path_dict['BIASANGLE']
                            else:
                                self.final_path=path_dict['OUTSIDEANGLE']
                    else:
                        print("구현 안됨2")
            if (self.final_path == path_dict['SHORTANGLE'] or self.final_path == path_dict[
                'LONGANGLE'] or self.final_path == path_dict['BIASANGLE'] or self.final_path == path_dict[
                                           'OUTSIDEANGLE'] or self.final_path == path_dict['SIDEANGLE']):
                cussion_num = 0
                shoot_point = None
                object_point= None
                cussion = None
                for event in self.event:
                    try:
                        int(event)
                        if self.event[event]['name'] == 'shoot':
                            shoot_point = self.event[event]['pos']
                        if self.event[event]['name'] == 'cussion':
                            if cussion_num==0:
                                cussion=self.event[event]['cussion']
                            cussion_num += 1
                        if self.event[event]['name'] == 'object2':
                            object_point = self.event[event]['pos']
                            break
                    except:
                        continue

                if cussion_num >= 5:
                    self.is_round_table = True
                elif cussion_num >= 4:
                    if way == 'clock_wise' and cussion=='right' and shoot_point[0]<object_point[0] and shoot_point[1] < object_point[1]:
                        self.is_round_table = True
                    if way == 'clock_wise' and cussion=='top' and shoot_point[0]<object_point[0] and shoot_point[1] > object_point[1]:
                        self.is_round_table = True
                    if way == 'clock_wise' and cussion=='left' and shoot_point[0]>object_point[0] and shoot_point[1] > object_point[1]:
                        self.is_round_table = True
                    if way == 'clock_wise' and cussion=='buttom' and shoot_point[0]>object_point[0] and shoot_point[1] < object_point[1]:
                        self.is_round_table = True
                    if way == 'counter_clock_wise' and cussion=='right' and shoot_point[0]<object_point[0] and shoot_point[1] > object_point[1]:
                        self.is_round_table = True
                    if way == 'counter_clock_wise' and cussion=='top' and shoot_point[0]>object_point[0] and shoot_point[1] > object_point[1]:
                        self.is_round_table = True
                    if way == 'counter_clock_wise' and cussion=='left' and shoot_point[0]>object_point[0] and shoot_point[1] < object_point[1]:
                        self.is_round_table = True
                    if way == 'counter_clock_wise' and cussion=='buttom' and shoot_point[0]<object_point[0] and shoot_point[1] < object_point[1]:
                        self.is_round_table = True
        #쿠션
        else:
            #right_path
            bank_count = 0
            for event in self.event:
                try:
                    int(event)
                    if self.event[event]['name'] == 'cussion':
                        bank_count += 1
                    elif self.event[event]['name'] == 'object1':
                        break
                except:
                    continue
            if bank_count == 1:
                self.final_path=path_dict['ONEBANKSHOT']
            elif bank_count == 2:
                self.final_path=path_dict['TWOBANKSHOT']
            else:
                if self.second_ball==None:
                    self.final_path = path_dict['OTHERSHOT']
                else:
                    self.final_path=path_dict['THREEBANKSHOT']
    def remove_lf(self):
        self.event = {"event_num": 0}
        self.first_ball = None
        self.second_ball = None
        self.third_ball = None
        self.time = 0
        self.cussion_count = 0
        self.check_spin = False
        self.kiss = False
        self.shootpoint = 'center'
        self.dist_white_yellow = 99999
        self.dist_white_red = 99999
        self.dist_yellow_red = 99999
        self.valocity = {'red': -1, 'yellow': -1, 'white': -1}
        self.is_event = False
        self.final_path = None
        self.is_round_table = False
        # 실패시
        self.closest = -1
        #low_path_filter
        windowSize=10
        window=np.array([1,2,1])
        window=window/window.sum()

        new_red = np.concatenate((np.expand_dims(np.convolve(window, np.array(self.ball_path_list['red'])[:,0], mode='valid'), axis=(1)), np.expand_dims(np.convolve(window, np.array(self.ball_path_list['red'])[:,1], mode='valid'), axis=(1))), axis=1)
        new_yellow = np.concatenate((np.expand_dims(np.convolve(window, np.array(self.ball_path_list['yellow'])[:,0], mode='valid'), axis=(1)), np.expand_dims(np.convolve(window, np.array(self.ball_path_list['yellow'])[:,1], mode='valid'), axis=(1))), axis=1)
        new_white = np.concatenate((np.expand_dims(np.convolve(window, np.array(self.ball_path_list['white'])[:,0], mode='valid'), axis=(1)), np.expand_dims(np.convolve(window, np.array(self.ball_path_list['white'])[:,1], mode='valid'), axis=(1))), axis=1)

        if args.print_log:
            print("[하이 프리퀀시 지우기 위한 시뮬레이션 시작]")

        for i in range(START_TRACKING_THRES, len(new_red)):
            self.ball_path_list['red'] = new_red[:i]
            self.ball_path_list['yellow'] = new_yellow[:i]
            self.ball_path_list['white'] = new_white[:i]
            self.simulation_based_event(i)
        self.ball_path_list['red']=new_red
        self.ball_path_list['yellow'] = new_yellow
        self.ball_path_list['white'] = new_white

    def expand_path(self):
        start = None
        end = None
        object2_idx = 0
        for event in self.event:
            try:
                int(event)
                if self.event[event]['name'] == 'object2':
                    object2_idx = event
                    start = self.event[event-1]['pos']
                    end = self.event[event]['pos']
                    break
            except:
                continue
        if start == None or end==None:
            return 'error'

        move_vec = np.array(end)-np.array(start)
        move_vec = move_vec/(abs(min(move_vec))+1e-6)

        if start[0]<end[0]:
            left = end[0]
            right = 1000
        else:
            left = 0
            right = end[0]

        x = np.array(range(left, right + 1))
        y = np.array(x * (move_vec[1] / move_vec[0]) + end[1] - (
                        move_vec[1] / move_vec[0]) * end[0],
                         dtype=np.int16)

        out_idx_left = np.where(x < 0)
        out_idx_right = np.where(x > 1000)
        out_idx_top = np.where(y < 0)
        out_idx_buttom = np.where(y > 500)
        out_idx = np.concatenate((out_idx_left[0], out_idx_right[0], out_idx_top[0], out_idx_buttom[0]), 0)
        x = np.delete(x, out_idx)
        y = np.delete(y, out_idx)

        if move_vec[0] > 0:
            new_x = max(x)
        else:
            new_x = min(x)

        new_y = int(new_x * (move_vec[1] / (move_vec[0]+1e-6)) + end[1] - (
                        move_vec[1] / (move_vec[0]+1e-6)) * end[0])
        end = np.array([new_x, new_y])

        if end[0] == 0:
            cussion = 'left'
            if start[1]<end[1]:
                spin='left'
            else:
                spin='right'
        elif end[0] == 1000:
            cussion = 'right'
            if start[1]<end[1]:
                spin='right'
            else:
                spin='left'
        elif end[1] == 0:
            cussion = 'top'
            if start[0]<end[1]:
                spin='right'
            else:
                spin='left'
        else:
            cussion = 'buttom'
            if start[0]<end[0]:
                spin = 'left'
            else:
                spin = 'right'

        #cv2.line(show_img, (start[0], start[1]), (end[0], end[1]), (255,0,255), 5)
        #cv2.imshow("line_expand", show_img)
        #cv2.waitKey(0)

        self.event[object2_idx+1]=self.event[object2_idx]
        self.event[object2_idx]={'name':'cussion', 'cussion': cussion, 'pos':[new_x, new_y], 'spin':spin, 'time':self.event[object2_idx+1]['time']}
        self.event['event_num']+=1

    def write_json(self, send_data, name):

        if self.second_ball == None:
            object1 = 'red'
        else:
            object1 = self.second_ball['color']
        if self.third_ball == None:
            if 'red' not in [self.first_ball['color'], object1]:
                object2 = 'red'
            elif 'white' not in [self.first_ball['color'], object1]:
                object2 = 'white'
            else:
                object2 = 'yellow'
        else:
            object2 = self.third_ball['color']
        data = dict()
        if self.direction == None:
            self.direction = send_data['direction']

        displayobject2 = 0
        cussion_count = 0
        for event in self.event:
            try:
                int(event)
                if self.event[event]['name'] == 'cussion':
                    cussion_count = cussion_count + 1
                if self.event[event]['name'] == 'object2':
                    break
            except:
                continue
        if cussion_count < 2:
            displayobject2 = 1
        elif self.kiss == 'object1':
            displayobject2 = 2

        data["init"] = {"cuePosInit": (np.array(self.ball_path_list[self.first_ball['color']])[0] / 1000 * 2844).tolist(),
                        "object1PosInit": (np.array(self.ball_path_list[object1])[0] / 1000 * 2844).tolist(),
                        "object2PosInit": (np.array(self.ball_path_list[object2])[0] / 1000 * 2844).tolist(),
                        "startingBall": send_data["start"], "object1": object1,
                        "shootPoint": send_data['shoot'], "shootDirection": self.direction,
                        "shootValocity": send_data['shoot_velocity'], "displayobject2": displayobject2}
        data["startingBallPos"] = []
        action = 0
        for event in self.event:
            try:
                int(event)
                if self.event[event]['name'] == 'cussion':
                    data["startingBallPos"].append({"action": self.event[event]["cussion"],
                                                    "pos": (np.array(self.event[event]["pos"]) / 1000 * 2844).tolist()})
                else:
                    data["startingBallPos"].append({"action": self.event[event]["name"],
                                                "pos": (np.array(self.event[event]["pos"]) / 1000 * 2844).tolist()})
                action = action + 1
            except:
                continue

        data["allBallPath"] = {}
        data["allBallPath"]["cueBallPath"] = []
        data["allBallPath"]["object1Path"] = []
        data["allBallPath"]["object2Path"] = []
        for pos in (np.array(self.ball_path_list[send_data["start"]])[:] / 1000 * 2844).tolist():
            data["allBallPath"]["cueBallPath"].append({"pos": pos})
        for pos in (np.array(self.ball_path_list[object1])[:] / 1000 * 2844).tolist():
            data["allBallPath"]["object1Path"].append({"pos": pos})
        for pos in (np.array(self.ball_path_list[object2])[:] / 1000 * 2844).tolist():
            data["allBallPath"]["object2Path"].append({"pos": pos})

        with open('./../communication/'+str(name)+'.json', 'w', encoding='utf-8') as make_file:
            json.dump(data, make_file, indent="\t")


def find_table(img):
    #mean=img[]
    img = cv2.medianBlur(img, 5, 0)
    #img = cv2.GaussianBlur(img, (3, 3), 0)
    box_binary = 255-cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)
    #box_binary = cv2.dilate(gray, None, iterations=2)
    box_binary = cv2.dilate(box_binary, None, iterations=3)

    #_, box_binary = cv2.threshold(img, 240, 255, 0)
    mask = np.zeros_like(box_binary)
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

    table = {'top': max(0, top-10),
             'buttom': min(buttom+10, img.shape[0]),
             'left': max(0, left-10),
             'right': min(right+10, img.shape[1])}

    return table

def inital_ball(img_color):
    circles=None
    mean = np.median(img_color, axis=(0, 1))

    # print(img_color[np.sum(np.abs(img_color-mean), axis=2)<50].shape)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    masked_img = img_color.copy()
    masked_img[np.sum(np.abs(img_color - mean), axis=2) < 120]=0
    mask = np.array(img_color[:,:,0]>img_color[:,:,1], dtype=np.uint8)*255
    blurred = cv2.medianBlur(masked_img, 11, 0)
    blurred = cv2.blur(blurred, (5,5), 1)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 10, 255, 0)
    if len(binary[binary == 255]) > len(binary[binary == 0]):
        binary = 255 - binary
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        segment_area = cv2.contourArea(cnt)
        if segment_area <= 100 or segment_area >= 700:
            mask = cv2.drawContours(binary, [cnt], -1, 0, -1)
    binary = cv2.blur(mask, (5,5), 1)
    circles_low = cv2.HoughCircles(binary, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=15, minRadius=8, maxRadius=15)
    circles_high = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=15, minRadius=8, maxRadius=15)

    if circles_low is not None and circles_low.shape[1]==3:
        circles = circles_low
    elif circles_high is not None:
        circles = circles_high

    ball_list = {}

    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(13, 13))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    high_const = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    if circles is not None:
        for i in range(circles.shape[1]):
            x,y, radius = circles[0][i]
            mask=np.zeros_like(gray)
            cv2.circle(mask, (int(x), int(y)), int(radius), 255, -1)
            median_color = np.median(high_const[mask==255], axis=(0))
            if median_color[0] == 0 and median_color[1]==0 and median_color[2]==0:
                continue
            #median_color = cv2.cvtColor(median_color, cv2.COLOR_BGR2HLS)
            ball_info = {'color': median_color,
                         'radius': int(radius),
                         'x':int(x),
                         'y':int(y),
                         #'weight': weight[i]}
                         }
            ball_list[i]=ball_info

        copyed_ball_list = ball_list.copy()
        if len(ball_list)==3:
            # red부터
            red_value=-1
            red_idx=0
            for ball in copyed_ball_list:
                #print(copyed_ball_list)
                cur_value = (copyed_ball_list[ball]['color'][2]-copyed_ball_list[ball]['color'][1])/(copyed_ball_list[ball]['color'][1] + copyed_ball_list[ball]['color'][2])
                if red_value == -1 or cur_value > red_value:
                    red_value=cur_value
                    red_idx=ball
            del(copyed_ball_list[red_idx])

            yellow_value = -1
            yellow_idx = 0
            for ball in copyed_ball_list:
                cur_value = (copyed_ball_list[ball]['color'][2] + copyed_ball_list[ball]['color'][1] -
                             copyed_ball_list[ball]['color'][0]) / (
                                    copyed_ball_list[ball]['color'][0] + copyed_ball_list[ball]['color'][1] +
                                    copyed_ball_list[ball]['color'][2])
                if yellow_value == -1 or cur_value > yellow_value:
                    yellow_value = cur_value
                    yellow_idx = ball
            del (copyed_ball_list[yellow_idx])

            # white
            white_value = -1
            white_idx = 0
            for ball in copyed_ball_list:
                cur_value = (abs(copyed_ball_list[ball]['color'][2] - copyed_ball_list[ball]['color'][1]) + abs(
                    copyed_ball_list[ball]['color'][1] - copyed_ball_list[ball]['color'][0]) + abs(
                    copyed_ball_list[ball]['color'][0] - copyed_ball_list[ball]['color'][2])) / (
                                    copyed_ball_list[ball]['color'][0] + copyed_ball_list[ball]['color'][1] +
                                    copyed_ball_list[ball]['color'][2]) + (
                                        copyed_ball_list[ball]['color'][2] + copyed_ball_list[ball]['color'][1] +
                                        copyed_ball_list[ball]['color'][0]) / 3
                if white_value == -1 or cur_value < white_value:
                    white_value = cur_value
                    white_idx = ball
            del (copyed_ball_list[white_idx])

            if len(ball_list)==3:
                sorted_ball_list = [ball_list[red_idx], ball_list[yellow_idx], ball_list[white_idx]]
                return sorted_ball_list
            else:
                return None


def find_ball(img, show_img, img_color, event_checker):
    radius = None

    COLORS = [(1, 1, 1), (0, 0, 1), (0, 1, 1)]

    height, width = img_color.shape[:2]
    mean = np.median(img_color, axis=(0, 1))
    img_color[np.sum(np.abs(img_color - mean), axis=2) < 120] = 0
    x = torch.from_numpy(transform(img_color)[0]).permute(2, 0, 1).cuda()
    x = Variable(x.unsqueeze(0))
    y = net(x)  # forward pass
    detections = y.data[0]
    # scale each detection back up to the image
    scale = torch.Tensor([width, height, width, height])

    ball_list = {}


    for i in range(detections.size(0)):
        if detections[i, 0, 0] >= 0.3:

            pt = (detections[i, 0, 1:] * scale).cpu().numpy()
            cv2.rectangle(show_img,
                          (int(pt[0]), int(pt[1])),
                          (int(pt[2]), int(pt[3])),
                          COLORS[i % 3], 2)
            pt = (detections[i, 0, 1:] * scale).cpu().numpy()
            x = (pt[0]+pt[2])/2
            y = (pt[1]+pt[3])/2
            radius_x = abs((int(pt[0])-int(pt[2]))/2)
            radius_y = abs((int(pt[1]) - int(pt[3])) / 2)
            radius = (radius_x+radius_y)/2

            if radius_x/radius_y>1.5 or radius_y/radius_x>1.5:
                continue

            cls=['','red', 'yellow', 'white']
            #median_color = cv2.cvtColor(median_color, cv2.COLOR_BGR2HLS)
            ball_info = {'color': cls[i],
                         'radius': int(radius),
                         'x':int(x),
                         'y':int(y),
                         #'weight': weight[i]}
                         }
            ball_list[cls[i]]=ball_info

    return show_img, ball_list, radius

# Press the green button in the gutter to run the script.
def predict_func(path):
        vid_name = path.split('/')[-1]
        directory = ''
        for i in args.input.split('/')[:-1]:
            directory += i
            directory += '/'

        if args.data_type=='video':
            imgs = Videoloader(path)
            fps = imgs.get_fps()
        else:
            imgs = Image_loader(path)
        import time

        ball_path_list = None
        sample = imgs[0]
        img, _, _ = sample
        table = find_table(img)
        ball_list = None

        send_data = {
            'red': [0,0],
            'yellow': [0,0],
            'white': [0,0],
            'direction': [0,0],
            'shoot': 'center',
            'start': 'white',
            'object1': 'yellow',
            'shoot_velocity':0
        }

        event = None

        event_checker = None

        shoot_count=0
        is_shooted=False

        for t, datas in enumerate(imgs):
            start_time=time.time()

            if datas==None:
                break
            if event_checker!=None:
                event_checker.prev_img = img_color
            img_original, img_color, name = datas
            table = find_table(img_original)
            img = img_original[table['top']:table['buttom'], table['left']:table['right']]
            img = cv2.resize(img, (WIDTH,HEIGHT), cv2.INTER_LINEAR)

            img_color = img_color[table['top']:table['buttom'], table['left']:table['right']]
            img_color = cv2.resize(img_color, (WIDTH, HEIGHT), cv2.INTER_LINEAR)

            # cv2.imshow("asdf", img_color)
            # cv2.waitKey(0)

            show_img = np.zeros((img.shape[0], img.shape[1],3))
            show_img[:, :, 0] = img/255
            show_img[:, :, 1] = img/255
            show_img[:, :, 2] = img/255
            # cv2.imshow("asdf", show_img)
            # cv2.waitKey(0)
            if ball_path_list == None:
                show_img, ball_list, radius = find_ball(img, show_img, img_color, None)
                if len(ball_list) == 3:
                    if args.print_log:
                        print("Find balls")
                        print("red: (",ball_list['red']['x'],', ', ball_list['red']['y'], ')')
                        print("yellow: (", ball_list['yellow']['x'], ', ', ball_list['yellow']['y'], ')')
                        print("white: (", ball_list['white']['x'], ', ', ball_list['white']['y'], ')')
                    send_data['red'][0] = ball_list['red']['x']/1000*2844
                    send_data['red'][1] = ball_list['red']['y']/500*1422
                    send_data['yellow'][0] = ball_list['yellow']['x']/1000*2844
                    send_data['yellow'][1] = ball_list['yellow']['y']/500*1422
                    send_data['white'][0] = ball_list['white']['x']/1000*2844
                    send_data['white'][1] = ball_list['white']['y']/500*1422
                    ball_path_list = {'red': [[ball_list['red']['x'], ball_list['red']['y']]],
                                  'yellow': [[ball_list['yellow']['x'], ball_list['yellow']['y']]],
                                  'white': [[ball_list['white']['x'], ball_list['white']['y']]]}
                    event_checker = Event_checker(ball_path_list, radius)
                else:
                    continue
            else:
                if is_shooted==True:
                    shoot_count +=1
                show_img, ball_list, _ = find_ball(img, show_img, img_color, event_checker)
                if ball_list != None:
                    show_img = event_checker.tracker(img, show_img, ball_list)

                    if shoot_count==1: #shoot 다음 프레임
                        send_data['shoot_velocity'] = event_checker.get_valocity()
                        x_value = (event_checker.ball_path_list[send_data['start']][-1][0] - event_checker.ball_path_list[send_data['start']][-2][0]) / 1000 * 2844
                        y_value = (event_checker.ball_path_list[send_data['start']][-1][1] -
                                                     event_checker.ball_path_list[send_data['start']][-2][1]) / 500 * 1422
                        send_data['direction'][0] = x_value/max(abs(x_value), abs(y_value))
                        send_data['direction'][1] = y_value/max(abs(x_value), abs(y_value))

                    event = event_checker.check_event(t, img_color)
                    if event == 'shoot':
                        send_data['start'] = event_checker.first_ball['color']
                        is_shooted=True
                    if event == 'object1':
                        send_data['object1']=event_checker.second_ball['color']
                        send_data['shoot'] = event_checker.shootpoint
                        if event_checker.init_ball_pos!=None and len(event_checker.init_ball_pos)==3:
                            send_data['red'][0] = event_checker.init_ball_pos[0]['x'] / 1000 * 2844
                            send_data['red'][1] = event_checker.init_ball_pos[0]['y'] / 500 * 1422
                            send_data['yellow'][0] = event_checker.init_ball_pos[1]['x'] / 1000 * 2844
                            send_data['yellow'][1] = event_checker.init_ball_pos[1]['y'] / 500 * 1422
                            send_data['white'][0] = event_checker.init_ball_pos[2]['x'] / 1000 * 2844
                            send_data['white'][1] = event_checker.init_ball_pos[2]['y'] / 500 * 1422
                        #print(send_data)

                    state = event_checker.score
            cv2.imshow("img", show_img)
            cv2.waitKey(1)

        # event_checker.remove_lf()

        event_checker.write_json(send_data, vid_name)

        event_checker.check_result()
        event_checker.path_analysis()
        event_checker.event['path']=path_to_name[event_checker.final_path]
        event_checker.event['isround'] = event_checker.is_round_table
        event_checker.event['score'] = event_checker.score

        # with open('./jsons/'+str(name)+'.json', 'w', encoding='utf-8') as make_file:
        #     json.dump(event_checker.event, make_file, indent="\t")

        print("구종: ", path_to_name[event_checker.final_path])
        print("대회전 여부: ", event_checker.is_round_table)
        print("득점 여부: ", event_checker.score)

        # cv2.destroyAllWindows()

        print("총 실행 시간: ", time.time()-start_time)

        if event_checker.score != True:
            print(event_checker.fail_description)
        del imgs
        return
    # except:
    #     return

def file_move(input_path):
    import shutil
    while(1):
        try:
            shutil.move('./input_file/' + input_path, './processing_file/'+input_path.split('/')[-1])
            return str('./processing_file/'+input_path.split('/')[-1])
        except:
            continue

if __name__ == '__main__':

    print("[Program Start]")
    predict_func(args.input)

    #t = threading.Thread()