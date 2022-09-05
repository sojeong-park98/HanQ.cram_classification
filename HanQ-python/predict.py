# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from dataloader import Image_loader
import cv2
import numpy as np
import math
import argparse
import csv
import pandas as pd

# 길게치기 LONGSHOT
# 앞돌리기 INSIDEANGLESHOT
# 옆돌리기 SIDESHOT
# 빗겨치기 BIASANGLESHOT
# 뒤돌리기 OUTSIDEANGLESHOT
# 횡단샷 CROSSSHOT
# 리버스 REVERSESHOT
# 더블쿠션 DOUBLECUSSIONSHOT
# 되돌아오기 PLATESHOT
# 1뱅크 ONEBANKSHOT
# 2뱅크 TWOBANKSHOT
# 3뱅크 THREEBANKSHOT
# 대회전 ROUNDTABLE

path_dict={
    'LONGSHOT':0,
    'INSIDEANGLESHOT':1,
    'SIDESHOT':2,
    'BIASANGLESHOT':3,
    'OUTSIDEANGLESHOT':4,
    'CROSSSHOT':5,
    'REVERSESHOT':6,
    'DOUBLECUSSIONSHOT':7,
    'PLATESHOT':8,
    'ONEBANKSHOT':9,
    'TWOBANKSHOT':10,
    'THREEBANKSHOT':11,
    'ROUNDTABLE':12
}

path_to_name={
    0:'LONGSHOT',
    1:'INSIDEANGLESHOT',
    2:'SIDESHOT',
    3:'BIASANGLESHOT',
    4:'OUTSIDEANGLESHOT',
    5:'CROSSSHOT',
    6:'REVERSESHOT',
    7:'DOUBLECUSSIONSHOT',
    8:'PLATESHOT',
    9:'ONEBANKSHOT',
    10:'TWOBANKSHOT',
    11:'THREEBANKSHOT',
    12:'ROUNDTABLE'
}

parser = argparse.ArgumentParser(description='HanQ')
parser.add_argument('--input', default='../data/billiard_data/side1', type=str)
args = parser.parse_args()

BALL_CRASH_THRES = 20
TOP_CRASH_THRES = 40
BUTTOM_CRASH_THRES = 40
LEFT_CRASH_THRES = 50
RIGHT_CRASH_THRES = 60
START_VALOCITY_THRES = 10
SHOOTING_DISTANCE_THRES = 100
START_TRACKING_THRES = 3
MIN_BALL_AREA = 130
MAX_BALL_AREA = 250

class event_checker():
    def __init__(self, table, ball_path_list):
        self.time=0
        self.first_ball = None
        self.second_ball = None
        self.third_ball = None
        self.score = None
        self.cussion_count = 0
        self.top = table['top']
        self.buttom = table['buttom']
        self.left = table['left']
        self.right = table['right']
        self.check_spin = False
        self.kiss = False

        self.dist_white_yellow = 99999
        self.dist_white_red = 99999
        self.dist_yellow_red = 99999

        self.valocity_red = -1
        self.valocity_yellow = -1
        self.valocity_white = -1

        self.ball_path_list = ball_path_list

        self.event = {"event_num":0}
        self.is_event=False

        self.captured_img = None
        self.final_path = None
        self.is_round_table = False

        self.pd_data = None

    def tracker(self, img, show_img, ball_list):
        if len(ball_list) == START_TRACKING_THRES:
            red_dist = np.sqrt(np.sum(
                np.square(np.array(self.ball_path_list['red'][-1]) - np.array([ball_list[0]['x'], ball_list[0]['y']]))))
            yellow_dist = np.sqrt(np.sum(
                np.square(np.array(self.ball_path_list['yellow'][-1]) - np.array([ball_list[1]['x'], ball_list[1]['y']]))))
            white_dist = np.sqrt(np.sum(
                np.square(np.array(self.ball_path_list['white'][-1]) - np.array([ball_list[2]['x'], ball_list[2]['y']]))))

            if red_dist < SHOOTING_DISTANCE_THRES:
                self.ball_path_list['red'].append([ball_list[0]['x'], ball_list[0]['y']])
                show_img = cv2.polylines(show_img, [np.array(self.ball_path_list['red'])], False, (0, 0, 255), 1)
            if yellow_dist < SHOOTING_DISTANCE_THRES:
                self.ball_path_list['yellow'].append([ball_list[1]['x'], ball_list[1]['y']])
                show_img = cv2.polylines(show_img, [np.array(self.ball_path_list['yellow'])], False, (0, 255, 255), 1)
            if white_dist < SHOOTING_DISTANCE_THRES:
                self.ball_path_list['white'].append([ball_list[2]['x'], ball_list[2]['y']])
                show_img = cv2.polylines(show_img, [np.array(self.ball_path_list['white'])], False, (255, 255, 255), 1)

        return show_img

    def capture_shoot(self, img, table):
        _, img_binary = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
        img_binary[:, :table['left'] + 10] = 0
        img_binary[:, table['right'] - 10:] = 0
        img_binary[:table['top'] + 10, :] = 0
        img_binary[table['buttom'] - 10:, :] = 0

        enum = {
            'red':0,
            'yellow':1,
            'white':2
        }

        # 공 주변 10픽셀 이외에 다 지우고 시각화해서 확인
        cv2.circle(img_binary, (ball_list[enum[self.first_ball['color']]]['x'], ball_list[enum[self.first_ball['color']]]['y']), ball_list[enum[self.first_ball['color']]]['radius'], 0, -1)
        cv2.circle(img_binary,
                   (ball_list[enum[self.first_ball['color']]]['x'], ball_list[enum[self.first_ball['color']]]['y']),
                   ball_list[enum[self.first_ball['color']]]['radius'], 0, 3)
        img_binary[:,:ball_list[enum[self.first_ball['color']]]['x']-50]=0
        img_binary[:, ball_list[enum[self.first_ball['color']]]['x'] + 50:] = 0
        img_binary[:ball_list[enum[self.first_ball['color']]]['y'] - 50, :] = 0
        img_binary[ball_list[enum[self.first_ball['color']]]['x'] + 50:, :] = 0

        return img_binary

    def calc_distance(self):
        self.dist_white_yellow = np.sqrt(np.sum(np.square(np.array(self.ball_path_list['yellow'][-1]) - np.array(self.ball_path_list['white'][-1]))))
        self.dist_white_red = np.sqrt(np.sum(np.square(np.array(self.ball_path_list['white'][-1]) - np.array(self.ball_path_list['red'][-1]))))
        self.dist_yellow_red = np.sqrt(np.sum(np.square(np.array(self.ball_path_list['yellow'][-1]) - np.array(self.ball_path_list['red'][-1]))))

    def calc_valocity(self):
        self.valocity_white = np.sqrt(np.sum(np.square(np.array(self.ball_path_list['white'][-1]) - np.array(self.ball_path_list['white'][-2]))))
        self.valocity_red = np.sqrt(np.sum(np.square(np.array(self.ball_path_list['red'][-1]) - np.array(self.ball_path_list['red'][-2]))))
        self.valocity_yellow = np.sqrt(np.sum(np.square(np.array(self.ball_path_list['yellow'][-1]) - np.array(self.ball_path_list['yellow'][-2]))))

    def check_ball_left_right(self):
        v1 = np.array(self.first_ball['shoot_position']) - np.array(self.first_ball['position'])
        v2 = np.array(self.second_ball['position']) - np.array(self.first_ball['position'])
        theta = np.arcsin((v1[0]*v2[1]-v1[1]*v2[0])/(np.sqrt(v1[0]**2+v1[1]**2)*np.sqrt(v2[0]**2+v2[1]**2)))
        if theta < 0:
            self.event[self.event['event_num']]['hit_point'] = 'right'
        else:
            self.event[self.event['event_num']]['hit_point'] = 'left'
        print(self.event[self.event['event_num']]['hit_point'])
        self.event['event_num'] += 1
        self.is_event = True

    def check_first_ball(self, ball_path_list):
        if len(ball_path_list['red'])>=START_TRACKING_THRES and len(ball_path_list['yellow'])>=START_TRACKING_THRES and len(ball_path_list['white'])>=START_TRACKING_THRES:
            if self.valocity_red >= START_VALOCITY_THRES:
                self.first_ball = {'color': 'red', 'position':ball_path_list['red'][-1]}
                self.score = False
                self.event[self.event['event_num']] = {'name': 'shoot', 'discribe': '첫번째 공 빨간색', 'pos': ball_path_list['red'][-1], 'time': self.time}
                print(self.event[self.event['event_num']]['discribe'])
                self.event['event_num'] += 1
            if self.valocity_yellow >= START_VALOCITY_THRES:
                self.first_ball = {'color': 'yellow', 'position':ball_path_list['yellow'][-1]}
                self.event[self.event['event_num']] = {'name': 'shoot', 'discribe': '첫번째 공 노란색', 'pos': ball_path_list['yellow'][-1], 'time': self.time}
                print(self.event[self.event['event_num']]['discribe'])
                self.event['event_num'] += 1
            if self.valocity_white >= START_VALOCITY_THRES:
                self.first_ball = {'color': 'white', 'position':ball_path_list['white'][-1]}
                self.event[self.event['event_num']] = {'name': 'shoot', 'discribe': '첫번째 공 하얀색', 'pos': ball_path_list['white'][-1], 'time': self.time}
                print(self.event[self.event['event_num']]['discribe'])
                self.event['event_num'] += 1
    def check_second_ball(self):
        if self.first_ball['color'] == 'yellow':
            if self.dist_yellow_red < BALL_CRASH_THRES:
                self.second_ball = {'color': 'red', 'position':self.ball_path_list['red'][-1]}
                self.first_ball['shoot_position']=self.ball_path_list['yellow'][-1]
                self.event[self.event['event_num']] = {'name': 'object1', 'discribe': '제1목적구 빨간공', 'pos': ball_path_list['yellow'][-1], 'time': self.time}
                print(self.event[self.event['event_num']]['discribe'])
            elif self.dist_white_yellow < BALL_CRASH_THRES:
                self.second_ball = {'color': 'white', 'position':self.ball_path_list['white'][-1]}
                self.first_ball['shoot_position'] = self.ball_path_list['yellow'][-1]
                self.event[self.event['event_num']] = {'name': 'object1', 'discribe': '제1목적구 흰공', 'pos': ball_path_list['yellow'][-1], 'time': self.time}
                print(self.event[self.event['event_num']]['discribe'])
        if self.first_ball['color'] == 'white':
            if self.dist_white_red < BALL_CRASH_THRES:
                self.second_ball = {'color': 'red', 'position':self.ball_path_list['red'][-1]}
                self.first_ball['shoot_position'] = self.ball_path_list['white'][-1]
                self.event[self.event['event_num']] = {'name': 'object1', 'discribe': '제1목적구 빨간공', 'pos': ball_path_list['white'][-1], 'time': self.time}
                print(self.event[self.event['event_num']]['discribe'])
            elif self.dist_white_yellow:
                self.second_ball = {'color': 'yellow', 'position':self.ball_path_list['yellow'][-1]}
                self.first_ball['shoot_position'] = self.ball_path_list['white'][-1]
                self.event[self.event['event_num']] = {'name': 'object1', 'discribe': '제1목적구 노란공', 'pos': ball_path_list['white'][-1], 'time': self.time}
                print(self.event[self.event['event_num']]['discribe'])
    def check_third_ball(self):
        if self.first_ball['color'] == 'yellow':
            if self.second_ball['color'] == 'red':
                if self.dist_white_yellow < BALL_CRASH_THRES:
                    self.third_ball = {'color': 'white', 'position':self.ball_path_list['white'][-1]}
                    self.event[self.event['event_num']] = {'name': 'object2', 'discribe': '제2목적구 흰공', 'pos': ball_path_list['yellow'][-1], 'time': self.time}
                    print(self.event[self.event['event_num']]['discribe'])
                    self.event['event_num'] += 1
                    self.is_event = True
            elif self.second_ball['color'] == 'white':
                if self.dist_yellow_red < BALL_CRASH_THRES:
                    self.third_ball = {'color': 'red', 'position':self.ball_path_list['red'][-1]}
                    self.event[self.event['event_num']] = {'name': 'object2', 'discribe': '제2목적구 빨간공', 'pos': ball_path_list['yellow'][-1], 'time': self.time}
                    print(self.event[self.event['event_num']]['discribe'])
                    self.event['event_num'] += 1
                    self.is_event = True
        if self.first_ball['color'] == 'white':
            if self.second_ball['color'] == 'red':
                if self.dist_white_yellow < BALL_CRASH_THRES:
                    self.third_ball = {'color': 'yellow', 'position':self.ball_path_list['yellow'][-1]}
                    self.event[self.event['event_num']] = {'name': 'object2', 'discribe': '제2목적구 노란공', 'pos': ball_path_list['white'][-1], 'time': self.time}
                    print(self.event[self.event['event_num']]['discribe'])
                    self.event['event_num'] += 1
                    self.is_event = True
            elif self.second_ball['color'] == 'yellow':
                if self.dist_white_red < BALL_CRASH_THRES:
                    self.third_ball = {'color': 'red', 'position':self.ball_path_list['red'][-1]}
                    self.event[self.event['event_num']] = {'name': 'object2', 'discribe': '제2목적구 빨간공', 'pos': ball_path_list['white'][-1], 'time': self.time}
                    print(self.event[self.event['event_num']]['discribe'])
                    self.event['event_num'] += 1
                    self.is_event = True

    def check_bend_path(self):
        rad_prev = math.atan2(self.ball_path_list[self.first_ball['color']][-3][1] - self.ball_path_list[self.first_ball['color']][-2][1], self.ball_path_list[self.first_ball['color']][-3][0] - self.ball_path_list[self.first_ball['color']][-2][0])
        rad_cur = math.atan2(self.ball_path_list[self.first_ball['color']][-2][1] - self.ball_path_list[self.first_ball['color']][-1][1], self.ball_path_list[self.first_ball['color']][-2][0] - self.ball_path_list[self.first_ball['color']][-1][0])
        deg_prev = (rad_prev*180)/math.pi
        deg_cur = (rad_cur*180)/math.pi

        return np.abs(deg_prev-deg_cur)

    def check_cussion(self):
        if np.abs(self.ball_path_list[self.first_ball['color']][-1][1] - self.top) < TOP_CRASH_THRES \
                and (self.ball_path_list[self.first_ball['color']][-1][1]-self.ball_path_list[self.first_ball['color']][-2][1])\
                *(self.ball_path_list[self.first_ball['color']][-2][1]-self.ball_path_list[self.first_ball['color']][-3][1]) < 0:
            self.cussion_count += 1
            self.check_spin = 'top'
            self.event[self.event['event_num']] = {'name': 'cussion', 'discribe': '윗쿠션 침', 'cussion': 'top', 'pos': self.ball_path_list[self.first_ball['color']][-1], 'time': self.time}
            print(self.event[self.event['event_num']]['discribe'])
        if np.abs(self.ball_path_list[self.first_ball['color']][-1][1] - self.buttom) < BUTTOM_CRASH_THRES \
                and (self.ball_path_list[self.first_ball['color']][-1][1]-self.ball_path_list[self.first_ball['color']][-2][1])\
                *(self.ball_path_list[self.first_ball['color']][-2][1]-self.ball_path_list[self.first_ball['color']][-3][1]) < 0:
            self.cussion_count += 1
            self.check_spin = 'buttom'
            self.event[self.event['event_num']] = {'name': 'cussion', 'discribe': '아래 쿠션 침', 'cussion': 'buttom', 'pos': self.ball_path_list[self.first_ball['color']][-1], 'time': self.time}
            print(self.event[self.event['event_num']]['discribe'])
        if np.abs(self.ball_path_list[self.first_ball['color']][-1][0] - self.left) < LEFT_CRASH_THRES \
                and (self.ball_path_list[self.first_ball['color']][-1][0]-self.ball_path_list[self.first_ball['color']][-2][0])\
                *(self.ball_path_list[self.first_ball['color']][-2][0]-self.ball_path_list[self.first_ball['color']][-3][0]) < 0:
            self.cussion_count += 1
            self.check_spin = 'left'
            self.event[self.event['event_num']] = {'name': 'cussion', 'discribe': '왼쪽 쿠션 침', 'cussion': 'left', 'pos': self.ball_path_list[self.first_ball['color']][-1], 'time': self.time}
            print(self.event[self.event['event_num']]['discribe'])
        if np.abs(self.ball_path_list[self.first_ball['color']][-1][0] - self.right) < RIGHT_CRASH_THRES \
                and (self.ball_path_list[self.first_ball['color']][-1][0]-self.ball_path_list[self.first_ball['color']][-2][0])\
                *(self.ball_path_list[self.first_ball['color']][-2][0]-self.ball_path_list[self.first_ball['color']][-3][0]) < 0:
            self.cussion_count += 1
            self.check_spin = 'right'
            self.event[self.event['event_num']] = {'name': 'cussion', 'discribe': '오른 쿠션 침', 'cussion': 'right', 'pos': self.ball_path_list[self.first_ball['color']][-1], 'time': self.time}
            print(self.event[self.event['event_num']]['discribe'])

    def check_kiss(self):
        if self.first_ball == 'yellow':
            if np.sqrt(np.sum(np.square(np.array(ball_path_list['white'][-1]) - np.array(ball_path_list['red'][-1])))) < BALL_CRASH_THRES:
                self.kiss = True
                self.event[self.event['event_num']] = {'name': 'kiss', 'discribe': 'Kiss-1목적구 2목적구 충돌', 'pos': self.ball_path_list[self.first_ball['color']][-1], 'time': self.time}
                print(self.event[self.event['event_num']]['discribe'])
                self.event['event_num'] += 1
                self.is_event = True
            if self.second_ball != None:
                if np.sqrt(np.sum(np.square(np.array(ball_path_list['yellow'][-1]) - np.array(ball_path_list[self.second_ball][-1])))) < BALL_CRASH_THRES:
                    self.kiss = True
                    self.event[self.event['event_num']] = {'name': 'kiss', 'discribe': 'Kiss-1목적구 재충돌', 'pos': self.ball_path_list[self.first_ball['color']][-1], 'time': self.time}
                    print(self.event[self.event['event_num']]['discribe'])
                    self.event['event_num'] += 1
                    self.is_event = True
        if self.first_ball == 'white':
            if np.sqrt(np.sum(np.square(np.array(ball_path_list['yellow'][-1]) - np.array(ball_path_list['red'][-1])))) < BALL_CRASH_THRES:
                self.kiss = True
                self.event[self.event['event_num']] = {'name': 'kiss', 'discribe': 'Kiss-1목적구 2목적구 충돌', 'pos': self.ball_path_list[self.first_ball['color']][-1], 'time': self.time}
                print(self.event[self.event['event_num']]['discribe'])
                self.event['event_num'] += 1
                self.is_event = True
            if self.second_ball != None:
                if np.sqrt(np.sum(np.square(np.array(ball_path_list['white'][-1]) - np.array(ball_path_list[self.second_ball][-1])))) < BALL_CRASH_THRES:
                    self.kiss = True
                    self.event[self.event['event_num']] = {'name': 'kiss', 'discribe': 'Kiss-1목적구 재충돌', 'pos': self.ball_path_list[self.first_ball['color']][-1], 'time': self.time}
                    print(self.event[self.event['event_num']]['discribe'])
                    self.event['event_num'] += 1
                    self.is_event = True

    def check_spin_direction(self):
        if self.check_spin=='top':
            if ball_path_list[self.first_ball['color']][-1][0] > ball_path_list[self.first_ball['color']][-3][0]:
                self.event[self.event['event_num']]['spin'] = 'right'
                print(self.event[self.event['event_num']]['spin'])
                self.event['event_num'] += 1
                self.is_event = True
            else:
                self.event[self.event['event_num']]['spin'] = 'left'
                print(self.event[self.event['event_num']]['spin'])
                self.event['event_num'] += 1
                self.is_event = True
        elif self.check_spin=='buttom':
            if ball_path_list[self.first_ball['color']][-1][0] < ball_path_list[self.first_ball['color']][-3][0]:
                self.event[self.event['event_num']]['spin'] = 'right'
                print(self.event[self.event['event_num']]['spin'])
                self.event['event_num'] += 1
                self.is_event = True
            else:
                self.event[self.event['event_num']]['spin'] = 'left'
                print(self.event[self.event['event_num']]['spin'])
                self.event['event_num'] += 1
                self.is_event = True
        elif self.check_spin=='left':
            if ball_path_list[self.first_ball['color']][-1][1] < ball_path_list[self.first_ball['color']][-3][1]:
                self.event[self.event['event_num']]['spin'] = 'right'
                print(self.event[self.event['event_num']]['spin'])
                self.event['event_num'] += 1
                self.is_event = True
            else:
                self.event[self.event['event_num']]['spin'] = 'left'
                print(self.event[self.event['event_num']]['spin'])
                self.event['event_num'] += 1
                self.is_event = True
        elif self.check_spin=='right':
            if ball_path_list[self.first_ball['color']][-1][1] > ball_path_list[self.first_ball['color']][-3][1]:
                self.event[self.event['event_num']]['spin'] = 'right'
                print(self.event[self.event['event_num']]['spin'])
                self.event['event_num'] += 1
                self.is_event = True
            else:
                self.event[self.event['event_num']]['spin'] = 'left'
                print(self.event[self.event['event_num']]['spin'])
                self.event['event_num'] += 1
                self.is_event = True
        self.check_spin = False

    def check_result(self):
        if self.cussion_count>=3 and self.first_ball != None and self.second_ball != None and self.third_ball != None:
            self.score = True
        if self.score == True:
            print("Success")
        else:
            if self.first_ball == None:
                print("수구 안침")
            elif self.second_ball == None:
                print("1 목적구 안침")
            elif self.third_ball == None:
                print("2 목적구 안침")
            elif self.cussion_count <3:
                print("쓰리쿠 안침")

    def check_event(self, time):

        self.time=time
        self.calc_distance()
        self.calc_valocity()

        # 쪽났는지 확인하자
        if self.first_ball != None:
            self.check_kiss()

        #수구가 무엇인지 알아보자
        if self.first_ball == None:
            self.check_first_ball(ball_path_list)
            if self.first_ball != None:
                self.captured_img = self.capture_shoot(img, table)
        #수구를 쳤다면 1목적구가 무엇인지 알아보자
        elif self.second_ball == None:
            self.check_second_ball()
            if self.second_ball != None:
                self.check_ball_left_right()
        #쿠션을 쳤는지 확인하자
        if self.first_ball != None:
            self.check_cussion()
        #스핀을 확인하자
        if self.check_spin != False:
            self.check_spin_direction()

        return show_img

    def write_data(self, time):
        top=self.top
        buttom=self.buttom
        left=self.left
        right=self.right
        if self.first_ball != None:
            firstball_pos=np.array(self.ball_path_list[self.first_ball['color']][-1])-np.array([self.left, self.top])
        else:
            firstball_pos=['NA', 'NA']
        if self.second_ball != None:
            secondball_pos=np.array(self.ball_path_list[self.second_ball['color']][-1])-np.array([self.left, self.top])
        else:
            secondball_pos=['NA', 'NA']
        if self.third_ball != None:
            thirdball_pos=np.array(self.ball_path_list[self.third_ball['color']][-1])-np.array([self.left, self.top])
        else:
            thirdball_pos=['NA', 'NA']
        if len(ball_path_list)>=3:
            if self.first_ball!=None :
                firstball_movement=np.array(self.ball_path_list[self.first_ball['color']][-1])-np.array(self.ball_path_list[self.first_ball['color']][-2])
            else:
                firstball_movement=['NA', 'NA']
            if self.second_ball!=None :
                secondball_movement=np.array(self.ball_path_list[self.second_ball['color']][-1])-np.array(self.ball_path_list[self.second_ball['color']][-2])
            else:
                secondball_movement=['NA', 'NA']
            if self.third_ball!=None:
                thirdball_movement=np.array(self.ball_path_list[self.third_ball['color']][-1])-np.array(self.ball_path_list[self.third_ball['color']][-2])
            else:
                thirdball_movement=['NA', 'NA']
        else:
            firstball_movement = ['NA', 'NA']
            secondball_movement = ['NA', 'NA']
            thirdball_movement = ['NA', 'NA']
        if time == 0:
            self.pd_data=pd.DataFrame(columns=['time', 'top', 'buttom', 'left', 'right', 'firstball_pos_x', 'firstball_pos_y',
                         'secondball_pos_x', 'secondball_pos_y', 'thirdball_pos_x', 'thirdball_pos_y', 'firstball_movement_x', 'firstball_movement_y',
                         'secondball_movement_x', 'secondball_movement_y', 'thirdball_movement_x', 'thirdball_movement_y'])
        self.pd_data.loc[time]={'time': time, 'top': top, 'buttom': buttom, 'left': left, 'right': right, 'firstball_pos_x': firstball_pos[0],
                                'firstball_pos_y': firstball_pos[1], 'secondball_pos_x': secondball_pos[0], 'secondball_pos_y': secondball_pos[1],
                                'thirdball_pos_x': thirdball_pos[0], 'thirdball_pos_y': thirdball_pos[1], 'firstball_movement_x': firstball_movement[0],
                               'firstball_movement_y': firstball_movement[1], 'secondball_movement_x': secondball_movement[0], 'secondball_movement_y': secondball_movement[1],
                               'thirdball_movement_x': thirdball_movement[0], 'thirdball_movement_y': thirdball_movement[1]}

    def write_event(self):
        self.pd_data['event']='NA'
        self.pd_data['direction']='NA'
        for event in self.event:
            try:
                int(event)
                self.pd_data.loc[self.event[event]['time'], 'event'] = self.event[event]['name']
                if self.event[event]['name']=='shoot':
                    direction = self.event[event]['shootpoint']
                elif self.event[event]['name']=='object1' or self.event[event]['name']=='object2':
                    direction = self.event[event]['hit_point']
                elif self.event[event]['name']=='cussion':
                    direction = self.event[event]['spin']
                self.pd_data.loc[self.event[event]['time'], 'direction'] = direction
            except:
                continue

    def path_analysis(self):
        ball_point = None
        ball_direction = None
        for event in self.event:
            try:
                if self.event[event]['name']=='shoot':
                    ball_point = self.event[event]['pos']
                    ball_direction = self.event[event+1]['pos']
                    break
            except:
                continue

        y_idx, x_idx = np.where(self.captured_img != 0)
        min_dis = 99999
        cue_point = None
        for x, y in zip(x_idx, y_idx):
            dis = np.sqrt(np.abs(x-ball_point[0])**2+np.abs(y-ball_point[1])**2)
            if dis < min_dis:
                min_dis = dis
                cue_point = [x, y]
        if cue_point != None:
            v1 = np.array(ball_point) - np.array(cue_point)
            v2 = np.array(ball_direction) - np.array(ball_point)
            theta = np.arcsin(
                (v1[0] * v2[1] - v1[1] * v2[0]) / (np.sqrt(v1[0] ** 2 + v1[1] ** 2) * np.sqrt(v2[0] ** 2 + v2[1] ** 2)))
            if theta < -0.1:
                shoot_point = 'left'
            elif theta > 0.1:
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
                          "left", "top"], ["left", "top", "right"]]
            counter_clock_wise = [["top", "left", "buttom"], ["right", "top", "left"], ["buttom", "right", "top"], ["left",
                                  "buttom", "right"]]
            cussion_list=[]
            cussion_num = 0
            for event in self.event:
                try:
                    int(event)
                    if self.event[event]['name']=='cussion':
                        cussion_list.append(self.event[event]['cussion'])
                        cussion_num += 1
                    if self.event[event]['name']=='object2':
                        break
                    if cussion_num >= 3:
                        break
                except:
                    continue
            print(cussion_list)
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
                if cross == 'long' or cross == 'short':
                    is_pure = True
                    for event in self.event:
                        try:
                            int(event)
                            if self.event[event]['name'] == 'cussion':
                                if cross == 'long' and (self.event[event]['cussion']=='left' or self.event[event]['cussion']=='right'):
                                    is_pure = False
                                    break
                                elif cross == 'short' and (self.event[event]['cussion']=='top' or self.event[event]['cussion']=='buttom'):
                                    is_pure = False
                                    break
                        except:
                            continue
                    if is_pure:
                        self.final_path=path_dict['CROSSSHOT']
                    else:
                        shootpoint=None
                        first_cussion_spin=None
                        for event in self.event:
                            try:
                                int(event)
                                if self.event[event]['name'] == 'shoot':
                                    shoot_point = self.event[event]['shootpoint']
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
                        height = self.buttom-self.top
                        under_point = self.top+height*0.625
                        up_point = self.top+height*0.375
                        if way == 'clock_wise' and three_cussion == 'left':
                            if three_cussion_point[1]>=under_point:
                                self.final_path=path_dict['LONGSHOT']
                            else:
                                self.final_path = path_dict['INSIDEANGLESHOT']
                        if way == 'clock_wise' and three_cussion == 'right':
                            if three_cussion_point[1] <= up_point:
                                self.final_path = path_dict['LONGSHOT']
                            else:
                                self.final_path = path_dict['INSIDEANGLESHOT']
                        if way == 'counter_clock_wise' and three_cussion == 'left':
                            if three_cussion_point[1] <= up_point:
                                self.final_path = path_dict['LONGSHOT']
                            else:
                                self.final_path = path_dict['INSIDEANGLESHOT']
                        if way == 'counter_clock_wise' and three_cussion == 'right':
                            if three_cussion_point[1] >= under_point:
                                self.final_path = path_dict['LONGSHOT']
                            else:
                                self.final_path = path_dict['INSIDEANGLESHOT']
                    else:
                        self.final_path=path_dict['SIDESHOT']
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
                                second_ball_point = self.event[event]['pos']
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
                    if first_cussion == 'top' or 'buttom':
                        if first_ball_point[0] < second_ball_point[0]:
                            if first_cussion_point[0] < second_ball_point[0]:
                                self.final_path=path_dict['BIASANGLESHOT']
                            else:
                                self.final_path=path_dict['OUTSIDEANGLESHOT']
                        else:
                            if first_cussion_point[0] > second_ball_point[0]:
                                self.final_path=path_dict['BIASANGLESHOT']
                            else:
                                self.final_path=path_dict['OUTSIDEANGLESHOT']
                    elif first_cussion == 'left' or 'right':
                        if first_ball_point[1] < second_ball_point[1]:
                            if first_cussion_point[1] < second_ball_point[1]:
                                self.final_path=path_dict['BIASANGLESHOT']
                            else:
                                self.final_path=path_dict['OUTSIDEANGLESHOT']
                        else:
                            if first_cussion_point[1] > second_ball_point[1]:
                                self.final_path=path_dict['BIASANGLESHOT']
                            else:
                                self.final_path=path_dict['OUTSIDEANGLESHOT']
                    else:
                        print("구현 안됨")
            if self.score == True and (self.first_ball == path_dict['LONGSHOT'] or self.first_ball == path_dict[
                'INSIDEANGLESHOT'] or self.first_ball == path_dict['BIASANGLESHOT'] or self.first_ball == path_dict[
                                           'OUTSIDEANGLESHOT'] or self.first_ball == path_dict['SIDESHOT']):
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
                    if self.event[event] == 'cussion':
                        bank_count += 1
                    elif self.event[event] == 'object1':
                        break
                except:
                    continue
            if bank_count == 1:
                self.final_path=path_dict['ONEBANKSHOT']
            elif bank_count == 2:
                self.final_path=path_dict['TWOBANKSHOT']
            else:
                self.final_path=path_dict['THREEBANKSHOT']


def find_table(img, show_img):
    _, box_binary = cv2.threshold(img, 100, 255, 0)
    mask = np.zeros_like(box_binary)
    left_list = []
    right_list = []
    top_list = []
    buttom_list = []
    for idx, w in enumerate(box_binary):
        if len(w[w==255]) > len(w[w==0])*0.5:
            if idx < box_binary.shape[0]*0.5:
                top_list.append(idx)
            else:
                buttom_list.append(idx)
    for idx, h in enumerate(box_binary.transpose(1,0)):
        if len(h[h==255]) > len(h[h==0])*0.5:
            if idx < box_binary.shape[1] * 0.5:
                left_list.append(idx)
            else:
                right_list.append(idx)
    left_list = np.array(left_list)
    right_list = np.array(right_list)
    top_list = np.array(top_list)
    buttom_list = np.array(buttom_list)
    top = top_list.max()
    buttom = buttom_list.min()
    left = left_list.max()
    right = right_list.min()
    mask[top, :] = 1
    mask[buttom, :] = 1
    mask[:, left] = 1
    mask[:, right] = 1

    show_img[top, :] = 1
    show_img[buttom, :] = 1
    show_img[:, left] = 1
    show_img[:, right] = 1

    table = {'top': top,
             'buttom': buttom,
             'left': left,
             'right': right}

    return show_img, table

def find_ball(img, show_img, table):
    img_binary = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,15,2)
    contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ball_list = []
    for cnt in contours:
        ball_area = cv2.contourArea(cnt)
        if ball_area <= MIN_BALL_AREA or ball_area >=MAX_BALL_AREA:
            continue
        (x,y), radius = cv2.minEnclosingCircle(cnt)
        if x<table['left'] or x>table['right'] or y<table['top'] or y>table['buttom']:
            continue
        ratio = radius*radius*math.pi/ball_area
        if int(ratio) != 1:
            continue
            #cv2.circle(img, (int(x), int(y)), int(radius), (255,0,0), 2)
        ball_info = {'color': img[int(y),int(x)],
                     'radius': int(radius),
                     'x':int(x),
                     'y':int(y)}
        ball_list.append(ball_info)
    sorted_ball_list = sorted(ball_list, key=lambda ball_list: (ball_list['color']))
    if len(sorted_ball_list) == 3:
        sorted_ball_list[0]['color'] = 'red'
        cv2.circle(show_img, (sorted_ball_list[0]['x'], sorted_ball_list[0]['y']), sorted_ball_list[0]['radius'], (0, 0, 1), 2)
        sorted_ball_list[1]['color'] = 'yellow'
        cv2.circle(show_img, (sorted_ball_list[1]['x'], sorted_ball_list[1]['y']), sorted_ball_list[1]['radius'], (0, 1, 1), 2)
        sorted_ball_list[2]['color'] = 'white'
        cv2.circle(show_img, (sorted_ball_list[2]['x'], sorted_ball_list[2]['y']), sorted_ball_list[2]['radius'], (1, 1, 1), 2)

        return show_img, sorted_ball_list
    return show_img, None

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    imgs = Image_loader(args.input)
    import time

    name = args.input.split('/')[-1]
    directory=''
    for i in args.input.split('/')[:-1]:
        directory += i
        directory += '/'
    path = directory +name+'.csv'
    print("Write in ", path)

    start_time = time.time()
    ball_path_list = None
    for t, img in enumerate(imgs):
        show_img = np.zeros((img.shape[0], img.shape[1],3))
        show_img[:, :, 0] = img/255
        show_img[:, :, 1] = img/255
        show_img[:, :, 2] = img/255
        show_img, table = find_table(img, show_img)
        if ball_path_list == None:
            show_img, ball_list = find_ball(img, show_img, table)

            if ball_list != None:
                print("Find balls")
                print("red: (",ball_list[0]['x'],', ', ball_list[0]['y'], ')')
                print("yellow: (", ball_list[1]['x'], ', ', ball_list[1]['y'], ')')
                print("white: (", ball_list[2]['x'], ', ', ball_list[2]['y'], ')')
                ball_path_list = {'red': [[ball_list[0]['x'], ball_list[0]['y']]],
                              'yellow': [[ball_list[1]['x'], ball_list[1]['y']]],
                              'white': [[ball_list[2]['x'], ball_list[2]['y']]]}
                event_checker = event_checker(table, ball_path_list)
        else:
            show_img, ball_list = find_ball(img, show_img, table)
            if ball_list != None:
                show_img = event_checker.tracker(img, show_img, ball_list)
                event_checker.check_event(t)
        cv2.imshow("img", show_img)
        cv2.waitKey(1)
        event_checker.write_data(t)
    event_checker.check_result()
    event_checker.path_analysis()
    print(event_checker.event)
    print("구종: ", path_to_name[event_checker.final_path])
    print("대회전 여부: ", event_checker.is_round_table)
    cv2.destroyAllWindows()

    print("총 실행 시간: ", time.time()-start_time)
    event_checker.write_event()
    event_checker.pd_data.to_csv(path)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
