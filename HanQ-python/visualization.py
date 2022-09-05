import pandas as pd
import numpy as np
import cv2

data = pd.read_csv("D:/data/billiard_data/side1.csv")

width = data.loc[0, 'right']-data.loc[0, 'left']+100
height = data.loc[0, 'buttom']-data.loc[0, 'top']
whiteboard = np.ones((height, width, 3))
whiteboard[:,width-100]=[0,0,0]

for index, row in data.iterrows():
    firstball_x = row.firstball_pos_x
    firstball_y = row.firstball_pos_y
    vec_x = row.firstball_movement_x
    vec_y = row.firstball_movement_y
    if not np.isnan(vec_x) and not np.isnan(vec_y):
        cv2.line(whiteboard, (int(firstball_x), int(firstball_y)), (int(firstball_x+vec_x), int(firstball_y+vec_y)), (0,0,255), 1)
    if row.event==row.event:
        cv2.circle(whiteboard, (int(firstball_x), int(firstball_y)), 3, (0, 255, 0), 2)
        cv2.putText(whiteboard, row.event+'/'+row.direction, (int(firstball_x+vec_x), int(firstball_y+vec_y)), 0, 0.5, (255,0,0), 1, cv2.LINE_AA)

cv2.imshow('visualization', whiteboard)
cv2.waitKey(0)
cv2.destroyAllWindows()