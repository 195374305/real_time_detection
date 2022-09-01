from unittest import result
import numpy as np
import cv2 as cv
import cv2
from mss import mss
from PIL import Image
# PyTorch Hub
import torch
import pandas as pd
import pyautogui

# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  #可用

model = torch.hub.load('.', 'custom', 'yolov5s.pt', source='local')

# model = torch.hub.load('.', 'custom','yolov5x.pt', source='local',force_reload=True)


bounding_box = {'top': 0, 'left': 0, 'width': 850, 'height': 700}
sct = mss()
number = 0
while True:
    sct_img = sct.grab(bounding_box)
    scr_img = np.array(sct_img)

    # cv2.imshow('screen', scr_img) # display screen in box
    # scr_img = model(scr_img)
    results = model(scr_img)
    results.render()  # updates results.imgs with boxes and labels

    cv.imshow('testing', results.ims[0])

    df = pd.DataFrame(results.pandas().xywh[0])  # 把list转换成DATAframe

    # for i in df.index:
    #     print('索引：', i, '横坐标：', df.xcenter[i], '中坐标', df.ycenter[i], '名字', df.name[i])

    if df.empty != True:
        for i in df.index:
            if df.name[i] == 'person':
              pyautogui.moveTo(df.xcenter[0], df.ycenter[0])  # 定位到识别的第一个单位



    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        cv2.destroyAllWindows()
        break
