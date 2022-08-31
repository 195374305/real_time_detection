from unittest import result
import numpy as np
import cv2 as cv
import cv2
from mss import mss
from PIL import Image
# PyTorch Hub
import torch

#model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model = torch.hub.load('.', 'custom','yolov5n.pt', source='local')
#model = torch.hub.load('.', 'custom','yolov5x-cls.pt', source='local',force_reload=True)


bounding_box = {'top': 400, 'left': 200, 'width': 800, 'height': 600}
sct = mss()
number = 0
while True:
    sct_img = sct.grab(bounding_box)
    scr_img = np.array(sct_img)

    # cv2.imshow('screen', scr_img) # display screen in box
    # scr_img = model(scr_img)
    results = model(scr_img)
    results.render()  # updates results.imgs with boxes and labels


    for im in results.ims:
        cv.imshow('testing', im)

    if (cv2.waitKey(1) & 0xFF)==ord('q'):
        cv2.destroyAllWindows()
        break
