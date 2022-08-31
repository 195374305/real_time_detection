import cv2 as cv
import numpy as np
import cv2
from mss import mss
from PIL import Image

bounding_box = {'top': 340, 'left': 800, 'width': 350, 'height': 400}

sct = mss()


while True:
    sct_img = sct.grab(bounding_box)
    scr_img = np.array(sct_img)

    #cv2.imshow('screen', scr_img) # display screen in box
    cv.imshow('Testing', scr_img)

    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        cv2.destroyAllWindows()
        break