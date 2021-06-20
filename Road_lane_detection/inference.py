import cv2
import numpy as np
from road_lane_detection import pipeline, init_all
from helpers import get_screenshot
import keyboard
import pyvjoy


def move(x, y):
    device.data.wAxisX = x
    device.data.wAxisY = y
    device.update()


birdEye, curves, cap = init_all()
go = False

mn, md, mx = 16120, 16383, 16730
device = pyvjoy.VJoyDevice(1)
curX, curY = 16383, 0
move(curX, curY)

while True:
    try:
        if keyboard.is_pressed("o"):
            go = True
        elif keyboard.is_pressed("i"):
            go = False
        elif cv2.waitKey(1) and keyboard.is_pressed("q"):
            go = False
        screen = np.array(get_screenshot())
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
        if np.average(screen) > 80:
            offset = 50
        else:
            offset = 30
        resized = screen[620:-60, 400:-600]
        lines_image, position = pipeline(resized, birdEye, curves, offset)
        print(position)
        cv2.imshow("lines image", lines_image)
        if go:
            if -0.3 < position < -0.1:
                curX = mn - 200
            elif position <= -0.3:
                curX = mn
            elif 0.3 > position > 0.1:
                curX = mx
            elif position >= 0.3:
                curX = mx + 200
            else:
                curX = md
            move(curX, curY)
            print(curX, curY)
        else:
            move(md, 0)


    except:
        print('pass')
        move(md, 0)
