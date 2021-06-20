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
        resized = screen[620:-60, 500:-600]
        # cv2.imshow("screenshot", screen)
        if go:
            lines_image, position = pipeline(resized, birdEye, curves)
            cv2.imshow("lines image", lines_image)
            if position < -0.1:
                curX = mn
            elif position > 0.1:
                curX = mx
            else:
                curX = md
            move(curX, curY)
            print(position)
        else:
            move(md, 0)


    except:
        print('pass')
        move(md, 0)
