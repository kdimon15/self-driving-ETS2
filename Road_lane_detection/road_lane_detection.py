import matplotlib.pyplot as plt

from functions import *
import time
import numpy as np
import cv2
import keyboard

cap = cv2.VideoCapture("/data/output.mov")

while cap.isOpened():
    ret, frame = cap.read()
    resized = frame[550:-60, 600:1320]
    # cv2.imshow("resized", resized)
    lines_image = find_lines(resized)

    cv2.imshow("frame", frame[300:, :-150])
    cv2.imshow("lines", lines_image)
    if cv2.waitKey(1) and keyboard.is_pressed("q"):
        break
cap.release()

# image = cv2.imread("1.png")
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# plt.imshow(find_lines(image))
# plt.waitforbuttonpress()
