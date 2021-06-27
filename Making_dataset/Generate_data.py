import time
import cv2
import pygame
import json
import pyautogui
import numpy as np

for i in range(5, 0, -1):
    print(i)
    time.sleep(1)

dic = {
    "names": [],
    'x': [],
    "count": 0
}

work = False
reset = False

pygame.init()
joystick = pygame.joystick.Joystick(0)
joystick.init()
x = 0

while True:
    if work:
        screen = np.array(pyautogui.screenshot(region=(600, 500, 800, 500)))
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(screen, (256, 256))
        cv2.imwrite(f"data/{dic['count']}.jpg", resized)
        dic['names'].append(f'{str(dic["count"])}.jpg')
        dic['x'].append(x)
        dic['count'] += 1
        if reset:
            dic['names'] = dic['names'][:-4]
            dic['x'] = dic['x'][:-4]
            dic['count'] -= 4
            print(len(dic['names']))
        if dic['count'] % 100 == 0:
            with open("data/data.json", "w") as j:
                json.dump(dic, j)
        time.sleep(0.3)

    events = pygame.event.get()
    for event in events:
        if event.type == pygame.JOYAXISMOTION:
            if event.axis == 0:
                x = event.value
            # elif event.axis == 4:
            #     positions[1] = round((event.value + 1) / 2, 3)
            # elif event.axis == 5:
            #     positions[2] = round((event.value + 1) / 2, 3)

        elif event.type == pygame.JOYBUTTONDOWN:
            if event.button == 0:
                work = True
            elif event.button == 1:
                work = False
            elif event.button == 3:
                reset = True
        elif event.type == pygame.JOYBUTTONUP:
            if event.button == 3:
                reset = False
    if not reset:
        print(work, dic['count'], x)
