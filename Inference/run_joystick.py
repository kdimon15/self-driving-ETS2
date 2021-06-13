import albumentations as albu
import keyboard
import onnxruntime
from albumentations.pytorch import ToTensorV2
import numpy as np
import pyautogui
import cv2
import time
import pyvjoy

for i in list(range(4))[::-1]:
    print(i+1)
    time.sleep(1)


clas_session = onnxruntime.InferenceSession("data/efficientnet-b0.onnx")
transform = albu.Compose([
                        albu.Resize(256, 256),
                        albu.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225]),
                        ToTensorV2()
])


min, mid, max = 0, 16383, 32767
curX, curY = mid, min
device = pyvjoy.VJoyDevice(1)


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def inf_onnx(image):
    ort_inputs = {clas_session.get_inputs()[0].name: to_numpy(image)}
    ort_outs = clas_session.run(None, ort_inputs)
    return ort_outs


def move(x, y):
    device.data.wAxisX = x
    device.data.wAxisY = y
    device.update()


def balance(x, mid):
    if x > mid:
        x -= 1000
        if x < mid:
            x = mid
    elif x < mid:
        x += 1000
        if x > mid:
            x = mid
    return x


def forward(x, y):
    y += 1000
    if y > max:
        y = max
    x = balance(x, mid)
    return x, y


def right(x, y):
    y += 1000
    if y > max:
        cur_y = max
    x += 1000
    if x > max:
        x = max
    return x, y


def left(x, y):
    y += 1000
    if y > max:
        y = max
    x += 1000
    if x < min:
        x = min
    return x, y


def stop(x, y):
    y -= 1000
    if y < min:
        y = min
    x = balance(x, mid)


work = False

while True:
    if keyboard.is_pressed("q"):
        break
    if work:
        screen = np.array(pyautogui.screenshot(region=(920, 470, 760, 450)))
        screen = transform(image=screen)['image'].unsqueeze(0)
        pred = inf_onnx(screen)[0][0][0]
        if pred > 0.8:
            curX, curY = right(curX, curY)
        elif pred < -0.8:
            curX, curY = left(curX, curY)
        else:
            curX, curY = forward(curX, curY)
        print(pred)
        move(curX, curY)
    else:
        time.sleep(0.1)
        curX, curY = mid, min
        move(curX, curY)

    if keyboard.is_pressed("o"):
        work = True
    if keyboard.is_pressed("i"):
        work = False

    print(work, curX, curY)
