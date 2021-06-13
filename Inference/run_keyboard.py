import albumentations as albu
import keyboard
import onnxruntime
from albumentations.pytorch import ToTensorV2
import numpy as np
import pyautogui
import cv2
import time
import pydirectinput

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


min, mid, max = 0, 16383, 20000
cur_x, cur_y = mid, 0

work = False


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def inf_onnx(image):
    ort_inputs = {clas_session.get_inputs()[0].name: to_numpy(image)}
    ort_outs = clas_session.run(None, ort_inputs)
    return ort_outs


def forward():
    pydirectinput.keyDown("w")
    pydirectinput.keyUp("a")
    pydirectinput.keyUp("d")


def right():
    pydirectinput.keyDown("w")
    pydirectinput.keyDown("d")
    pydirectinput.keyUp("a")


def left():
    pydirectinput.keyDown("w")
    pydirectinput.keyDown("a")
    pydirectinput.keyUp("d")


def stop():
    pydirectinput.keyUp("w")
    pydirectinput.keyUp("a")
    pydirectinput.keyUp("d")


while True:
    if keyboard.is_pressed("q"):
        break

    if work:
        screen = np.array(pyautogui.screenshot(region=(920, 470, 760, 450)))
        screen = transform(image=screen)['image'].unsqueeze(0)
        pred = inf_onnx(screen)[0][0][0]
        if pred > 0.8:
            right()
        elif pred < -0.8:
            left()
        else:
            forward()
        print(pred)

    if keyboard.is_pressed("o"):
        work = True
    if keyboard.is_pressed("i"):
        work = False
    print(work)