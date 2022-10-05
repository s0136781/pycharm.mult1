import cv2
import math
import numpy as np

cap = cv2.VideoCapture(r'C:\Users\vlad5\PycharmProjects\pythonProject6\ЛР4_main_video.mov', cv2.CAP_ANY)

_ , frame = cap.read()

while True:
    ret, frame = cap.read()
    res, frame1 = cap.read()
    old = frame.copy()
    res, frame = cap.read()
    if res == False:
        break
    cv2.absdiff(frame, old)

    if not (ret):
        break
    cv2.imshow('frame', frame)
    if cv2.waitKey(25) & 0xFF == 27:
        break

old = frame.copy
cap = cv2.VideoCapture(0)