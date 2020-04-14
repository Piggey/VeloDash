import cv2, imutils, time
import numpy as np
from time import sleep

cap = cv2.VideoCapture("sample.mp4")
test_frame = cap.read()[1]
dim = test_frame.shape
height, width = dim[0], dim[1]
start_p = (width // 3, height // 3 + 100)
end_p = (width // 2, height - height // 2 + 100)
FPS = cap.get(cv2.CAP_PROP_FPS)

detected_frames = 0
flag = False

def calculateVelocity(frames, distance, FPS):
    time = frames / FPS * (1/3600)
    return distance / time


while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 150, 350)

    field = canny[start_p[1]:end_p[1], start_p[0]:end_p[0]]
    thresh = cv2.threshold(field, 1, 1, cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        if(len(approx) > 1):
            cv2.drawContours(field, [c], -1, (255, 0, 0), 5)
            detected_frames += 1
            flag = True
        elif(flag):
            V = calculateVelocity(detected_frames, 0.009, FPS)
            print(detected_frames)
            print(V)
            detected_frames = 0
            flag = False
            


    cv2.rectangle(canny, start_p, end_p, (255, 0, 0), 2)

    cv2.imshow('gowno', canny)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()