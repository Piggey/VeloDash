import cv2, imutils
import numpy as np

cap = cv2.VideoCapture("sample.mp4")
test_frame = cap.read()[1]
dim = test_frame.shape
height, width = dim[0], dim[1]
start_p = (width // 3, height // 3 + 100)
end_p = (width // 2, height - height // 2 + 100)
FPS = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (width, height))
vels = []

detected_frames = 0
flag = False

def calculateVelocity(frames, distance, FPS):
    # returns km/h
    time = frames / FPS * (1/3600)
    return distance / time

avg = lambda arr: sum(arr) / len(arr)
vel = 0

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 150, 350)

    field = canny[start_p[1]:end_p[1], start_p[0]:end_p[0]]
    frame_field = frame[start_p[1]:end_p[1], start_p[0]:end_p[0]]
    # field = canny[start_p[1] + 45:start_p[1] + 50, start_p[0]:end_p[0]]

    thresh = cv2.threshold(field, 1, 1, cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.015 * peri, True)
        if(len(approx) > 2):
            cv2.drawContours(field, [c], -1, (128, 0, 0), 6)
            cv2.drawContours(frame_field, [c], -1, (0, 0, 255), 2)
            detected_frames += 1
            flag = True

        elif(flag):
            if(detected_frames < 3):
                continue
            vels.append(round(calculateVelocity(detected_frames, 0.009, FPS),0))
            print(f'{detected_frames} caught and used to calculate velocity')
            vel = int(avg(vels))
            detected_frames = 0
            flag = False
            
    # cv2.rectangle(canny, start_p, end_p, (255, 0, 0), 2)
    # cv2.imshow('test', canny)


    cv2.rectangle(frame, start_p, end_p, (0, 255, 0), 1)
    cv2.putText(frame, f'estimated velocity: {vel}kmph', (0, height - 50), cv2.FONT_ITALIC, 2, (128, 128, 128), 2)
    out.write(frame)
    cv2.imshow('VeloDash', frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()