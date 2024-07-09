import numpy as np
import cv2
import os

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades
# download the below files from the abover url and save them in opencv folder
face_cascade = cv2.CascadeClassifier('opencv/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('opencv/haarcascade_eye.xml')

video_feed = os.environ.get('DROIDCAM_URL', "http://172.16.3.78:4747/video") # using droidcam on phone, use 0 for webcam on windows
cap = cv2.VideoCapture(video_feed) 
if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)

while True:
    ret, img = cap.read()
    if img is None:
        print('--(!) No captured frame -- Break!')
        break
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break