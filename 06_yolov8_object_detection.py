import cv2
import numpy as np
from ultralytics import YOLO

import os



# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # 'n' for nano, can be 's', 'm', 'l', 'x' for larger models

# Video capture (use 0 for webcam, or provide a video file path)
video_feed = os.environ.get('DROIDCAM_URL', "http://172.16.2.101:4747/video") # using droidcam on phone, use 0 for webcam on windows
cap = cv2.VideoCapture(video_feed) 

frame_count = 0
last_results = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Perform detection every 2 frames
    if frame_count % 2 == 0:
        results = model(frame)
        last_results = results

    # Visualize the results on the frame
    if last_results is not None:
        annotated_frame = last_results[0].plot()
    else:
        annotated_frame = frame

    # Display the annotated frame
    cv2.imshow("YOLOv8 Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()