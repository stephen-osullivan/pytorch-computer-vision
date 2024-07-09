import cv2
import numpy as np
import torch

import os

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')#

# Video capture (use 0 for webcam, or provide a video file path)
video_feed = os.environ.get('DROIDCAM_URL', "http://172.16.2.101:4747/video") # using droidcam on phone, use 0 for webcam on windows
cap = cv2.VideoCapture(video_feed) 

frame_count = 0
last_results = None

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Perform detection every 2 frames
    if frame_count % 2 == 0:
        results = model(frame)
        last_results = results
    
    # Process and display detections
    if last_results is not None:
        for det in last_results.xyxy[0]:  # det: (x1, y1, x2, y2, confidence, class)
            x1, y1, x2, y2, conf, cls = det.tolist()
            class_name = model.names[int(cls)]
            
            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 0), 1)
            
            # Put class name and confidence
            label = f"{class_name} {conf:.2f}"
            cv2.putText(frame, label, (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 1.5)
    
    # Display the frame
    cv2.imshow('Object Detection', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()