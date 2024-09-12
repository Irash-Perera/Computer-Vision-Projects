import os
import cv2
import random
from ultralytics import YOLO
from tracker import Tracker

video_path = os.path.join('.', 'data', 'people.mp4')
video_output_path = os.path.join('.', 'people_output.mp4')
cap = cv2.VideoCapture(video_path)


ret, frame = cap.read()
cap_output = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*'MP4V'),cap.get(cv2.CAP_PROP_FPS) , (frame.shape[1], frame.shape[0])) #cap.get(cv2.CAP_PROP_FPS) is the frame rate of the video

# Load YOLO
yolo = YOLO("yolov8n.pt")

#Initialize Tracker
tracker = Tracker()

#Generate 10 random colors for 10 different tracks
colors =[]
for i in range(10):
    colors.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

while ret:
    detections = yolo(frame)
    for detection in detections:
        resutls =[]
        for det in detection.boxes.data.tolist(): # x1, y1, x2, y2, conf, cls for each person detected
            x1, y1, x2, y2, conf, cls = det
            x1 = int(x1); y1 = int(y1); x2 = int(x2); y2 = int(y2); cls = int(cls)
            if int(cls) == 0 and conf >= 0.3: # if the detected object is a person and the confidence is greater than 0.5 
                resutls.append([x1, y1, x2, y2, conf])
           
        #Update tracker for each frame with the detections 
        tracker.update(frame, resutls)
        
        #Draw bounding boxes for each updated track
        for track in tracker.tracks:
            bbox = track.bbox
            x1, y1, x2, y2 = bbox
            track_id = track.track_id
            cv2.rectangle(frame, (int(x1),int(y1)), (int(x2), int(y2)), colors[track_id%len(colors)], 3)    
                    
    cap_output.write(frame)
    ret, frame = cap.read()
    if not ret:
        break
    # if cv2.waitKey(25) & 0xFF ==27:
    #     break
cap.release()
cap_output.release()
cv2.destroyAllWindows()