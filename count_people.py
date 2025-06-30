"""Count people that cross a line, specifying the direction (up or down)"""

import cv2
import torch
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model (Make sure you have the yolov8 weights file)
yolo_model = YOLO("yolov8x.pt")  # Change to 'yolov8s.pt' for better accuracy

# Define the counting line (y = constant)
line_position = 700  # Adjust based on your video
count_up = 0
count_down = 0
tracked_ids = set()

# Open video
cap = cv2.VideoCapture("jetson5_video_20180628-191507.mp4")
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter("count.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO tracking
    results = yolo_model.track(frame, persist=True)[0]
    id_name_dict = results.names

    for i, box in enumerate(results.boxes):
        track_id = box.id
        if track_id is None:
            track_id = i
        else:
            track_id = int(track_id.tolist()[0])
        result = box.xyxy.tolist()[0]
        object_cls_id = box.cls.tolist()[0]
        object_cls_name = id_name_dict[object_cls_id]
        
        if object_cls_name != "person":
            continue

        x1, y1, x2, y2 = result
        center_y = (y1 + y2) // 2
        center_x = (x1 + x2) // 2
        
        # Draw a triangle
        # triangle_pts = np.array([
        #     [int(center_x), int(y1) + 30],  # Bottom point
        #     [int(center_x) - 10, int(y1) + 10],  # Top left
        #     [int(center_x) + 10, int(y1) + 10]  # Top right
        # ], np.int32)
        # cv2.fillPoly(frame, [triangle_pts], (0, 255, 0))

        # Draw bounding box and track ID
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if track_id in tracked_ids:
            continue
        
        if center_y < line_position and y2 > line_position:
            count_down += 1
            tracked_ids.add(track_id)
        elif center_y > line_position and y1 < line_position:
            count_up += 1
            tracked_ids.add(track_id)

    cv2.line(frame, (0, line_position), (frame_width, line_position), (0, 0, 255), 2)
    cv2.putText(frame, f"Up: {count_up}  Down: {count_down}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    out.write(frame)
    # cv2.imshow("Tracking", frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

cap.release()
cv2.destroyAllWindows()
