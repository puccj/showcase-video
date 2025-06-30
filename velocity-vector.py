"""Track people in a video and show their velocity vectors."""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# Load YOLO model
model = YOLO("yolov8x.pt")  # Use YOLOv8 nano model

# Open video file
video_path = "jetson5_video_20180628-191507.mp4"  # Replace with your video path
cap = cv2.VideoCapture(video_path)

# Store previous positions and timestamps for each tracked object
track_history = defaultdict(lambda: [])

# Function to calculate velocity
def calculate_velocity(prev_position, curr_position, time_elapsed):
    if time_elapsed == 0:
        return 0
    
    vx = (curr_position[0] - prev_position[0]) / time_elapsed
    vy = (curr_position[1] - prev_position[1]) / time_elapsed
    mod = np.sqrt(vx ** 2 + vy ** 2)
    return vx, vy, mod

fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))

# Process video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO tracking
    results = model.track(frame, persist=True, classes=0)  # Track only people (class 0)

    # Get tracking results
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu()  # Get bounding boxes in (x, y, w, h) format
        track_ids = results[0].boxes.id.int().cpu().tolist()  # Get track IDs

        # Annotate frame with bounding boxes and velocity vectors
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            center = (int(x), int(y))  # Center of the bounding box

            # Store current position and timestamp
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # Current time in seconds
            track_history[track_id].append((center, current_time))

            # Keep only the last 10 positions for velocity calculation
            if len(track_history[track_id]) > 10:
                track_history[track_id].pop(0)

            # Calculate velocity if there are at least 2 positions
            if len(track_history[track_id]) >= 2:
                prev_position, prev_time = track_history[track_id][-2]
                curr_position, curr_time = track_history[track_id][-1]
                time_elapsed = curr_time - prev_time
                vx,vy,velocity = calculate_velocity(prev_position, curr_position, time_elapsed)
                
                # TODO filter out the velocity vector that is too large

                # Draw velocity vector
                scale = 0.5  # Scale factor for better visibility
                # direction = (curr_position[0] - prev_position[0], curr_position[1] - prev_position[1])
                # end_point = (int(curr_position[0] + direction[0] * scale), int(curr_position[1] + direction[1] * scale))
                end_point = (int(curr_position[0] + vx * scale), int(curr_position[1] + vy * scale))

                cv2.arrowedLine(frame, curr_position, end_point, (0, 0, 255), 2)  # Green arrow

                # Display velocity near the bounding box
                cv2.putText(frame, f"V: {velocity:.2f} px/s", (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Draw bounding box
            # cv2.rectangle(frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (255, 0, 0), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

    # Display the frame
    cv2.imshow("YOLO Tracking with Velocity", frame)
    writer.write(frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
writer.release()
cv2.destroyAllWindows()