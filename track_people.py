from ultralytics import YOLO
import cv2
import argparse
import os
from tqdm import tqdm

def track_people(input_video_path, output_video_path, model='yolov8x.pt', counter=True, y_divider=0.5, trail_seconds=0, show_video=False, save_video=True):
    """Track people in a video and optionally count them, distinguishing between those in the upper and lower part of the frame.
    
    Parameters
    ----------
    input_video_path : str
        Path to the input video file.
    output_video_path : str
        Path to save the output video file.
    model : str, optional
        Path to the YOLO model file (default is 'yolov8x.pt').
    counter : bool, optional
        Whether to add a counter to the video (default is True).
    y_divider : float, optional
        Fraction of the frame height to use as the divider between upper and lower parts of the frame.
        This value should be between 0 and 1, where 0 means the top of the frame and 1 means the bottom of the frame.
        For example, 0.5 (default) means the middle of the frame.
        If set to 0, the total count will be displayed without distinguishing between upper and lower parts.
    trail_seconds : float, optional
        If set to positive, it will draw trails for tracked objects. The value specifies the number of seconds to keep the trail.
        If set to 0, no trails will be drawn (default is 0).
        If the trail is enabled, each tracked object will have a unique color.
    show_video : bool, optional
        Whether to display the video while processing (default is False).
    save_video : bool, optional
        Whether to save the processed video (default is True).

    Raises
    ------
    FileNotFoundError
        If the input video file cannot be opened.
    ValueError
        If `y_divider` is not between 0 and 1.
    ValueError
        If `trail` is negative.
    """


    
    dirname = os.path.dirname(output_video_path)
    if dirname != '':
        os.makedirs(dirname, exist_ok=True)
    model = YOLO(model)
    
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {input_video_path}")
    
    if y_divider < 0 or y_divider > 1:
        raise ValueError("y_divider must be between 0 and 1.")
    
    if trail_seconds < 0:
        raise ValueError("trail must be a non-negative integer.")
    
    # size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    size = (int(width), int(height))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    
    trail = int(trail_seconds * fps) if trail_seconds > 0 else 0

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = height/1440
    text_y = int(height/36)   # y coordinate for text placement
    text_thickness = int(height/640)
    
    if not show_video:
        pbar = tqdm(total=total_frames, desc="Processing", unit="frame")

    trails = {}  # Dictionary to hold trails for each track ID
    frame_count = 0
    color = (0, 255, 0)  # Default color if trail is not enabled (green)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model.track(frame, persist=True, verbose=False)[0]

        up_count = 0
        down_count = 0
        
        if trail > 0:
            seen_ids = set()  # To keep track of seen track IDs
            index = frame_count % trail

        for i,box in enumerate(results.boxes):
            if box.cls.tolist()[0] != 0:  # Assuming '0' is the class ID for 'person'
                continue
            
            track_id = box.id
            if track_id is None:
                track_id = i
            else:
                track_id = int(track_id.tolist()[0])
            result = box.xyxy.tolist()[0]

            x1, y1, x2, y2 = map(int, result)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            if trail > 0:
                # Change color to a unique one for each track ID
                color = (int(track_id * 37) % 255, int(track_id * 73) % 255, int(track_id * 97) % 255)
                
                # Initialize the trail for this track ID if it doesn't exist, creating an empty list of size `trail`
                if track_id not in trails:
                    trails[track_id] = [None] * trail

                # Add the current position to the trail
                seen_ids.add(track_id)
                trails[track_id][index] = (center_x, y2)
                
                # Draw the trail
                for j in range(1, len(trails[track_id])):
                    if j == index+1:
                        continue
                    pt1 = trails[track_id][j - 1]
                    if pt1 is None:
                        continue
                    pt2 = trails[track_id][j]
                    if pt2 is None:
                        continue
                
                    cv2.line(frame, pt1, pt2, color, 2)

            # Draw bounding box and track ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), font, 0.5, color, 2)

            if center_y < height * y_divider:
                up_count += 1
            else:
                down_count += 1

        if trail > 0:
            for track_id in list(trails.keys()):
                if track_id not in seen_ids:
                    # If the track ID was not seen in this frame, remove its trail
                    trails[track_id][index] = None

        # Add counter
        if counter:
            if y_divider == 0:
                cv2.putText(frame, f"Total: {up_count + down_count} people", (10, text_y), font, font_size, (0, 255, 0), text_thickness)
            else:
                cv2.putText(frame, f"Up floor: {up_count} people", (10, text_y), font, font_size, (0, 255, 0), text_thickness)
                cv2.putText(frame, f"Down floor: {down_count} people", (10, int(text_y + font_size*40)), font, font_size, (0, 255, 0), text_thickness)
                cv2.putText(frame, f"Total: {up_count + down_count} people", (10, text_y + int(2*font_size*40)), font, font_size, (0, 255, 0), text_thickness)

        # Display the frame
        if save_video:
            out.write(frame)
        if show_video:
            cv2.imshow("Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            pbar.update(1)

        frame_count += 1
        
    cap.release()
    out.release()
    if show_video:
        cv2.destroyAllWindows()
    else:
        pbar.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track people in a video using YOLOv8.")
    parser.add_argument("input_video_path", type=str, help="Path to the input video file.")
    parser.add_argument("output_video_path", type=str, help="Path to save the output video file.")
    parser.add_argument("--model", type=str, default="yolov8x.pt", help="Path to the YOLO model file.")
    parser.add_argument("--counter", action=argparse.BooleanOptionalAction, default=True, help="Add a counter to the video.")
    parser.add_argument("--y_divider", type=float, default=0.5, help="Fraction of the frame height to use as the divider between upper and lower parts of the frame (0 to 1).")
    parser.add_argument("--trail_seconds", type=int, default=0, help= "Number of seconds to keep the trail for tracked objects (default is 0, meaning no trail).")
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=False, help="Display the video while processing.")
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=True, help="Save the processed video.")

    args = parser.parse_args()
    
    track_people(args.input_video_path, args.output_video_path, args.model, args.counter, args.y_divider, args.trail_seconds, args.show, args.save)