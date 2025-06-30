from ultralytics import YOLO
import cv2
import argparse
import os
from tqdm import tqdm

def track_people(input_video_path, output_video_path, model='yolov8x.pt', counter=True, y_divider=0.5, show_video=False, save_video=True):
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
    show_video : bool, optional
        Whether to display the video while processing (default is False).
    save_video : bool, optional
        Whether to save the processed video (default is True).
    
    """
    
    dirname = os.path.dirname(output_video_path)
    if dirname != '':
        os.makedirs(dirname, exist_ok=True)
    model = YOLO(model)
    
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return
    
    # size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    size = (int(width), int(height))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = height/1440
    text_y = int(height/36)   # y coordinate for text placement
    text_thickness = int(height/720)
    
    if not show_video:
        pbar = tqdm(total=total_frames, desc="Processing", unit="frame")        

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model.track(frame, persist=True, verbose=False)[0]

        up_count = 0
        down_count = 0
        
        for i,box in enumerate(results.boxes):
            track_id = box.id
            if track_id is None:
                track_id = i
            else:
                track_id = int(track_id.tolist()[0])
            result = box.xyxy.tolist()[0]

            if box.cls.tolist()[0] != 0:  # Assuming '0' is the class ID for 'person'
                continue
            x1, y1, x2, y2 = result
            center_y = (y1 + y2) // 2

            # Draw bounding box and track ID
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (int(x1), int(y1) - 10), font, 0.5, (0, 255, 0), 2)

            if center_y < height * y_divider:
                up_count += 1
            else:
                down_count += 1

        # Add counter
        if counter:
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
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=False, help="Display the video while processing.")
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=True, help="Save the processed video.")

    args = parser.parse_args()
    
    track_people(args.input_video_path, args.output_video_path, model=args.model, counter=args.counter, show_video=args.show, save_video=args.save)