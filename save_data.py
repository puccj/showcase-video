from ultralytics import YOLO
import cv2
import argparse
import os
from tqdm import tqdm

def save_data(input_video_path, output_data_path, model='yolov8x.pt', show_video=False, save_video=False):
    """Track people in a video and save their bounding box coordinates and track IDs to a file.
    
    Parameters
    ----------
    input_video_path : str
        Path to the input video file.
    output_data_path : str
        Path to save the output text file with tracking data.
    model : str, optional
        Path to the YOLO model file (default is 'yolov8x.pt').
    show_video : bool, optional
        Whether to display the video while processing (default is False).
    save_video : bool, optional
        Whether to save the processed video (default is True).

    Raises
    ------
    FileNotFoundError
        If the input video file cannot be opened.
    """
    
    dirname = os.path.dirname(output_data_path)
    if dirname != '':
        os.makedirs(dirname, exist_ok=True)
    model = YOLO(model)
    
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {input_video_path}")
    
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    size = (int(width), int(height))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    out = None
    if save_video:
        base, ext = os.path.splitext(output_data_path)
        out = cv2.VideoWriter(base + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = height/1440
    text_y = int(height/36)   # y coordinate for text placement
    text_thickness = int(height/640)
    
    if not show_video:
        pbar = tqdm(total=total_frames, desc="Processing", unit="frame")

    frame_count = 0

    with open(output_data_path, 'w') as f:
        f.write("frame,track_id,xmin,ymin,xmax,ymax,center_x,center_y\n")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            results = model.track(frame, persist=True, verbose=False)[0]

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
                # Draw bounding box and track ID
                if show_video or save_video:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), font, 0.5, (0, 255, 0), 2)
                f.write(f"{frame_count},{track_id},{x1},{y1},{x2},{y2},{center_x},{center_y}\n")
            # Display the frame
            if save_video and out is not None:
                out.write(frame)
            if show_video:
                cv2.imshow("Tracking", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                pbar.update(1)
            frame_count += 1
    cap.release()
    if save_video and out is not None:
        out.release()
    if show_video:
        cv2.destroyAllWindows()
    else:
        pbar.close()

def track_from_data(input_video_path, input_data_path, output_video_path, show_video=False, save_video=True, draw_others=True):
    """Track people in a video using pre-saved tracking data and save the processed video.
    
    Parameters
    ----------
    input_video_path : str
        Path to the input video file.
    input_data_path : str
        Path to the input text file with tracking data.
    output_video_path : str
        Path to save the output video file.
    show_video : bool, optional
        Whether to display the video while processing (default is False).
    save_video : bool, optional
        Whether to save the processed video (default is True).
    draw_others : bool, optional
        Whether to draw bounding boxes for non-highlighted track IDs (default is True).

    Raises
    ------
    FileNotFoundError
        If the input video file or data file cannot be opened.
    """
    
    if not os.path.isfile(input_data_path):
        raise FileNotFoundError(f"Could not find data file: {input_data_path}")
    
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {input_video_path}")
    
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    size = (int(width), int(height))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    out = None
    if save_video:
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = height/1440
    text_y = int(height/36)   # y coordinate for text placement
    text_thickness = int(height/640)
    
    if not show_video:
        pbar = tqdm(total=total_frames, desc="Processing", unit="frame")

    frame_count = 0

    # Load tracking data
    tracking_data = {}
    with open(input_data_path, 'r') as f:
        next(f)  # Skip header line
        for line in f:
            parts = line.strip().split(',')
            frame_num = int(parts[0])
            track_id = int(parts[1])
            if track_id == 51 or track_id == 94 or track_id == 98:
                track_id = 7  # Cheat: remap ID for better visibility
            # elif (track_id == 486 or track_id == 535 or track_id == 587 or track_id == 639 
            #       or track_id == 706 or track_id == 893 or track_id == 1105 or track_id == 1221):
            #     track_id = 366  # Cheat: remap IDs 486 and 487 to 366 for better visibility
            bbox = list(map(int, parts[2:6]))
            center = list(map(int, parts[6:8]))
            x1, y1, x2, y2 = bbox
            bottom = ((x1 + x2) // 2, y2)
            if frame_num not in tracking_data:
                tracking_data[frame_num] = []
            tracking_data[frame_num].append((track_id, bbox, center, bottom))

    # Select some specific track ID to highlight (for demonstration)
    selected_ids = [7]   # Set to None to not highlight any specific ID
    colors = [(0,0,255), (255,0,255), (0,255,255), (255,0,0)]
    trails = {track_id: [] for track_id in selected_ids}  # Trails for selected IDs
    max_trail_length = 10000  # Maximum length of the trail

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count in tracking_data:
            for track_id, bbox, center, bottom in tracking_data[frame_count]:
                x1, y1, x2, y2 = bbox

                if track_id in selected_ids:
                    trails[track_id].append(center) # or bottom
                    if len(trails[track_id]) > max_trail_length:
                        trails[track_id].pop(0)
                    # Draw the trail
                    for j in range(1, len(trails[track_id])):
                        cv2.line(frame, tuple(trails[track_id][j-1]), tuple(trails[track_id][j]), colors[selected_ids.index(track_id)], 2)
                    cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), font, 0.5, colors[selected_ids.index(track_id)], 2)
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), colors[selected_ids.index(track_id)], 2)
                elif draw_others:
                    # Draw bounding box and track ID
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), font, 0.5, (0, 255, 0), 2)
        
        # Display the frame
        if save_video and out is not None:
            out.write(frame)
        if show_video:
            cv2.imshow("Tracking from Data", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            pbar.update(1)
        frame_count += 1
    cap.release()
    if save_video and out is not None:
        out.release()
    if show_video:
        cv2.destroyAllWindows()
    else:
        pbar.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track people in a video and save their bounding box coordinates and track IDs to a file.")
    parser.add_argument("input_video_path", type=str, help="Path to the input video file.")
    parser.add_argument("--output_data_path", type=str, help="Path to save the output text file with tracking data.")
    parser.add_argument("--track", action=argparse.BooleanOptionalAction, default=False, 
                        help="If set, the script will read tracking data from the specified file and overlay it on the video.")
    parser.add_argument("--input_data_path", type=str, default=None, 
                        help="Path to the input text file with tracking data. Required if --track is set.")
    parser.add_argument("--output_video_path", type=str, default=None, 
                        help="Path to save the output video file. Required if --track is set.")
    parser.add_argument("--draw_others", action=argparse.BooleanOptionalAction, default=True, 
                        help="Whether to draw bounding boxes for non-highlighted track IDs when using --track.")
    parser.add_argument("--model", type=str, default="yolov8x.pt", help="Path to the YOLO model file (default is 'yolov8x.pt').")
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=False, help="Display the video while processing.")
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=True, help="Save the processed video.")
    
    args = parser.parse_args()
    if args.track:
        if args.input_data_path is None or args.output_video_path is None:
            parser.error("--input_data_path and --output_video_path are required when --track is set.")
        track_from_data(args.input_video_path, args.input_data_path, args.output_video_path, show_video=args.show, save_video=args.save, draw_others=args.draw_others)
    else:
        save_data(args.input_video_path, args.output_data_path, model=args.model, show_video=args.show, save_video=args.save)
