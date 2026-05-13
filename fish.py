"""Detect fish in a tank using background subtraction. 
Multiple ROI can be used, and fish are tracked separately in each ROI. 
The tracking is done by linking detections across frames using a simple nearest neighbor approach. 
Collisions are detected by checking for missing detections, and trajectories are re-started at collision and
later merged based on spatial proximity before and after the collision.
"""

import cv2
import numpy as np
import random
from tqdm import tqdm
import csv

def distance(p1, p2):
    return ((p1["x"] - p2["x"])**2 + (p1["y"] - p2["y"])**2)**0.5

def inside(pt, roi):
    if roi is None:
        return False
    x, y = pt
    x1, y1, x2, y2 = roi
    return x1 <= x <= x2 and y1 <= y <= y2

def get_background(video_path, num_frames=50, save_path="bg.png"):
    """Get the background image from the video using `num_frames` random frames"""

    print(f"Extracting background from video: {video_path} using {num_frames} random frames")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    # sort frames for faster seeking
    frame_indices = sorted(random.sample(range(total_frames), min(num_frames, total_frames)))

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()

    bg = np.median(frames, axis=0).astype(np.uint8)
    print(f"Background extraction done.")

    cv2.imwrite(save_path, bg)
    print(f"Background saved to: {save_path}")

    return bg

def detect(video_path, background, output_path="Default", show_video=False, out_video="Default", thresh_value=55, min_area=40, start_second=0, end_second=None, rois=None):
    """Detect fish in video by identifying them and following their movement. Save the detections to a CSV file.
    Optionally save a video with the detections drawn on it.

    Parameters
    ----------
    video_path: str
        Path to the input video file.
    background: np.ndarray
        Background image to use for subtraction.
    output_path: str or None
        Path to save the output CSV file with detections. If "Default", saves in the same location with '_detections' suffix. If None, no file is saved.
    show_video: bool
        Whether to show the video with detections in real-time.
    out_video: str or None
        Path to save the output video with detections drawn. If "Default", saves in the same location with '_detection' suffix. If None, no video is saved.
    thresh_value: int
        Threshold value for background subtraction. Default is 55.
    min_area: int
        Minimum area of contours to be considered as fish. Default is 40.
    start_second: int
        Start second for tracking. Default is 0.
    end_second: int or None
        End second for tracking. Default is None (track until the end of the video).
    rois: list of tuples or None
        List of regions of interest (ROIs) to consider for detection. Each ROI is a tuple (x1, y1, x2, y2). If None, all detections are accepted.

    Returns
    -------
    detections: dict
        Dictionary containing the detected fish positions for each frame.

    """

    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))


    if output_path == "Default":
        output_path = video_path.rsplit('.', 1)[0] + "_detections.csv"
    
    if out_video == "Default":
        out_video = video_path.rsplit('.', 1)[0] + "_detection.mp4"
    if out_video is not None:
        out = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    
    detections = {}

    start_frame = int(start_second * fps)
    end_frame = int(end_second * fps) if end_second is not None else cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    if show_video:
        window_name = "Fish Detection"
        cv2.namedWindow(window_name)

        def _noop(_value):
            pass

        max_min_area = min(1000, (size[0] * size[1]) // 4)
        cv2.createTrackbar("Threshold", window_name, thresh_value, 255, _noop)
        cv2.createTrackbar("Min Area", window_name, min_area, max_min_area, _noop)

    pbar = tqdm(total=end_frame - start_frame, desc="Detecting fish")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if cap.get(cv2.CAP_PROP_POS_FRAMES) > end_frame:
            break
        
        if show_video:
            thresh_value = cv2.getTrackbarPos("Threshold", window_name)
            min_area = cv2.getTrackbarPos("Min Area", window_name)

        current_rois = rois

        # current frame index (int) to record detections per-frame
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        fg_mask = cv2.absdiff(frame, background)
        gray = cv2.cvtColor(fg_mask, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, thresh_value, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cv2.contourArea(cnt) > min_area:  # filter small contours
                # find centroid of the contour
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                center = (cX, cY)

                # Find which ROI(s) contain this detection
                roi_index = None
                if not current_rois:
                    # No ROIs: accept all detections
                    keep = True
                else:
                    # Find first ROI that contains this detection
                    for idx, roi in enumerate(current_rois):
                        if inside(center, roi):
                            roi_index = idx
                            keep = True
                            break
                    else:
                        keep = False

                if not keep:
                    continue

                det_dict = {"x": cX, "y": cY, "roi_index": roi_index}
                detections.setdefault(current_frame, []).append(det_dict)
                if out_video is not None:
                    cv2.circle(frame, center, 5, (0, 0, 255), -1)

        if out_video is not None:
            out.write(frame)

        if show_video:
            cv2.drawContours(frame, contours, -1, (255, 0, 0), 1)
            # draw ROIs
            if rois is not None:
                for idx, roi in enumerate(rois):
                    if roi is None:
                        continue
                    x1, y1, x2, y2 = roi
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Draw ROI index at top-left corner
                    cv2.putText(frame, str(idx), (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        pbar.update(1)

    pbar.close()
    cap.release()
    if show_video:
        cv2.destroyAllWindows()
    total_detections = sum(len(v) for v in detections.values())
    print(f"Tracking complete. Total detections: {total_detections}")
    
    if out_video is not None:
        out.release()
        print(f"Output video saved to: {out_video}")
    
    # Save detections to CSV (frame,x,y,roi_index)
    if output_path is not None:
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "x", "y", "roi_index"])
            for frame_idx in sorted(detections.keys()):
                for det in detections[frame_idx]:
                    writer.writerow([frame_idx, det["x"], det["y"], det["roi_index"]])
        print(f"Detections saved to: {output_path}")

    return detections

def track(detections, output_path="Default"):
    """Track fish across frames using the detections, considering each roi separately.    

    Parameters
    ----------
    detections: dict or str
        Detections to use for tracking OR path to CSV file containing detections.
        If dict, it should be in the format {frame: [{"x": x, "y": y, "roi_index": roi_index}, ...]}.
        If str, it should be the path to a CSV file with columns: frame,x,y,roi_index
    output_path: str or None
        Path to save the output CSV file with tracks. 
        If "Default", saves in the same location with '_tracks' suffix. If None, no file is saved.
    
    Returns
    -------
    all_tracks: dict
        Dictionary containing the tracked fish trajectories for each ROI.
        Format: {roi_index: {fish_id: [(frame, x, y), ...]}}

    Notes
    -----
    The tracking algorithm is a simple nearest neighbor approach that links detections across frames.
    Collisions are detected by checking for two detections becoming one. 
    In this case the previous track is ended and a new track is started when the blob separates again. 
    Separate tracks are later merged to create a complete trajectory for each fish. The merge can be based on:
      1. the positions of the fish before and after the collision
    [ 2 ] the average size of the blob of the trajectories before and after the collision (if available) 
    [ 3 ] the velocity of the fish before and after the collision, 
    [ 4 ] the color of the blob before and after the collision (if available) 
    [ 5 ] the shape of the blob before and after the collision (if available) 
    Only the first one is implemented for now.
    """

    if isinstance(detections, str):
        # Read detections from CSV
        detections_by_roi_frame = {}  # {roi_index: {frame: [{"x": x, "y": y}, ...]}}
        
        with open(detections, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                frame = int(row["frame"])
                x = int(row["x"])
                y = int(row["y"])
                roi_index = int(row["roi_index"]) if row["roi_index"] != "None" else None
                
                if roi_index not in detections_by_roi_frame:
                    detections_by_roi_frame[roi_index] = {}
                if frame not in detections_by_roi_frame[roi_index]:
                    detections_by_roi_frame[roi_index][frame] = []
                
                detections_by_roi_frame[roi_index][frame].append({"x": x, "y": y})
    else:
        # Organize detections by ROI and frame
        detections_by_roi_frame = {}  # {roi_index: {frame: [{"x": x, "y": y}, ...]}}
        for frame, dets in detections.items():
            for det in dets:
                roi_index = det["roi_index"]
                if roi_index not in detections_by_roi_frame:
                    detections_by_roi_frame[roi_index] = {}
                if frame not in detections_by_roi_frame[roi_index]:
                    detections_by_roi_frame[roi_index][frame] = []
                detections_by_roi_frame[roi_index][frame].append({"x": det["x"], "y": det["y"]})
    
    # Track fish for each ROI separately
    all_tracks = {}  # {roi_index: {fish_id: [(frame, x, y), ...]}}
    
    for roi_index in sorted(detections_by_roi_frame.keys()):
        frames_data = detections_by_roi_frame[roi_index]
        sorted_frames = sorted(frames_data.keys())
        
        tracks_in_roi = {}  # {fish_id: [(frame, x, y), ...]}
        next_fish_id = 0
        frame_to_fish_id = {}  # {frame: {detection_idx: fish_id}}
        
        # Track max distance for nearest neighbor (adaptive based on ROI size)
        max_distance = 100  # pixels
        
        for i, frame in enumerate(sorted_frames):
            current_detections = frames_data[frame]
            frame_to_fish_id[frame] = {}
            
            # For first frame, assign new IDs to all detections
            if i == 0:
                for det_idx, det in enumerate(current_detections):
                    fish_id = next_fish_id
                    tracks_in_roi[fish_id] = [(frame, det["x"], det["y"])]
                    frame_to_fish_id[frame][det_idx] = fish_id
                    next_fish_id += 1
            else:
                prev_frame = sorted_frames[i - 1]
                prev_detections = frames_data[prev_frame]
                used_prev_ids = set()
                assigned_current = set()
                # If the number of detections changed between previous and current frame,
                # treat it as a merge (prev > curr) or a split (prev < curr).
                # In both cases, end previous tracks and assign new IDs to all current detections.
                if len(prev_detections) != len(current_detections):
                    # Collision or split: end previous tracks and assign new IDs to all current detections
                    # (no action needed for previous tracks; they end simply by not being extended)
                    # Assign new IDs to all current detections
                    for det_idx, det in enumerate(current_detections):
                        fish_id = next_fish_id
                        tracks_in_roi[fish_id] = [(frame, det["x"], det["y"])]
                        frame_to_fish_id[frame][det_idx] = fish_id
                        next_fish_id += 1
                        assigned_current.add(det_idx)
                else:
                    # Nearest neighbor matching
                    for det_idx, det in enumerate(current_detections):
                        best_prev_idx = None
                        best_distance = max_distance
                        
                        for prev_idx, prev_det in enumerate(prev_detections):
                            if prev_idx in used_prev_ids:
                                continue
                            
                            d = distance(det, prev_det)
                            if d < best_distance:
                                best_distance = d
                                best_prev_idx = prev_idx
                        
                        if best_prev_idx is not None:
                            # Found a match: continue the track
                            fish_id = frame_to_fish_id[prev_frame][best_prev_idx]
                            tracks_in_roi[fish_id].append((frame, det["x"], det["y"]))
                            frame_to_fish_id[frame][det_idx] = fish_id
                            used_prev_ids.add(best_prev_idx)
                            assigned_current.add(det_idx)
                        else:
                            # No match: new fish ID (spawn)
                            fish_id = next_fish_id
                            tracks_in_roi[fish_id] = [(frame, det["x"], det["y"])]
                            frame_to_fish_id[frame][det_idx] = fish_id
                            next_fish_id += 1
                            assigned_current.add(det_idx)

        # Post-process: merge fragmented trajectories when a track ends and another starts shortly after.
        # Strategy:
        # - For each pair (A,B) where A.end_frame < B.start_frame and gap <= merge_gap_max,
        #   compute spatial distance between A.end_pos and B.start_pos.
        # - Greedily merge the pair with smallest distance if distance <= merge_dist_max.
        # - When merging A and B, drop any temporary tracks that exist entirely in the gap
        #   (these represent the single-blob merged frames) to "discard the few frames where
        #   the two trajectories had become one".
        if tracks_in_roi:
            def _track_stats(points):
                start_f, sx, sy = points[0]
                end_f, ex, ey = points[-1]
                return {
                    "start_frame": start_f,
                    "end_frame": end_f,
                    "start_pos": (sx, sy),
                    "end_pos": (ex, ey),
                }

            merge_gap_max = 30
            merge_dist_max = 200

            while True:
                track_ids = sorted(tracks_in_roi.keys())
                if len(track_ids) < 2:
                    break

                stats = {tid: _track_stats(tracks_in_roi[tid]) for tid in track_ids}
                best_pair = None
                best_dist = None

                for a in track_ids:
                    for b in track_ids:
                        if a == b:
                            continue
                        a_end = stats[a]["end_frame"]
                        b_start = stats[b]["start_frame"]
                        if b_start <= a_end:
                            continue
                        gap = b_start - a_end
                        if gap > merge_gap_max:
                            continue

                        d = distance({"x": stats[a]["end_pos"][0], "y": stats[a]["end_pos"][1]},
                                     {"x": stats[b]["start_pos"][0], "y": stats[b]["start_pos"][1]})
                        if d > merge_dist_max:
                            continue

                        if best_dist is None or d < best_dist:
                            best_dist = d
                            best_pair = (a, b)

                if best_pair is None:
                    break

                parent_id, child_id = best_pair
                parent_stats = stats[parent_id]
                child_stats = stats[child_id]

                # Merge: append child's points to parent and sort
                tracks_in_roi[parent_id].extend(tracks_in_roi[child_id])
                tracks_in_roi[parent_id].sort(key=lambda p: p[0])

                # Remove temporary tracks whose frames lie entirely within the gap
                gap_start = parent_stats["end_frame"] + 1
                gap_end = child_stats["start_frame"] - 1
                to_delete = []
                for tid in list(tracks_in_roi.keys()):
                    if tid in (parent_id, child_id):
                        continue
                    t_start = tracks_in_roi[tid][0][0]
                    t_end = tracks_in_roi[tid][-1][0]
                    if t_start >= gap_start and t_end <= gap_end:
                        to_delete.append(tid)
                for tid in to_delete:
                    del tracks_in_roi[tid]

                # Remove the child track after merging
                del tracks_in_roi[child_id]
        
        all_tracks[roi_index] = tracks_in_roi
    
    # Save tracks to CSV
    if output_path == "Default":
        if isinstance(detections, str):
            output_path = detections.rsplit('.', 1)[0] + "_tracks.csv"
        else:
            output_path = "tracks.csv"

    if output_path is not None:
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["roi_index", "fish_id", "frame", "x", "y"])
            
            for roi_index in sorted(all_tracks.keys()):
                tracks = all_tracks[roi_index]
                for fish_id in sorted(tracks.keys()):
                    for frame, x, y in tracks[fish_id]:
                        writer.writerow([roi_index, fish_id, frame, x, y])
    
    print(f"Tracking complete. Tracks saved to: {output_path}")
    
    # Print summary
    for roi_index in sorted(all_tracks.keys()):
        num_fish = len(all_tracks[roi_index])
        print(f"  ROI {roi_index}: {num_fish} unique fish tracked")

    return all_tracks

def save_tracking_video(video_path, all_tracks, output_video_path="Default", show_video=False, trajectory_frames=60, start_second=0, end_second=None):
    """Draw trajectories on video and save.
    
    Parameters
    ----------
    video_path: str
        Path to the input video file.
    all_tracks: dict or str
        Tracks to visualize OR path to CSV file containing tracks.
        If dict, it should be in the format {roi_index: {fish_id: [(frame, x, y), ...]}}.
        If str, it should be the path to a CSV file with columns: roi_index,fish_id,frame,x,y
    output_video_path: str or None
        Path to save the output video with trajectories drawn. 
        If "Default", saves in the same location with '_tracking' suffix. If None, no video is saved.
    show_video: bool
        Whether to show the video with trajectories in real-time.
    trajectory_frames: int
        Number of previous frames to show in trajectory. If -1 shows all previous frames. Default is 60.
    start_second: int
        Start second for tracking. Default is 0.
    end_second: int or None
        End second for tracking. Default is None (track until the end of the video).
    """

    if isinstance(all_tracks, str):
        # Read tracks from CSV
        all_tracks_dict = {}  # {roi_index: {fish_id: [(frame, x, y), ...]}}
        
        with open(all_tracks, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                roi_index = int(row["roi_index"])
                fish_id = int(row["fish_id"])
                frame = int(row["frame"])
                x = int(row["x"])
                y = int(row["y"])
                
                if roi_index not in all_tracks_dict:
                    all_tracks_dict[roi_index] = {}
                if fish_id not in all_tracks_dict[roi_index]:
                    all_tracks_dict[roi_index][fish_id] = []
                
                all_tracks_dict[roi_index][fish_id].append((frame, x, y))
        
        all_tracks = all_tracks_dict


    if output_video_path == "Default":
        output_video_path = video_path.rsplit('.', 1)[0] + "_tracking.mp4"

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    
    # Generate colors for different fish IDs
    colors = [
        (255, 0, 0),    # Blue
        (0, 255, 0),    # Green
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 128, 0),  # Olive
        (128, 0, 128),  # Purple
        (0, 128, 128),  # Teal
        (128, 128, 128) # Gray
    ]
    
    def get_color(fish_id):
        return colors[fish_id % len(colors)]

    start_frame = int(start_second * fps)
    end_frame = int(end_second * fps) if end_second is not None else cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    pbar = tqdm(total=end_frame - start_frame, desc="Saving tracking video")
    
    frame_idx = start_frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if cap.get(cv2.CAP_PROP_POS_FRAMES) > end_frame:
            break
        
        # Draw trajectories for all tracks visible in this frame
        for roi_index in all_tracks:
            if roi_index is None or roi_index < 0 or roi_index >= 26:
                roi_letter = "F"
            else:
                roi_letter = chr(65 + roi_index)
            for fish_id in all_tracks[roi_index]:
                trajectory = all_tracks[roi_index][fish_id]

                first_frame = trajectory[0][0]
                last_frame = trajectory[-1][0]
                # Do not render tracks before birth or after death.
                if frame_idx < first_frame or frame_idx > last_frame:
                    continue
                
                # Find points within trajectory_frames before current frame
                if trajectory_frames == -1:
                    relevant_points = [(x, y) for f, x, y in trajectory if f <= frame_idx]
                else:
                    relevant_points = [(x, y) for f, x, y in trajectory if f >= frame_idx - trajectory_frames and f <= frame_idx]
                
                if len(relevant_points) > 0:
                    # Draw trajectory line
                    for i in range(len(relevant_points) - 1):
                        pt1 = tuple(map(int, relevant_points[i]))
                        pt2 = tuple(map(int, relevant_points[i + 1]))
                        cv2.line(frame, pt1, pt2, get_color(fish_id), 2)
                    
                    # Draw current point as circle
                    if len(relevant_points) > 0:
                        current_pt = tuple(map(int, relevant_points[-1]))
                        cv2.circle(frame, current_pt, 5, get_color(fish_id), -1)
                        # Draw fish ID near the current position
                        cv2.putText(frame, f"{roi_letter}{fish_id}", (current_pt[0] + 10, current_pt[1] - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, get_color(fish_id), 1)
        
        if show_video:
            cv2.imshow("Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        out.write(frame)
        pbar.update(1)
        frame_idx += 1
    
    pbar.close()
    cap.release()
    out.release()
    if show_video:
        cv2.destroyAllWindows()
    print(f"Tracking video saved to: {output_video_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Track fish in a video using background subtraction.")
    parser.add_argument("video_path", help="Path to the input video file.")
    parser.add_argument("-b", "--background_path", default=None, help="Path to a pre-computed background image. If not provided, the background will be computed from the video.")
    parser.add_argument("-o", "--out_video", default="Default", help="Path to save the output video with trajectories drawn. Default is to save in the same location with '_detection' suffix.")
    parser.add_argument("-t", "--thresh_value", type=int, default=55, help="Threshold value for background subtraction. Default is 55.")
    parser.add_argument("-a", "--min_area", type=int, default=40, help="Minimum area of contours to be considered as fish. Default is 40.")
    parser.add_argument("-s", "--start_second", type=int, default=0, help="Start second for tracking. Default is 0.")
    parser.add_argument("-e", "--end_second", type=int, default=None, help="End second for tracking. Default is None (track until the end of the video).")
    parser.add_argument("--detect", action="store_true", help="Run detection and save detections to CSV. Required for tracking.")
    parser.add_argument("--track", action="store_true", help="Run tracking using detections from CSV. Requires --detect to be run first.")
    parser.add_argument("--detections_csv", type=str, default=None, help="Path to pre-computed detections CSV. If not provided, detections will be computed from the video.")
    parser.add_argument("--show", action="store_true", help="Show the video while tracking and enable live sliders for threshold and minimum area.")
    parser.add_argument("--trajectory-frames", type=int, default=60, help="Number of previous frames to show in trajectory. Default is 60.")

    rois = [
        (0, 119, 898, 528),  # Example ROI (x1, y1, x2, y2)
        (951, 124, 1919, 531),
        (0, 578, 892, 986),
        (941, 580, 1919, 985)
    ]

    args = parser.parse_args()

    if args.background_path is not None:
        background = cv2.imread(args.background_path)
        if background is None:
            print(f"Could not read background image: {args.background_path}")
            exit(1)
    else:
        background = get_background(args.video_path)
    
    if background is None:
        print("Error: Background image could not be obtained. Exiting.")
        exit(1)
    
    if args.detect:
        detect(args.video_path, background, show_video=args.show, out_video=args.out_video, thresh_value=args.thresh_value, min_area=args.min_area, start_second=args.start_second, end_second=args.end_second, rois=rois)
    
    # If track video is requested, generate tracking visualization
    if args.track:
        if args.detections_csv is None:
            detections = detect(args.video_path, background, show_video=False, out_video=None, thresh_value=args.thresh_value, min_area=args.min_area, start_second=args.start_second, end_second=args.end_second, rois=rois)
        else:
            detections = args.detections_csv
        
        all_tracks = track(detections, output_path="Default")
        save_tracking_video(args.video_path, all_tracks, output_video_path=args.out_video, show_video=args.show, trajectory_frames=args.trajectory_frames, start_second=args.start_second, end_second=args.end_second)