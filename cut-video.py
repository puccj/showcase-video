import cv2

# Input video file
# input_video_path = "full_output.mp4"  # Replace with your input video path
input_video_path = "output_videos/full_count.mp4"  # Replace with your input video path
output_video_path = "count_1min.mp4"  # Output video file

# Open the video file
cap = cv2.VideoCapture(input_video_path)

# Check if the video was opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Frame width
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Frame height

# Define the duration of the clip (30 seconds)
clip_duration = 60  # in seconds
total_frames = fps * clip_duration  # Total frames in 30 seconds

# Define the codec and create a VideoWriter object to save the output clip
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames*0)  # Skip to the 30-second mark

# Read and write frames for the first 30 seconds
frame_count = 0
while frame_count < total_frames:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Write the frame to the output video
    out.write(frame)
    frame_count += 1

    # Display progress
    print(f"Processed frame {frame_count}/{total_frames}")

# Release resources
cap.release()
out.release()
print(f"30-second clip saved to {output_video_path}")