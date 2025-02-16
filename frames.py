# splitting videos into frame with size 200x174
import cv2
import os
import matplotlib.pyplot as plt
import nest

# Define paths and files
video_file = "output_video.mp4"
results_dir = "results"
frames_dir = "video_frames"
target_size = (200, 174)  # width, height

# Ensure the frames directory exists
if not os.path.exists(frames_dir):
    os.makedirs(frames_dir)

# Open the video file
cap = cv2.VideoCapture(video_file)

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break  # Stop if there are no frames left to read

    # Resize the frame to the target size
    resized_frame = cv2.resize(frame, target_size)

    # Construct the path for saving the frame with a zero-padded frame index
    frame_path = os.path.join(frames_dir, f"frame_{frame_count:04d}.jpg")

    # Save the resized frame
    cv2.imwrite(frame_path, resized_frame)

    frame_count += 1

cap.release()
print(f"All frames have been saved to {frames_dir}.")