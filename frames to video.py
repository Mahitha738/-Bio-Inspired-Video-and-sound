import cv2
import os

video_dir = 'video_folder'
reconstructedvideo = 'reconstructed_vid.mp4'

files = sorted([i for i in os.listdir(video_dir) if i.endswith('.png')])

f1 = cv2.imread(os.path.join(video_dir, files[0]))
height, width, _ = f1.shape

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
final_vid = cv2.VideoWriter(reconstructedvideo, fourcc, 10.0, (width, height))

for frame in files:
    frame_path = os.path.join(video_dir, frame)
    frame1 = cv2.imread(frame_path)
    final_vid.write(frame1)

final_vid.release()

# Open the video file once, before the loop
video = cv2.VideoCapture(reconstructedvideo)

for i in range(12):  # Attempt to read the first 10 frames
    ret, frame = video.read()
    if ret:  # If a frame was successfully read
        cv2.imshow('Frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):  # Wait for a short time or until 'q' is pressed
            break
    else:
        break

video.release()  # Release the video file
cv2.destroyAllWindows()  # Close the video window
