import cv2

# File path for the video
video_path = 'video.mp4/'

# Open the video capture
cap = cv2.VideoCapture(video_path)
# Check if the video capture is successfully opened
if not cap.isOpened():
    print(f"Error: Unable to open the video at path {video_path}")
else:
    print("Press 'q' to stop the video.")

    # Read and display each frame from the video
    while True:
        ret, frame = cap.read()

        # Break the loop if the video has ended
        if not ret:
            break
        # Display the frame
        cv2.imshow('Video', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Release the video capture object
    cap.release()

    # Close all windows
    cv2.destroyAllWindows()

# separating video from audio
from moviepy.editor import VideoFileClip

# Your video path
video_path = 'video.mp4'

# Load the video file
video_clip = VideoFileClip(video_path)

# Extract audio from the video
audio_clip = video_clip.audio

# Save the audio to a file (e.g., in MP3 format)
audio_path = '/home/ntu-user/PycharmProjects/Assesment_1/audio.mp3'
audio_clip.write_audiofile(audio_path)

# video without audio
video_path = 'video.mp4'

# Load the video file
video_clip = VideoFileClip(video_path)

# Extract audio from the video
audio_clip = video_clip.audio

# Save the audio to a file (e.g., in MP3 format)
audio_path = '/home/ntu-user/PycharmProjects/Assesment_1/audio.mp3'
audio_clip.write_audiofile('audio.mp3', codec='mp3')

video_clip.write_videofile('output_video.mp4', codec='libx264', audio_codec='aac')