import cv2
import os

# Define the path to the directory containing the frames
frames_dir = "video_frames"
intensities_file = "frame_intensities_rgb_from_folder.txt"

# Ensure the output file directory exists
output_dir = os.path.dirname(intensities_file)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Get a sorted list of frame filenames in the directory
frame_filenames = sorted([f for f in os.listdir(frames_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

# Open the file for writing intensities
with open(intensities_file, 'w') as file:
    for filename in frame_filenames:
        # Construct the full path to the frame image
        frame_path = os.path.join(frames_dir, filename)

        # Read the image
        frame = cv2.imread(frame_path)

        # Check if the image was successfully read
        if frame is None:
            print(f"Error reading frame {filename}.")
            continue

        # Convert frame from BGR to RGB (if your images are already in RGB, you can skip this step)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Reshape the frame to get a 2D array of RGB intensities
        intensities = rgb_frame.reshape((-1, 3))

        # Save the RGB intensities to the file
        for intensity in intensities:
            file.write(','.join(map(str, intensity)) + '\n')
        file.write('\n')  # Separate frames by an empty line for easier reading

    print(f"All frame intensities in RGB format have been saved to {intensities_file}.")
