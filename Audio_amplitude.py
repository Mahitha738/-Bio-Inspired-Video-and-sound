import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
import io
from scipy.io import wavfile

# Function to convert mp3 to wav and return its data
def read_mp3(filename):
    sound = AudioSegment.from_mp3(filename)
    with io.BytesIO() as wav_data:
        sound.export(wav_data, format="wav")
        wav_data.seek(0)
        rate, data = wavfile.read(wav_data)
    return rate, data

# Function to display sound file information
def display_info(data, rate):
    num_samples = len(data)
    length = len(data) / rate
    min_val = np.min(data)
    max_val = np.max(data)
    print("Number of samples:", num_samples)
    print("Length (seconds):", length)
    print("Minimum value:", min_val)
    print("Maximum value:", max_val)

# Read the mp3 files
rate1, data1 = read_mp3('/home/ntu-user/PycharmProjects/Assesment_3/audio.mp3')

# Display information about each sound file
print("Sound 01 Information:")
display_info(data1, rate1)


# Plot the data
plt.figure(figsize=(10, 5))

plt.subplot(2, 1, 1)
plt.plot(data1, color='blue')  # Plot data1 in blue
plt.title('Sound 01')
plt.xlabel('Sample')
plt.ylabel('Amplitude')



plt.tight_layout()
plt.show()

# Store the data in npy files
np.save('results/data1.npy', data1)
np.save('results/rate1.npy', rate1)
