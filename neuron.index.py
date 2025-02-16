import cv2
import matplotlib.pyplot as plt
import nest
import numpy as np
import os
import h5py
from tqdm import tqdm

results_dir = "encodedFiles/"
images_folder = 'video_frames/'
image_extension = '.jpg'

# Function to convert pixel intensity to current for each color channel
def pixel_intensity_to_current(intensity, offset=380):
    return intensity + offset

# Function to resize image
def resize_image(image, target_size=(300, 200)):
    if target_size is None:
        return image
    else:
        return cv2.resize(image, target_size)

# Function to simulate raster plot for each image
def simulate_raster_plot(image_file, current_funcs, sim_time=50.0):
    # Clear console
    os.system('clear')

    # Read the image
    print("Reading image {}...".format(image_file))
    image = cv2.imread(os.path.join(images_folder, image_file + image_extension))
    # Resize the image
    resized_image = resize_image(image)
    # Get image dimensions
    height, width, _ = resized_image.shape
    print("Image dimensions: Height: {}, Width: {}".format(height, width))

    # Initialize NEST kernel
    nest.ResetKernel()
    nest.set_verbosity(20)  # Set NEST verbosity level to 20
    nest.SetKernelStatus({'print_time': False})

    # Create layers for Blue, Green, and Red channels
    layers = []
    spikerecorders = []
    for i, color in enumerate(['Blue', 'Green', 'Red']):
        # Create layer with iaf_psc_alpha neurons
        layers.append(nest.Create('iaf_psc_alpha', width * height))
        # Connect each layer to a spike recorder
        spikerecorders.append(nest.Create("spike_recorder"))

        # Progress bar for setting currents
        progress_bar = tqdm(total=height * width, desc="Setting currents for {} channel".format(color), position=0,
                            leave=True)

        # Create spike generators for each neuron and inject analog values
        for row in range(height):
            for col in range(width):
                # Calculate the current based on pixel intensity for the corresponding color channel
                intensity = resized_image[row, col, i]
                current = current_funcs[i](intensity)

                # Set current for each neuron
                neuron_index = row * width + col
                nest.SetStatus(layers[i][neuron_index], {"I_e": current})

                # Update progress bar
                progress_bar.update(1)

        nest.Connect(layers[i], spikerecorders[i])

    # Simulate
    print("Simulating for", image_file)
    nest.Simulate(sim_time)
    print("Simulation completed for", image_file)

    # Save spike events and senders in HDF5 format
    os.makedirs(results_dir, exist_ok=True)
    with h5py.File(os.path.join(results_dir, image_file + "_spikes.h5"), "w") as file:
        for i, color in enumerate(['Blue', 'Green', 'Red']):
            events = spikerecorders[i].get("events")
            senders = events["senders"]
            times = events["times"]
            grp = file.create_group(color)
            grp.create_dataset("senders", data=senders)
            grp.create_dataset("times", data=times)
            grp.attrs["image_filename"] = image_file
            grp.attrs["image_dimensions"] = (height, width)
            grp.attrs["simulation_time"] = sim_time

            # Save times array to NumPy file
            np.save(os.path.join(results_dir, image_file + "_"+ color + "_times.npy"), times)

    # Plot raster plot for each color channel
    plt.figure(figsize=(15, 5))
    for i, color in enumerate(['Blue', 'Green', 'Red']):
        plt.subplot(1, 3, i + 1)
        plt.title('{} Channel - {}'.format(color, image_file))
        plt.xlabel('Time (ms)')
        plt.ylabel('Neuron Index')
        plt.grid()
        ts = spikerecorders[i].get("events")["times"]
        if i == 0:
            senders = spikerecorders[i].get("events")["senders"] - 1  # normalize values between 0 and 300*200-1
        elif i == 1:
            senders = spikerecorders[i].get("events")["senders"] - 300 * 200 - 2  # normalize values between 0 and 300*200-1
        else:
            senders = spikerecorders[i].get("events")["senders"] - 300 * 200 * 2 - 3  # normalize values between 0 and 300*200-1
        np.save(os.path.join(results_dir, image_file + "_"+ color + "_senders.npy"), senders)
        plt.vlines(ts, senders, senders + 1, color=color.lower(), linewidths=0.5)
    plt.tight_layout()
    #plt.show()


# Specify the start and stop indices for the image range
start_index = 1
stop_index = 15

# List of image file names within the specified range
image_files = ['frame_{:04d}'.format(i) for i in range(start_index, stop_index + 1)]

# Simulate raster plot for each image within the specified range
for image_file in image_files:
    simulate_raster_plot(image_file, [pixel_intensity_to_current] * 3)


