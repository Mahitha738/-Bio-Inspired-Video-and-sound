import cv2
import nest
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# Define paths and files
images_folder = "video_frames"  # Adjust folder name as per your directory structure
image_extension = ".jpg"
image_files = sorted([f for f in os.listdir(images_folder) if f.endswith(image_extension)])[:3]  # Take only the first 3 images
results_dir = "results"

# Define target size for images
target_size = (300, 200)  # width, height

def pixel_intensity_to_current(intensity, offset=380):
    """Convert pixel intensity to current with an offset."""
    return intensity + offset

def simulate_raster_plot_rgb(image_file, current_func, sim_time=100.0):
    image_path = os.path.join(images_folder, image_file + image_extension)
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, target_size)

    # Separate color channels
    blue_channel, green_channel, red_channel = cv2.split(resized_image)

    # Flatten color channels for processing
    blue_flat = blue_channel.flatten()
    green_flat = green_channel.flatten()
    red_flat = red_channel.flatten()

    # Prepare NEST simulation
    nest.ResetKernel()
    num_neurons = target_size[0] * target_size[1]

    # Create neuron layers for RGB channels
    layer_red = nest.Create('iaf_psc_alpha', num_neurons)
    layer_green = nest.Create('iaf_psc_alpha', num_neurons)
    layer_blue = nest.Create('iaf_psc_alpha', num_neurons)

    # Create spike recorders
    spikerecorder_red = nest.Create("spike_recorder")
    spikerecorder_green = nest.Create("spike_recorder")
    spikerecorder_blue = nest.Create("spike_recorder")

    # Connect neurons to spike recorders
    nest.Connect(layer_red, spikerecorder_red)
    nest.Connect(layer_green, spikerecorder_green)
    nest.Connect(layer_blue, spikerecorder_blue)

    # Set currents based on pixel values
    for i in range(num_neurons):
        nest.SetStatus(layer_red[i:i + 1], {"I_e": current_func(red_flat[i])})
        nest.SetStatus(layer_green[i:i + 1], {"I_e": current_func(green_flat[i])})
        nest.SetStatus(layer_blue[i:i + 1], {"I_e": current_func(blue_flat[i])})

    # Run simulation
    nest.Simulate(sim_time)

    # Plotting
    plt.figure(figsize=(15, 10))
    for idx, (color, spikerecorder) in enumerate(zip(['Red', 'Green', 'Blue'],
                                                     [spikerecorder_red, spikerecorder_green, spikerecorder_blue]), 1):
        events = nest.GetStatus(spikerecorder, keys="events")[0]
        times = events["times"]
        senders = events["senders"]
        plt.subplot(3, 1, idx)
        plt.vlines(times, senders, senders + 1, color=color.lower(), label=f"{image_file} - {color}")
        plt.title(f'Raster Plot - {color} Channel')
        plt.xlabel('Time (ms)')
        plt.ylabel('Neuron Index')
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.show()

# Ensure the results directory exists
os.makedirs(results_dir, exist_ok=True)

# Run simulation for each image
for image_file in image_files:
    filename_without_extension = os.path.splitext(image_file)[0]
    simulate_raster_plot_rgb(filename_without_extension, pixel_intensity_to_current)
