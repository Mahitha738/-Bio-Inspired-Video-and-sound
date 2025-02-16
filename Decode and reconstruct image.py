import matplotlib.pyplot as plt
import nest
import numpy as np
import os
import h5py
from tqdm import tqdm
import cv2

results_dir = "Videoresults/"
image_extension='.jpg'
images_folder ='video_frames'
rows=174
cols=200
#
# Function to convert pixel intensity to current for each color channel
def pixel_intensity_to_current(intensity, offset=380):
    return intensity + offset

# Function to resize image
def resize_image(image, target_size=(cols, rows)):
    if target_size is None:
        return image
    else:
        return cv2.resize(image, target_size)

# Function to simulate raster plot for each image
def simulate_raster_plot(image_file, current_funcs, sim_time=50.0, pd=13.0):
    nest.ResetKernel
    # Clear console
    os.system('clear')
    sim_time1=sim_time+pd
    # Read the image
    print("Reading image {}...".format(image_file))
    # image = cv2.imread(image_file)
    image = cv2.imread(os.path.join(images_folder, image_file + image_extension))
    # Resize the image
    resized_image = resize_image(image)
    # Get image dimensions
    height, width, _ = resized_image.shape
    print("Image dimensions: Height: {}, Width: {}".format(height, width))
    print(resized_image.shape)

    # Initialize NEST kernel
    nest.ResetKernel()
    nest.set_verbosity(20)  # Set NEST verbosity level to 20
    nest.SetKernelStatus({'print_time': False})

    # Create layers for Blue, Green, and Red channels
    layers_L1 = []
    layers_L2 = []
    layers_L3 = []
    spikerecorders_L1 = []
    spikerecorders_L2 = []
    spikerecorders_L3 = []
    for i, color in enumerate(['Blue','Green','Red']):
        # Create layer with iaf_psc_alpha neurons
        layers_L1.append(nest.Create('iaf_psc_alpha', width * height))
        layers_L2.append(nest.Create('iaf_psc_alpha', width * height))
        layers_L3.append(nest.Create('iaf_psc_alpha', width * height))
        # Connect each layer to a spike recorder
        spikerecorders_L1.append(nest.Create("spike_recorder"))
        spikerecorders_L2.append(nest.Create("spike_recorder"))
        spikerecorders_L3.append(nest.Create("spike_recorder"))

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
                nest.SetStatus(layers_L1[i][neuron_index], {"I_e": current})
                nest.Connect(layers_L1[i][neuron_index], layers_L2[i][neuron_index], "one_to_one", syn_spec={"weight": 1200.0})
                nest.Connect(layers_L2[i][neuron_index], layers_L3[i][neuron_index], "one_to_one", syn_spec={"weight": 1200.0})

                # Update progress bar
                progress_bar.update(1)

        nest.Connect(layers_L1[i], spikerecorders_L1[i])
        nest.Connect(layers_L2[i], spikerecorders_L2[i])
        nest.Connect(layers_L3[i], spikerecorders_L3[i])


    print("Simulating for", image_file)
    nest.Simulate(sim_time1)
    print("Simulation completed for", image_file)

    # Save spike events and senders in HDF5 format
    os.makedirs(results_dir, exist_ok=True)
    with h5py.File(os.path.join(results_dir, image_file + "_spikes.h5"), "w") as file:
        for i, color in enumerate(['Blue', 'Green','Red']):
            events = spikerecorders_L3[i].get("events")
            senders = events["senders"]
            times = events["times"]
            grp = file.create_group(color)
            grp.create_dataset("senders", data=senders)
            grp.create_dataset("times", data=times)
            grp.attrs["image_filename"] = image_file
            grp.attrs["image_dimensions"] = (height, width)
            grp.attrs["simulation_time"] = sim_time

    # Plot raster plot for each color channel
    plt.figure(figsize=(15, 5))
    for i, color in enumerate(['Blue', 'Green','Red']):
        plt.subplot(1, 3, i + 1)
        plt.title('{} Channel - {} - L3'.format(color, image_file))
        plt.xlabel('Time (ms)')
        plt.ylabel('Neuron Index')
        plt.grid()
        ts = spikerecorders_L3[i].get("events")["times"]
        if(np.min(times) > pd):
            print(f"Subtracting the propagation delay: {pd},ms for {color} channel.")
            times -= pd  # Adjusting for propagation delay

        senders = spikerecorders_L3[i].get("events")["senders"]-np.min(spikerecorders_L3[i].get("events")["senders"]) # normalise values between 0 and cols*rows-1

        print(np.min(senders),np.max(senders),(cols*rows))
        np.save(os.path.join(results_dir , image_file + "_"+ color + "_L3_senders.npy"), senders)
        np.save(os.path.join(results_dir , image_file + "_"+ color + "_L3_ts.npy"), ts)
        plt.vlines(ts, senders, senders + 1, color=color.lower(), linewidths=0.5)

    plt.tight_layout()
    # plt.show()

#getting frames
image_files = [f'frame_{i:04d}'for i in range(1,12)]

# Simulate raster plot for each image
for image_file in image_files:
    simulate_raster_plot(image_file, [pixel_intensity_to_current] * 3)


def create_normalised_3D_matrix(fig, times_red, senders_red, times_green, senders_green, times_blue, senders_blue, cols):
    rows = 174 * 200  # You might want to adjust this based on your actual data
    # Initialize the 3D matrix with dimensions for senders, time, and color channels
    normalised_3Dmat = np.zeros((rows, cols, 3))
    for i in range(len(times_red)):
        # Ensure that the time index does not exceed cols - 1
        time_index = min(int(np.round(times_red[i], 0)), cols - 1)  # Prevent out-of-bounds index
        sender_index = int(senders_red[i] - 1)  # Convert sender to zero-based index
        normalised_3Dmat[sender_index, time_index, 0] = 1.0  # red
    # Repeat for green and blue channels with their respective data
    for i in range(len(times_green)):
        time_index = min(int(np.round(times_green[i], 0)), cols - 1)
        sender_index = int(senders_green[i] - 1)
        normalised_3Dmat[sender_index, time_index, 1] = 1.0  # green
    for i in range(len(times_blue)):
        time_index = min(int(np.round(times_blue[i], 0)), cols - 1)
        sender_index = int(senders_blue[i] - 1)
        normalised_3Dmat[sender_index, time_index, 2] = 1.0  # blue
    # Convert to boolean if necessary or keep as is depending on your analysis needs
    return normalised_3Dmat  # Return the filled matrix

def plot_results(fig, normalised_3Dmat, cols):
    """
    Plot the raster plot.

    Parameters:
        fig (str): Figure name.
        normalised_3Dmat (numpy.ndarray): Normalised 3D matrix of spike events.
        cols (int): Number of time bins.
    """
    # Print size
    print(f"normalised_3Dmat: {np.round((normalised_3Dmat.nbytes) / (1024 * 1024), 2)} MB")
    print()

    # Plot the raster plot
    plt.figure(figsize=(10, 6))
    for channel, color in enumerate(['red', 'green', 'blue']):
        print(f"Preparing plot for {color} ...")
        for idx in range(normalised_3Dmat.shape[0]):
            spike_times = np.where(normalised_3Dmat[idx, :, channel])[0] * (cols // 3)  # Adjust the indexing
            if len(spike_times) > 0:
                plt.vlines(spike_times, idx, idx + 1, color=color, linewidth=1)
    print("Done!")
    plt.xlabel('Time [ms]')
    plt.ylabel('Neuron Index')
    plt.title(f"Raster Plot of {fig}")
    plt.ylim(0, normalised_3Dmat.shape[0])
    plt.grid(True)
    plt.savefig(f"results_png/{fig}_raster_plot.png")
    # plt.show()

#
#
# List of figure names
figures = [f'frame_{i:04d}'for i in range(1)]

# List colour names
colours = ["Red","Green","Blue"]
cols=51 # time in ms

# Loop over each figure
for idx in range(0,len(figures)):
    # Read events and senders from files
    print(f"Processing {figures[idx]} ...")
    times_red = np.load(f"Videoresults/{figures[idx]}_Red_L3_ts.npy", allow_pickle=True)
    senders_red = np.load(f"Videoresults/{figures[idx]}_Red_L3_senders.npy", allow_pickle=True)
    times_green = np.load(f"Videoresults/{figures[idx]}_Green_L3_ts.npy", allow_pickle=True)
    senders_green = np.load(f"Videoresults/{figures[idx]}_Green_L3_senders.npy", allow_pickle=True)
    times_blue = np.load(f"Videoresults/{figures[idx]}_Blue_L3_ts.npy", allow_pickle=True)
    senders_blue = np.load(f"Videoresults/{figures[idx]}_Blue_L3_senders.npy", allow_pickle=True)

    # Create the normalized 2D matrix
    normalised_3Dmat = create_normalised_3D_matrix(figures[idx],times_red, senders_red, times_green, senders_green, times_blue, senders_blue, cols)

    # Print results and plot raster plot
    plot_results(figures[idx], normalised_3Dmat,cols)


def pixel_intensity_to_current(intensity, offset=380):
    return intensity + offset

def create_normalised_2D_matrix(times, senders, cols):
    rows=np.max(senders)+1
    normalised_2Dmat = np.zeros((rows, cols))
    for i in range(len(times)):
        normalised_2Dmat[int(senders[i]), int(np.round(times[i], 0))] = 1.0
    normalised_2Dmat.dump(f"Videoresults/normalised_2Dmat.npy")
    return normalised_2Dmat.astype(bool)

def find_spike_pattern(pattern, normalised_2Dmat):
    """
    Find the index of the spike pattern in the normalized 2D matrix.

    Parameters:
        pattern (numpy.ndarray): Spike pattern to search for.
        normalised_2Dmat (numpy.ndarray): Normalized 2D matrix of spike events.

    Returns:
        int: Index of the spike pattern if found, -1 otherwise.
    """
    for idx in range(normalised_2Dmat.shape[0]):
        if np.array_equal(normalised_2Dmat[idx], pattern):
            return idx
    return -1


# Function to generate random spike patterns
def generate_random_pattern(length):
    random_array = np.random.randint(2, size=length)
    return random_array


# Reset NEST kernel
nest.ResetKernel()

# Define the number of neurons
num_neurons = 256
offset = 380

# Create layer with iaf_psc_alpha neurons
layer = nest.Create('iaf_psc_alpha', num_neurons)

spikerecorder = nest.Create("spike_recorder")

# Define the analog values from 0 to 255
min_analog_value = 0+offset
max_analog_value = 255+offset

# Create spike generators for each neuron and inject analog values
ind=0
for value in range(min_analog_value, max_analog_value):
    # Set current for each neuron
    nest.SetStatus(layer[ind], {"I_e": value})
    ind+=1
nest.Connect(layer, spikerecorder)
# Simulate
nest.Simulate(50.0)

# Get spike times
events = spikerecorder.get("events")
senders = events["senders"]
ts = events["times"]

normalised_2Dmat=create_normalised_2D_matrix(ts,senders,51)


# Function to generate random spike patterns
def generate_random_pattern(length):
    return np.random.randint(2, size=length)



#test a known patterns
for i in range(50):
# Find the index of the pattern in the normalized 2D matrix
    idx = np.random.randint(256)
    pattern_index = find_spike_pattern(normalised_2Dmat[idx,:], normalised_2Dmat)
    # Print the result
    if pattern_index != -1:
        print(f"The analogue value for pattern {normalised_2Dmat[idx,:]} is {pattern_index}.")
    else:
        print(f"Pattern {i + 1}: Not found")

    # Generate a random spike pattern
    pattern_to_find = generate_random_pattern(51).astype(bool)


    # Find the index of the pattern in the normalized 2D matrix
    pattern_index = find_spike_pattern(pattern_to_find, normalised_2Dmat)
    # Print the result
    if pattern_index != -1:
        print(f"The analogue value for pattern {pattern_to_find} is {pattern_index}.")
    else:
        print(f"Pattern {i + 1}b: Not found")


def spikes_to_analog(pattern, normalised_2Dmat):
    """
    Convert spikes to analogue values.

    Parameters:
        pattern (numpy.ndarray): Spike pattern to search for.
        normalised_2Dmat (numpy.ndarray): Normalised 2D matrix of spike events.

    Returns:
        int: Index of the spike pattern if found, -1 otherwise.
    """
    for idx in range(normalised_2Dmat.shape[0]):
        if np.array_equal(normalised_2Dmat[idx], pattern):
            return idx
    return -1

# Load the normalised 2D matrix
normalised_2Dmat = np.load(f"Videoresults/normalised_2Dmat.npy", allow_pickle=True)

# List of figure names
figures = [f"frame_{i:04d}" for i in range(1)]
for idx, fig_name in enumerate(figures):
    # Read events and senders from files
    print(f"Processing {fig_name} ...")
    normalised_3Dmat = np.load(f"Videoresults/normalised_3Dmat_{figures[idx]}.npy", allow_pickle=True)
    print(normalised_3Dmat.shape)

    # Create 2D matrix for each channel
    analogue_values_ch = np.zeros((174, 200, 3), dtype=np.uint8)
    for k in range(normalised_3Dmat.shape[2]):  # Iterate over the third dimension
        for i in range(normalised_3Dmat.shape[0]):
            tmp = spikes_to_analog(normalised_3Dmat[i, :, k], normalised_2Dmat)
            # if tmp < 0:
            #     raise ValueError("Invalid value!")
            r, c = divmod(i, 200)  # Calculate row and column indices
            analogue_values_ch[r, c, k] = np.uint8(tmp)
    analogue_values_ch = np.array(analogue_values_ch)

    # Load original image
    original_image = cv2.imread(f"video_frames/{fig_name}.jpg")
    original_image_resized = cv2.resize(original_image, (200, 174))

    # Calculate the absolute difference between original and reconstructed images
    diff_image = cv2.absdiff(original_image_resized, analogue_values_ch)

    # Save and display the plot
    plt.figure(figsize=(10, 10))
    plt.imshow(analogue_values_ch)
    plt.title(f"Reconstructed {figures[idx]}")
    plt.axis('off')
    plt.savefig(f"reconstructedFigures/{figures[idx]}_reconstructed.png")
    plt.show()

    # Save and display the plot
    plt.figure(figsize=(10, 10))
    # Plot the original image
    plt.subplot(3, 1, 1)
    plt.imshow(cv2.cvtColor(original_image_resized, cv2.COLOR_BGR2RGB))
    plt.title(f"Original {figures[idx]}")
    plt.axis('off')

    # Plot the reconstructed image
    plt.subplot(3, 1, 2)
    plt.imshow(analogue_values_ch)
    plt.title(f"Reconstructed {figures[idx]}")
    plt.axis('off')

    # Plot the absolute difference image
    plt.subplot(3, 1, 3)
    plt.imshow(cv2.cvtColor(diff_image, cv2.COLOR_BGR2RGB))
    plt.title(f"Absolute Difference between Original and Reconstructed {figures[idx]}")
    plt.axis('off')

    # Save and display the plot
    plt.savefig(f"reconstructedFigures/{figures[idx]}_comparinson.png")
    plt.show()