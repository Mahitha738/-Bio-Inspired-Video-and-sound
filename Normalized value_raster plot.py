import numpy as np
import matplotlib.pyplot as plt

def create_normalised_3D_matrix(fig,times_red, senders_red, times_green, senders_green, times_blue, senders_blue, cols):
    rows=300*200
    normalised_3Dmat = np.zeros((rows, cols,3))
    for i in range(len(times_red)):
        normalised_3Dmat[int(senders_red[i]-1), int(np.round(times_red[i], 0)),0] = 1.0 # red
    for i in range(len(times_green)):
        normalised_3Dmat[int(senders_green[i]-1), int(np.round(times_green[i], 0)), 1] = 1.0 # green
    for i in range(len(times_blue)):
        normalised_3Dmat[int(senders_blue[i]-1), int(np.round(times_blue[i], 0)), 2] = 1.0  # blue
    normalised_3Dmat.dump(f"results/normalised_3Dmat_{fig}.npy")
    return normalised_3Dmat.astype(bool)



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
    plt.savefig(f"results/{fig}_raster_plot.png")
    plt.show()



# List of figure names
figures = [f"frame_{i:04d}" for i in range(0,15)]

# List colour names
colours = ["Red","Green","Blue"]
cols=51 # time in ms

# Loop over each figure
for idx, fig_name in enumerate (figures):
    # Read events and senders from files
    print(f"Processing {fig_name} ...")
    times_red = np.load(f"encodedFiles/{fig_name}_Red_times.npy", allow_pickle=True)
    senders_red = np.load(f"encodedFiles/{fig_name}_Red_senders.npy", allow_pickle=True)
    times_green = np.load(f"encodedFiles/{fig_name}_Green_times.npy", allow_pickle=True)
    senders_green = np.load(f"encodedFiles/{fig_name}_Green_senders.npy", allow_pickle=True)
    times_blue = np.load(f"encodedFiles/{fig_name}_Blue_times.npy", allow_pickle=True)
    senders_blue = np.load(f"encodedFiles/{fig_name}_Blue_senders.npy", allow_pickle=True)

    # Create the normalized 2D matrix
    normalised_3Dmat = create_normalised_3D_matrix(fig_name,times_red, senders_red, times_green, senders_green, times_blue, senders_blue, cols)

    # Print results and plot raster plot
    plot_results(fig_name, normalised_3Dmat,cols)
