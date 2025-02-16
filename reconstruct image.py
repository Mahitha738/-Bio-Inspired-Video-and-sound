import numpy as np
import cv2
import matplotlib.pyplot as plt

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
normalised_2Dmat = np.load(f"results/normalised_2Dmat.npy", allow_pickle=True)

# List of figure names
figures = ["frame_0000",]
for idx in range(len(figures)):
    # Read events and senders from files
    print(f"Processing {figures[idx]} ...")
    normalised_3Dmat = np.load(f"results/normalised_3Dmat_{figures[idx]}.npy", allow_pickle=True)
    print(normalised_3Dmat.shape)

    # Create 2D matrix for each channel
    analogue_values_ch = np.zeros((200, 300, 3), dtype=np.uint8)
    for k in range(normalised_3Dmat.shape[2]):  # Iterate over the third dimension
        for i in range(normalised_3Dmat.shape[0]):
            tmp = spikes_to_analog(normalised_3Dmat[i, :, k], normalised_2Dmat)
            if tmp < 0:
                raise ValueError("Invalid value!")
            r, c = divmod(i, 300)  # Calculate row and column indices
            analogue_values_ch[r, c, k] = np.uint8(tmp)
    analogue_values_ch = np.array(analogue_values_ch)

    # Load original image
    original_image = cv2.imread(f"video_frames/{figures[idx]}.jpg")
    original_image_resized = cv2.resize(original_image, (300, 200))

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
