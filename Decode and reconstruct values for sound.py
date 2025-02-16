import numpy as np

def get_idx_for_spike_count(spike_count):
    """
    This function takes a spike count and returns the index if it exists in the current_spikes_values numpy array.
    If the spike count does not exist in the array, it returns -1.

    Parameters:
        spike_count (int): The spike count to search for.

    Returns:
        int: The index if the spike count exists, otherwise -1.
    """
    # Load the current_spikes_values from npy file
    spike_values = np.load('results/current_spikes_values.npy')[:, 1]

    # Iterate through the current_spikes_values array to find the index corresponding to the spike count
    for i in range(0,len(spike_values)):
        if spike_values[i] == spike_count:
            return i
    return -1

    # If the spike count does not exist in the array, return -1
    return -1

def reconstruct_value_from_currents(currents):
    """
    Reconstructs the value from the given currents.

    Parameters:
        currents (numpy.ndarray): Array containing currents for each neuron.

    Returns:
        int: Reconstructed value.
    """
    # Extract the indices of the spike counts corresponding to the given currents
    idx_neuron0 = get_idx_for_spike_count(currents[0])
    idx_neuron1 = get_idx_for_spike_count(currents[1])
    idx_neuron2 = get_idx_for_spike_count(currents[2])
    idx_neuron3 = get_idx_for_spike_count(currents[3])
    idx_neuron4 = get_idx_for_spike_count(currents[4])
    idx_neuron5 = get_idx_for_spike_count(currents[5])

    # Reconstruct the value based on the indices
    value = idx_neuron0 + \
            idx_neuron1 * 10 + \
            idx_neuron2 * 100 + \
            idx_neuron3 * 1000 + \
            idx_neuron4 * 10000

    # Check if the value should be negative
    if idx_neuron5 == 1:
        value *= -1

    return value


print ("current_spikes_values:",np.load('results/current_spikes_values.npy')[:,1])

# Test the function with sample currents arrays
sample_currents = [
    np.array([0, 0, 0, 0, 0, 0]),  # Should reconstruct 0
    np.array([1, 0, 0, 0, 0, 0]),  # Should reconstruct 1
    np.array([1, 1, 1, 1, 1, 1]),  # Should reconstruct -11111
    np.array([8, 7, 6, 5, 4, 1]),  # Should reconstruct -45678
    np.array([8, 7, 6, 5, 4, 0]),  # Should reconstruct 45678
    np.array([11, 11, 11, 11, 11, 1]),  # Should reconstruct -99999
    np.array([11, 11, 11, 11, 11, 0]),  # Should reconstruct 99999
]

# Test the function for each sample currents array
for currents in sample_currents:
    reconstructed_value = reconstruct_value_from_currents(currents)
    print("Currents:", currents)
    print("Reconstructed Value:", reconstructed_value)
