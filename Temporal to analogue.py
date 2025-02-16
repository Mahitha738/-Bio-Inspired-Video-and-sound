import numpy as np
import nest


def pixel_intensity_to_current(intensity, offset=380):
    return intensity + offset

def create_normalised_2D_matrix(times, senders, cols):
    rows=np.max(senders)+1
    normalised_2Dmat = np.zeros((rows, cols))
    for i in range(len(times)):
        normalised_2Dmat[int(senders[i]), int(np.round(times[i], 0))] = 1.0
    normalised_2Dmat.dump(f"results/normalised_2Dmat.npy")
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


