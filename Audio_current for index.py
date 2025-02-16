import numpy as np

# Function to get current for a given index
def get_current_for_idx(idx):
    # Load the current_spikes_values from npy file
    current_spikes_values = np.load('results/current_spikes_values.npy')

    # Check if the index is within the valid range
    if 0 <= idx < len(current_spikes_values):
        return current_spikes_values[idx][0]
    else:
        raise ValueError("Index out of range")


#####################################
# test values from 0 to 9 - Neuron 0
#####################################
# Generate 20 random indices from 0 to 9, including 0
random_indices = np.random.randint(0, 10, size=20)

# Test the function for each random index
for idx in random_indices:
    try:
        current = get_current_for_idx(idx)
        print(f"Current for index {idx}: {current}")
    except ValueError as e:
        print(f"Error: {e}")

#######################################
# test values from 10 to 90 - Neuron 1
#######################################
# Generate 20 random values from 10 to 90
random_values = np.random.randint(10, 90, size=20)
# normlise values
# Test the function for each random index
for idx in random_values:
    try:
        current = get_current_for_idx(int(idx/10))
        print(f"Current for value {idx}: {current}")
    except ValueError as e:
        print(f"Error: {e}")


#########################################
# test values from 100 to 900 - Neuron 2
#########################################
# Generate 20 random values from 100 to 90
random_values = np.random.randint(100, 900, size=20)
# normlise values
# Test the function for each random index
for idx in random_values:
    try:
        current = get_current_for_idx(int(idx/100))
        print(f"Current for value {idx}: {current}")
    except ValueError as e:
        print(f"Error: {e}")

###########################################
# test values from 1000 to 9000 - Neuron 3
###########################################
# Generate 20 random values from 1000 to 9000
random_values = np.random.randint(1000, 9000, size=20)
# normlise values
# Test the function for each random index
for idx in random_values:
    try:
        current = get_current_for_idx(int(idx/1000))
        print(f"Current for value {idx}: {current}")
    except ValueError as e:
        print(f"Error: {e}")

############################################
# test values from 10000 to 90000 - Neuron 4
############################################
# Generate 20 random values from 10000 to 90000
random_values = np.random.randint(10000, 90000, size=20)
# normlise values
# Test the function for each random index
for idx in random_values:
    try:
        current = get_current_for_idx(int(idx/1000))
        print(f"Current for value {idx}: {current}")
    except ValueError as e:
        print(f"Error: {e}")

############################################
# test values from -1 to 1 - Neuron 5
############################################
# Generate 20 random values from 10000 to 90000
random_values = np.random.randint(-1, 2, size=20)
# normlise values
# Test the function for each random index
for idx in random_values:
    try:
        idx1=-idx
        if idx1<0:
            idx1=0
        current = get_current_for_idx(idx1)
        print(f"Current for value {idx}: {current}")
    except ValueError as e:
        print(f"Error: {e}")