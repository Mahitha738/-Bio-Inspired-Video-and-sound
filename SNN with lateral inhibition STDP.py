import numpy as np
import matplotlib.pyplot as plt
import nest


def raster_plot(senders_layer1, ts_layer1, senders_layer2, ts_layer2, senders_layer3, ts_layer3,
                senders_noise_layer, ts_noise_layer, senders_lateral_ih_layer, ts_lateral_ih_layer, weights):
    plt.figure(figsize=(10, 8))

    # Layer 1
    plt.subplot(3, 2, 1)
    plt.title('Spike Raster Plot - Layer 1')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.grid()
    for sender, spike_time in zip(senders_layer1, ts_layer1):
        plt.vlines(spike_time, sender, sender + 1, color='red')

    # Layer 2
    plt.subplot(3, 2, 2)
    plt.title('Spike Raster Plot - Layer 2')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.grid()
    for sender, spike_time in zip(senders_layer2, ts_layer2):
        plt.vlines(spike_time, sender, sender + 1, color='blue')

    # Layer 3
    plt.subplot(3, 2, 3)
    plt.title('Spike Raster Plot - Layer 3')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.grid()
    for sender, spike_time in zip(senders_layer3, ts_layer3):
        plt.vlines(spike_time, sender, sender + 1, color='green')

    # Noise Layer
    plt.subplot(3, 2, 4)
    plt.title('Spike Raster Plot - Noise Layer')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.grid()
    for sender, spike_time in zip(senders_noise_layer, ts_noise_layer):
        plt.vlines(spike_time, sender, sender + 1, color='orange')

    # Lateral Ih Layer
    plt.subplot(3, 2, 5)
    plt.title('Spike Raster Plot - Lateral Ih Layer')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.grid()
    for sender, spike_time in zip(senders_lateral_ih_layer, ts_lateral_ih_layer):
        plt.vlines(spike_time, sender, sender + 1, color='violet')

    # STDP weights
    plt.subplot(3, 2, 6)
    plt.plot(weights[0], label='Layer 1 -> Ih Layer', color='red')
    plt.plot(weights[1], label='Layer 2 -> Ih Layer', color='blue')
    plt.plot(weights[2], label='Ih Layer -> Layer 2', color='green')
    plt.xlabel('Simulation Step')
    plt.ylabel('Synaptic Weight')
    plt.title('STDP Synaptic Weight Evolution')
    plt.grid()

    plt.tight_layout()
    plt.savefig("SNNresults/SNNwithSTDP_raster_plot.png")
    plt.show()

def simulate_neural_network(num_steps=20, simulation_duration=50.0, min_current=300.0, max_current=600.0):
    # Reset the NEST simulator
    nest.ResetKernel()
    nest.set_verbosity(20)  # Set NEST verbosity level to 20
    nest.SetKernelStatus({'print_time': False})

    # Create the neurons
    nest.SetDefaults("iaf_psc_alpha", {"I_e": 0.0})
    neuron_layer1 = nest.Create("iaf_psc_alpha", 8)
    neuron_layer2 = nest.Create("iaf_psc_alpha", 12)
    neuron_layer3 = nest.Create("iaf_psc_alpha", 12)
    noise_layer = nest.Create("poisson_generator", 2)
    lateral_ih_layer = nest.Create("iaf_psc_alpha", 16)


    nest.SetStatus(noise_layer, {"rate": 10.0})  # Set the firing rate of the noisy neurons

    # Create spike recorders for each layer
    spike_recorder_layer1 = nest.Create("spike_recorder")
    spike_recorder_layer2 = nest.Create("spike_recorder")
    spike_recorder_layer3 = nest.Create("spike_recorder")
    spike_recorder_noise_layer = nest.Create("spike_recorder")
    spike_recorder_lateral_ih_layer = nest.Create("spike_recorder")

    # Connect the spike recorders to the neurons
    nest.Connect(neuron_layer1, spike_recorder_layer1)
    nest.Connect(neuron_layer2, spike_recorder_layer2)
    nest.Connect(neuron_layer3, spike_recorder_layer3)
    nest.Connect(noise_layer, spike_recorder_noise_layer)
    nest.Connect(lateral_ih_layer, spike_recorder_lateral_ih_layer)

    # Define connectivity between neurons
    syn_spec_l1l2 = {"weight": 1200.0}
    syn_spec_l1lih = {"synapse_model": "stdp_synapse","weight": -1200.0}
    syn_spec_l2l3 = {"weight": 1200.0}
    syn_spec_lnl2 = {"weight": 1200.0}
    syn_spec_l2ih = {"synapse_model": "stdp_synapse","weight": 1200.0}
    syn_spec_ihl2 = {"synapse_model": "stdp_synapse","weight": -600.0}

    # Define the connections for neuron 1-8 of layer 1 to neurons of layer 2
    connections_layer1aih = [(neuron_layer1[0], lateral_ih_layer[i]) for i in range(2)]
    connections_layer1bih = [(neuron_layer1[1], lateral_ih_layer[i]) for i in range(1, )]
    connections_layer1cih = [(neuron_layer1[2], lateral_ih_layer[i]) for i in range(3, 5)]
    connections_layer1dih = [(neuron_layer1[3], lateral_ih_layer[i]) for i in range(4, 6)]
    connections_layer1eih = [(neuron_layer1[4], lateral_ih_layer[i]) for i in range(6, 8)]
    connections_layer1fih = [(neuron_layer1[5], lateral_ih_layer[i]) for i in range(7, 9)]
    connections_layer1gih = [(neuron_layer1[6], lateral_ih_layer[i]) for i in range(9, 11)]
    connections_layer1hih = [(neuron_layer1[7], lateral_ih_layer[i]) for i in range(11, 15)]


    # Connect neuron 1-8 of layer1  to neurons of layer 2
    for connection in connections_layer1aih:
        nest.Connect(connection[0], connection[1], syn_spec=syn_spec_l1l2)
    for connection in connections_layer1bih:
        nest.Connect(connection[0], connection[1],  syn_spec=syn_spec_l1l2)
    for connection in connections_layer1cih:
        nest.Connect(connection[0], connection[1], syn_spec=syn_spec_l1l2)
    for connection in connections_layer1dih:
        nest.Connect(connection[0], connection[1], syn_spec=syn_spec_l1l2)
    for connection in connections_layer1eih:
        nest.Connect(connection[0], connection[1],  syn_spec=syn_spec_l1l2)
    for connection in connections_layer1fih:
        nest.Connect(connection[0], connection[1], syn_spec=syn_spec_l1l2)
    for connection in connections_layer1gih:
        nest.Connect(connection[0], connection[1], syn_spec=syn_spec_l1l2)
    for connection in connections_layer1hih:
        nest.Connect(connection[0], connection[1], syn_spec=syn_spec_l1l2)



    # Define the connections for neuron 1 of layer 1 to neurons of inh layer
    connections_layer1 = [(neuron_layer1[0], neuron_layer2[i]) for i in range(2)]
    connections_layer2 = [(neuron_layer1[1], neuron_layer2[i]) for i in range(1, 3)]
    connections_layer3 = [(neuron_layer1[2], neuron_layer2[i]) for i in range(3, 5)]
    connections_layer4 = [(neuron_layer1[3], neuron_layer2[i]) for i in range(4, 6)]
    connections_layer5 = [(neuron_layer1[4], neuron_layer2[i]) for i in range(6, 8)]
    connections_layer6 = [(neuron_layer1[5], neuron_layer2[i]) for i in range(7, 9)]
    connections_layer7 = [(neuron_layer1[6], neuron_layer2[i]) for i in range(9, 11)]
    connections_layer8 = [(neuron_layer1[7], neuron_layer2[i]) for i in range(10, 12)]

    # Connect neuron 1 of layer 1 to neurons of inh layer
    for connection in connections_layer1:
        nest.Connect(connection[0], connection[1], syn_spec=syn_spec_l1lih)
    for connection in connections_layer2:
        nest.Connect(connection[0], connection[1], syn_spec=syn_spec_l1lih)
    for connection in connections_layer3:
        nest.Connect(connection[0], connection[1], syn_spec=syn_spec_l1lih)
    for connection in connections_layer4:
        nest.Connect(connection[0], connection[1], syn_spec=syn_spec_l1lih)
    for connection in connections_layer5:
        nest.Connect(connection[0], connection[1], syn_spec=syn_spec_l1lih)
    for connection in connections_layer6:
        nest.Connect(connection[0], connection[1], syn_spec=syn_spec_l1lih)
    for connection in connections_layer7:
        nest.Connect(connection[0], connection[1], syn_spec=syn_spec_l1lih)
    for connection in connections_layer8:
        nest.Connect(connection[0], connection[1], syn_spec=syn_spec_l1lih)


    # Connect layer 2 to layer 3
    for i in range(12):
        nest.Connect(neuron_layer2[i], neuron_layer3[i], "one_to_one", syn_spec=syn_spec_l2l3)

    # Define the connections from noisy neurons to layer 2 neurons
    connection_mapping = {0: [1, 4], 1: [7, 10]}

    # Connect the noisy neurons to specific neurons in layer 2
    for noise_neuron, target_neurons in connection_mapping.items():
        for target_neuron in target_neurons:
            nest.Connect(noise_layer[noise_neuron], neuron_layer2[target_neuron], syn_spec=syn_spec_lnl2)

    # compensation close loop approach with static synapses
    nest.Connect(lateral_ih_layer, neuron_layer2, syn_spec=syn_spec_l2ih) #ex
    nest.Connect(neuron_layer2, lateral_ih_layer, syn_spec=syn_spec_ihl2) #in

    # Define synaptic weight recording for STDP connections
    stdp_synapse_weights_l1lih = []
    stdp_synapse_weights_l2ih = []
    stdp_synapse_weights_ihl2 = []


    for step in range(num_steps):
        print(f"Step {step + 1}/{num_steps}")

        # Generate random currents for neurons 1 and 2 in layer 1
        random_currents = np.random.uniform(min_current, max_current, size=2)

        # Apply the random currents to neurons in layer 1
        for i, current in enumerate(random_currents):
            nest.SetStatus(neuron_layer1[i], {"I_e": current})

        # Simulate the network for 50 ms
        nest.Simulate(simulation_duration)
        # Record synaptic weights after each simulation step
        stdp_synapse_weights_l1lih.append(nest.GetStatus(nest.GetConnections(neuron_layer1, lateral_ih_layer), "weight"))
        stdp_synapse_weights_l2ih.append(nest.GetStatus(nest.GetConnections(neuron_layer2, lateral_ih_layer), "weight"))
        stdp_synapse_weights_ihl2.append(nest.GetStatus(nest.GetConnections(lateral_ih_layer, neuron_layer2), "weight"))

    # Retrieve spike times from spike recorders
    events_layer1 = nest.GetStatus(spike_recorder_layer1, "events")[0]
    events_layer2 = nest.GetStatus(spike_recorder_layer2, "events")[0]
    events_layer3 = nest.GetStatus(spike_recorder_layer3, "events")[0]
    events_noise_layer = nest.GetStatus(spike_recorder_noise_layer, "events")[0]
    events_lateral_ih_layer = nest.GetStatus(spike_recorder_lateral_ih_layer, "events")[0]

    # Convert lists to numpy arrays for easier manipulation
    stdp_synapse_weights_l1lih = np.array(stdp_synapse_weights_l1lih)
    stdp_synapse_weights_l2ih = np.array(stdp_synapse_weights_l2ih)
    stdp_synapse_weights_ihl2 = np.array(stdp_synapse_weights_ihl2)

    # Print or visualize the recorded synaptic weights
    print("Synaptic weights from layer 1 to lateral inhibition layer (l1 to lih):")
    print(stdp_synapse_weights_l1lih)

    print("Synaptic weights from layer 2 to lateral inhibition layer (l2 to lih):")
    print(stdp_synapse_weights_l2ih)

    print("Synaptic weights from lateral inhibition layer to layer 2 (lih to l2):")
    print(stdp_synapse_weights_ihl2)

    # Extract senders and spike times
    senders_layer1 = events_layer1["senders"]
    ts_layer1 = events_layer1["times"]

    senders_layer2 = events_layer2["senders"]
    ts_layer2 = events_layer2["times"]

    senders_layer3 = events_layer3["senders"]
    ts_layer3 = events_layer3["times"]

    senders_noise_layer = events_noise_layer["senders"]
    ts_noise_layer = events_noise_layer["times"]

    senders_lateral_ih_layer = events_lateral_ih_layer["senders"]
    ts_lateral_ih_layer = events_lateral_ih_layer["times"]

    weights=[stdp_synapse_weights_l1lih, stdp_synapse_weights_l2ih, stdp_synapse_weights_ihl2]
    # Call the function with the senders and ts
    raster_plot(senders_layer1, ts_layer1, senders_layer2, ts_layer2, senders_layer3, ts_layer3,
                senders_noise_layer, ts_noise_layer, senders_lateral_ih_layer, ts_lateral_ih_layer, weights)


simulate_neural_network(num_steps=20, simulation_duration=50.0, min_current=300.0, max_current=600.0)