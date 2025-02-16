import numpy as np
import matplotlib.pyplot as plt
import nest

def raster_plot(senders_layer1, ts_layer1, senders_layer2, ts_layer2, senders_layer3, ts_layer3,
                senders_noise_layer, ts_noise_layer, senders_lateral_ih_layer, ts_lateral_ih_layer):
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
    plt.title('Spike Raster Plot - GaussianNoise Layer')
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

    plt.tight_layout()
    plt.savefig("SNNresults/Gaussian_Noise_raster_plot.png")
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
    noise_layer = nest.Create("noise_generator", 2)
    lateral_ih_layer = nest.Create("iaf_psc_alpha", 16)

    nest.SetStatus(noise_layer, {"mean": 1.0, "std": 1.0})
    noise_neurons = nest.Create("iaf_psc_alpha", 2)
    nest.Connect(noise_layer, noise_neurons, "one_to_one")

    # Create spike recorders for each layer
    spike_recorder_layer1 = nest.Create("spike_recorder")
    spike_recorder_layer2 = nest.Create("spike_recorder")
    spike_recorder_layer3 = nest.Create("spike_recorder")
    spike_recorder_noise_layer = nest.Create("spike_recorder")
    spike_recorder_lateral_ih_layer = nest.Create("spike_recorder")

    nest.Connect(neuron_layer1, spike_recorder_layer1)
    nest.Connect(neuron_layer2, spike_recorder_layer2)
    nest.Connect(neuron_layer3, spike_recorder_layer3)
    nest.Connect(noise_neurons, spike_recorder_noise_layer)
    nest.Connect(lateral_ih_layer, spike_recorder_lateral_ih_layer)

    syn_spec_l1l2 = {"weight": 1200.0}
    syn_spec_l2l3 = {"weight": 1200.0}
    syn_spec_lnl2 = {"weight": 1200.0}

    connections_layer1 = [(neuron_layer1[0], neuron_layer2[i]) for i in range(2)]
    connections_layer2 = [(neuron_layer1[1], neuron_layer2[i]) for i in range(1, 3)]
    connections_layer3 = [(neuron_layer1[2], neuron_layer2[i]) for i in range(3, 5)]
    connections_layer4 = [(neuron_layer1[3], neuron_layer2[i]) for i in range(4, 6)]
    connections_layer5 = [(neuron_layer1[4], neuron_layer2[i]) for i in range(6, 8)]
    connections_layer6 = [(neuron_layer1[5], neuron_layer2[i]) for i in range(7, 9)]
    connections_layer7 = [(neuron_layer1[6], neuron_layer2[i]) for i in range(9, 11)]
    connections_layer8 = [(neuron_layer1[7], neuron_layer2[i]) for i in range(10, 12)]

    for connection in connections_layer1:
        nest.Connect(connection[0], connection[1], syn_spec=syn_spec_l1l2)
    for connection in connections_layer2:
        nest.Connect(connection[0], connection[1], syn_spec=syn_spec_l1l2)
    for connection in connections_layer3:
        nest.Connect(connection[0], connection[1], syn_spec=syn_spec_l1l2)
    for connection in connections_layer4:
        nest.Connect(connection[0], connection[1], syn_spec=syn_spec_l1l2)
    for connection in connections_layer5:
        nest.Connect(connection[0], connection[1], syn_spec=syn_spec_l1l2)
    for connection in connections_layer6:
        nest.Connect(connection[0], connection[1], syn_spec=syn_spec_l1l2)
    for connection in connections_layer7:
        nest.Connect(connection[0], connection[1], syn_spec=syn_spec_l1l2)
    for connection in connections_layer8:
        nest.Connect(connection[0], connection[1], syn_spec=syn_spec_l1l2)

    for i in range(12):
        nest.Connect(neuron_layer2[i], neuron_layer3[i], "one_to_one", syn_spec=syn_spec_l2l3)

    connection_mapping = {0: [1, 4], 1: [7, 10]}
    for noise_neuron, target_neurons in connection_mapping.items():
        for target_neuron in target_neurons:
            nest.Connect(noise_layer[noise_neuron], neuron_layer2[target_neuron], syn_spec=syn_spec_lnl2)

    for step in range(num_steps):
        print(f"Step {step + 1}/{num_steps}")

        random_currents = np.random.uniform(min_current, max_current, size=2)
        for i, current in enumerate(random_currents):
            nest.SetStatus(neuron_layer1[i], {"I_e": current})

        nest.Simulate(simulation_duration)
    # Create multimeter for recording membrane potentials
    multimeter_noise_layer = nest.Create("multimeter", params={"record_from": ["V_m"]})
    nest.Connect(multimeter_noise_layer, noise_neurons)

    # Create multimeter spike recorders for each layer
    multimeter_recorder_layer1 = nest.Create("multimeter", params={"record_from": ["V_m"]})
    multimeter_recorder_layer2 = nest.Create("multimeter", params={"record_from": ["V_m"]})
    multimeter_recorder_layer3 = nest.Create("multimeter", params={"record_from": ["V_m"]})
    multimeter_recorder_lateral_ih_layer = nest.Create("multimeter", params={"record_from": ["V_m"]})

    # Connect multimeter spike recorders to the neurons
    nest.Connect(multimeter_recorder_layer1, neuron_layer1)
    nest.Connect(multimeter_recorder_layer2, neuron_layer2)
    nest.Connect(multimeter_recorder_layer3, neuron_layer3)
    nest.Connect(multimeter_recorder_lateral_ih_layer, lateral_ih_layer)

    # Simulate the network
    nest.Simulate(simulation_duration)

    # Retrieve membrane potential data from multimeters
    membrane_potential_noise_layer = nest.GetStatus(multimeter_noise_layer, "events")[0]
    membrane_potential_recorder_layer1 = nest.GetStatus(multimeter_recorder_layer1, "events")[0]
    membrane_potential_recorder_layer2 = nest.GetStatus(multimeter_recorder_layer2, "events")[0]
    membrane_potential_recorder_layer3 = nest.GetStatus(multimeter_recorder_layer3, "events")[0]
    membrane_potential_recorder_lateral_ih_layer = nest.GetStatus(multimeter_recorder_lateral_ih_layer, "events")[0]

    # Extract membrane potential data
    times_noise_layer = membrane_potential_noise_layer["times"]
    V_m_noise_layer = membrane_potential_noise_layer["V_m"]
    times_noise_layer =  membrane_potential_recorder_layer1["times"]
    V_m_noise_layer =  membrane_potential_recorder_layer1["V_m"]

    # Plot membrane potential data
    plt.figure(figsize=(8, 6))
    plt.plot(times_noise_layer, V_m_noise_layer, label="Neuron in Noise Layer")
    plt.xlabel("Time (ms)")
    plt.ylabel("Membrane Potential (mV)")
    plt.title("Membrane Potential - GaussianNoise Layer")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    def plot_membrane_potential(spike_recorder, title):
        plt.figure(figsize=(8, 6))
        plt.title(title)
        plt.xlabel("Time (ms)")
        plt.ylabel("Membrane Potential (mV)")

        for i, recorder in enumerate(spike_recorder):
            events = nest.GetStatus(recorder, "events")[0]
            if "V_m" in events:  # Check if membrane potential is recorded
                plt.plot(events["times"], events["V_m"], label=f"Neuron {i + 1}")
            else:
                print(f"Membrane potential not recorded for Neuron {i + 1}")

        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

    plot_membrane_potential(spike_recorder_noise_layer, "Membrane Potential - GaussianNoise Layer")

    events_layer1 = nest.GetStatus(spike_recorder_layer1, "events")[0]
    events_layer2 = nest.GetStatus(spike_recorder_layer2, "events")[0]
    events_layer3 = nest.GetStatus(spike_recorder_layer3, "events")[0]
    events_noise_layer = nest.GetStatus(spike_recorder_noise_layer, "events")[0]
    events_lateral_ih_layer = nest.GetStatus(spike_recorder_lateral_ih_layer, "events")[0]

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

    raster_plot(senders_layer1, ts_layer1, senders_layer2, ts_layer2, senders_layer3, ts_layer3,
                senders_noise_layer, ts_noise_layer, senders_lateral_ih_layer, ts_lateral_ih_layer)

# Call the function with the updated parameters
simulate_neural_network(num_steps=20, simulation_duration=50.0, min_current=300.0, max_current=600.0)