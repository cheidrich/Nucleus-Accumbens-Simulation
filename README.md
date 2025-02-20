# Simulation of the Nucleus Accumbens with Spiking Neural Networks

Did you know that artificial neural networks can also be used to simulate natural neural networks? This repository contains a simulation of the Nucleus Accumbens (NAcc) using a Spiking Neural Network (SNN), providing insights into brain reward and motivation processes.

### What is Nucleus Accumbens Simulation?
The "Nucleus Accumbens Simulation" refers to using this code to mimic how the Nucleus Accumbens works in the brain. It’s like creating a digital version of this brain region to study its behavior, especially how it handles rewards and motivation. By running this simulation, we can see how neurons fire and interact, helping researchers understand real brain processes without experimenting directly on people or animals.

## What is the Nucleus Accumbens, and Why Simulate It?
The Nucleus Accumbens (NAcc) is a key brain region in the reward system. It helps us experience pleasure, motivation, and rewards, like eating something tasty or achieving goals that make us happy. It works with Medium-Spiny-Neurons (MSNs), inhibitory interneurons, and excitatory inputs from other brain areas. Neurons in the NAcc fire spikes—short electrical signals—to transmit information.

This code simulates the NAcc with an SNN because it models the main neuron types, their connections, and spiking behavior. It helps understand reward and motivation by simulating random inputs and dynamic interactions.

### Why Nucleus Accumbens Simulation?
Simulating the NAcc is useful because it’s hard to study this deep brain area directly in living brains. This simulation lets us test ideas about how the NAcc processes rewards, like why we feel motivated or enjoy certain things. It’s a tool for neuroscience research, education, and exploring brain functions in a controlled, virtual environment.

## How Does the Code Work as an NAcc Simulation?
The code uses `snntorch` to create a network with:
- **Input Neurons (100):** Simulate signals from other brain areas with random, irregular spikes.
- **Interneurons (50):** Inhibitory, dampening MSNs, as in the real NAcc.
- **Medium-Spiny-Neurons (20):** Main cells, influenced excitatorily by inputs and inhibited by interneurons.

`nn.Linear` layers model synapses: `fc1` (Input → Interneurons, excitatory), `fc2` (Interneurons → MSNs, inhibitory with `-1`), and `fc3` (Input → MSNs, excitatory). `snn.Leaky` simulates spikes with a threshold and reset, mimicking real neurons.

## What Do the Plots Show, and How Can You Interpret Them?
The code displays four plots:
1. **Input Neurons Activity (Raster Plot):** Black lines show when neurons fire (0–99). Irregular spacing indicates variable brain input. The red line marks the current time step.
2. **Interneurons Activity (Raster Plot):** Sparser lines (0–49) show how they inhibit MSNs.
3. **MSNs Activity (Raster Plot):** More frequent lines (0–19) reflect the combined effect of inputs and inhibition. More lines suggest stronger reward responses.
4. **Input Neurons → Interneurons Activity (Heatmap):** Blue shows activity between input neurons (0–99) and interneurons (0–49). White (weak activity < 0.1) is neutral, while light blue areas indicate strong interactions, revealing how inputs influence inhibition.

These plots show dynamic activities and interactions, similar to the NAcc, and help analyze reward processes.

## Neuronal Level and Code Implementation
In the brain, neurons have dendrites (input), a cell body (processing), and axons (output), connected by synapses. The NAcc has excitatory inputs, inhibitory interneurons, and MSNs for rewards. The code mirrors this: spikes with `snn.Leaky`, synapses with `nn.Linear`, and a three-layer structure to simulate real dynamics.

This code is a simple yet effective NAcc simulation for research on reward and motivation!
