# Simulation of the Nucleus Accumbens with Spiking Neural Networks

Did you know that ARTIFICIAL neural networks can also be used to simulate NATURAL neural networks? This repository contains a simulation of the Nucleus Accumbens (NAcc) using a Spiking Neural Network (SNN), to explore a way to study its behavior, especially how it handles rewards and motivation, by creating a digital version. Researchers are increasingly exploring such simulations for brain research, and I wanted to try it out on a small scale with this project.
## What is the Nucleus Accumbens?
The Nucleus Accumbens (NAcc) is a key brain region in the reward system. It helps us experience pleasure, motivation, and rewards, like eating something tasty or achieving goals. It works with Medium-Spiny-Neurons (MSNs), inhibitory interneurons, and excitatory inputs from other brain areas. Neurons in the NAcc fire spikes—short electrical signals—to transmit information.

This code simulates the NAcc with an SNN because it models the main neuron types, their connections, and spiking behavior. It helps explore reward and motivation by simulating random inputs and dynamic interactions.

## Architecture
The code uses `snntorch` to create a network with three main types of neurons, each mimicking specific cells in the Nucleus Accumbens. Here’s a detailed look at each type, why they’re structured this way, and how they function:

- **Input Neurons (100):** These represent signals coming from other brain areas, like the cortex or hippocampus, into the Nucleus Accumbens. In the real brain, these are excitatory neurons that send activating signals to start activity in the NAcc. They’re designed to fire random, irregular spikes in the code to mimic the unpredictable, varied inputs the NAcc receives from different brain regions. This randomness reflects how the brain constantly gets diverse signals, like sensory inputs or thoughts, that trigger reward or motivation responses. In the simulation, these neurons use `spikegen.rate` with variable, irregular timing to create a natural, fluctuating pattern of activity.

- **Inhibitory Interneurons (50):** These are special neurons in the NAcc that act like brakes, slowing down or stopping the activity of other neurons, especially the MSNs. They’re called “inhibitory” because they release chemicals (neurotransmitters) that reduce the firing of nearby neurons, helping to keep the NAcc’s activity balanced and preventing it from getting too excited, which could lead to overreactions or chaos. In the code, these interneurons are programmed to take input from the input neurons, process it, and then send inhibitory signals to the MSNs (multiplied by `-1` in `fc2`). This structure mirrors their role in the brain, where they regulate and fine-tune the NAcc’s responses to ensure stability, like calming down a reward signal that’s too strong, which is crucial for maintaining normal behavior and avoiding extremes like addiction or impulsiveness.

- **Medium-Spiny-Neurons (20):** These are the main cells in the Nucleus Accumbens, often called MSNs, and they’re the key players in deciding how strongly a reward or motivation signal gets sent out. MSNs are shaped like spines with lots of branches (dendrites) to receive many inputs, which makes them perfect for integrating complex signals from different sources—like excitatory inputs from outside and inhibitory inputs from interneurons. In the brain, they fire spikes to activate downstream areas, like the motor system, to drive motivated behavior or pleasure responses. In the code, MSNs receive direct excitatory inputs from input neurons (`fc3`) and inhibitory inputs from interneurons (`fc2`), combining these to produce their own spiking pattern. This setup reflects their role as the central decision-makers in the NAcc, balancing excitement and inhibition to control reward-related actions, like deciding whether to seek out a reward or not.

`nn.Linear` layers model the synapses connecting these neurons: `fc1` (Input → Interneurons, excitatory), `fc2` (Interneurons → MSNs, inhibitory with `-1`), and `fc3` (Input → MSNs, excitatory). `snn.Leaky` simulates spikes with a threshold and reset, mimicking how real neurons fire when their membrane potential exceeds a limit and then reset, just like in the brain.

## How to interpret the plots
The code displays four plots:
1. **Input Neurons Activity (Raster Plot):** Black lines show when neurons fire (0–99). Irregular spacing indicates variable brain input
2. **Interneurons Activity (Raster Plot):** Sparser lines (0–49) show how they inhibit MSNs
3. **MSNs Activity (Raster Plot):** More frequent lines (0–19) reflect the combined effect of inputs and inhibition. More lines suggest stronger reward responses
4. **Input Neurons → Interneurons Activity (Heatmap):** Blue shows activity between input neurons (0–99) and interneurons (0–49). White (weak activity < 0.1) is neutral, while light blue areas indicate strong interactions, revealing how inputs influence inhibition

As a disclaimer though, this code only simulates the Nucleus Accumbens on a very abstract level and a very tiny scale.
