import torch.nn as nn
import snntorch as snn
from .config import NUM_INPUTS, NUM_INTERNEURONS, NUM_MSNS

class NAccNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(NUM_INPUTS, NUM_INTERNEURONS)
        self.fc2 = nn.Linear(NUM_INTERNEURONS, NUM_MSNS)
        self.fc3 = nn.Linear(NUM_INPUTS, NUM_MSNS)
        self.spike_fn = snn.Leaky(beta=0.9, threshold=1.0, reset_mechanism="zero")

    def forward(self, x):
        h1 = self.fc1(x)
        spk1, _ = self.spike_fn(h1)
        h2 = -1.0 * self.fc2(spk1)
        spk2, _ = self.spike_fn(h2)
        h3 = self.fc3(x)
        spk3, _ = self.spike_fn(h3)
        msn_in = h2 + h3
        spk_out, _ = self.spike_fn(msn_in)
        return spk1, spk2, spk_out