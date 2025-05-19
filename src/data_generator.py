import torch
from .config import TIME_STEPS, NUM_INPUTS, BATCH_SIZE

# DEVICE lokal hier definieren
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_poisson_spikes(lambda_low=0.1, lambda_high=1.0):
    """
    Generiert einen Tensor [BATCH_SIZE, NUM_INPUTS, TIME_STEPS]
    mit Poisson-ähnlichen Spike-Trains.
    """
    rates = torch.rand(BATCH_SIZE, NUM_INPUTS, device=DEVICE) * (lambda_high - lambda_low) + lambda_low
    data = torch.zeros(BATCH_SIZE, NUM_INPUTS, TIME_STEPS, device=DEVICE)
    for b in range(BATCH_SIZE):
        for n in range(NUM_INPUTS):
            intervals = torch.exp(torch.rand(TIME_STEPS, device=DEVICE) * -rates[b, n])
            cum_t = torch.cumsum(intervals, dim=0) * TIME_STEPS
            spikes = torch.floor(cum_t).long() % TIME_STEPS
            uniq, _ = torch.unique(spikes, return_counts=True)
            data[b, n, uniq] = 1.0
    return data, DEVICE