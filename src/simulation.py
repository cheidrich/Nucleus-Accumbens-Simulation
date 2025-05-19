import numpy as np
import torch
import snntorch.utils as utils
from .config import TIME_STEPS
from .data_generator import generate_poisson_spikes
from .network import NAccNetwork


def run_simulation():
    """
    Führt die Simulation aus und liefert Spike-Records.

    :return: (rec_in, rec_int, rec_msn) als numpy-Arrays
    """
    net = NAccNetwork()
    data, DEVICE = generate_poisson_spikes()
    net.to(DEVICE)
    utils.reset(net)

    rec_in, rec_int, rec_msn = [], [], []
    for t in range(TIME_STEPS):
        inp = data[:, :, t]
        spk1, spk2, spk_out = net(inp)
        rec_in.append(inp.cpu().numpy())
        rec_int.append(spk1.cpu().numpy())
        rec_msn.append(spk_out.cpu().numpy())

    # Stapeln und Squeeze
    rec_in  = np.squeeze(np.stack(rec_in))
    rec_int = np.squeeze(np.stack(rec_int))
    rec_msn = np.squeeze(np.stack(rec_msn))
    return rec_in, rec_int, rec_msn