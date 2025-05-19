import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from .config import TIME_STEPS
from .config import NUM_INPUTS, NUM_INTERNEURONS, NUM_MSNS


def plot_spike_rasters(rec_in, rec_int, rec_msn):
    fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    def raster(ax, data, n_neurons, title):
        ax.clear()
        for n in range(n_neurons):
            ts = np.where(data[:, n])[0]
            ax.scatter(ts, np.ones_like(ts)*n, s=5, marker='|')
        ax.set_xlim(0, TIME_STEPS)
        ax.set_ylim(-1, n_neurons)
        ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.5)

    raster(axs[0], rec_in,  NUM_INPUTS,       "Input-Neuronen")
    raster(axs[1], rec_int, NUM_INTERNEURONS, "Interneuronen")
    raster(axs[2], rec_msn, NUM_MSNS,         "MSNs")

    # Beispiel-Aktivitätsmatrix
    weights = np.abs(np.random.randn(NUM_INPUTS, NUM_INTERNEURONS))
    im = axs[3].imshow(weights, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=axs[3], label="Aktivitätsstärke")
    axs[3].set_title("Input → Interneuronen")
    axs[3].set_xlabel("Interneuronen")
    axs[3].set_ylabel("Input-Neuronen")

    lines = [axs[i].axvline(0, color='r', linestyle='--') for i in range(3)]

    def update(frame):
        for ln in lines:
            ln.set_xdata([frame, frame])
        return lines

    ani = FuncAnimation(fig, update, frames=TIME_STEPS, interval=100, blit=True)
    plt.tight_layout()
    plt.show()