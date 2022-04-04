"""Visualization utils."""

import matplotlib.pyplot as plt
import numpy as np

def plot_samples(samples, color='g', label=None, density=None,
                 extent=None, fig=None):
  if fig is None:
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(1, 1, 1)
  else:
    ax = fig.get_axes()[0]

  if extent is None:
    low = np.min(samples, axis=0)
    high = np.max(samples, axis=0)
    extent = [low[0], high[0], low[1], high[1]]

  if density is not None:
    ax.imshow(density, alpha=0.3, extent=extent)

  ax.scatter(samples[:, 1], samples[:, 0], color=color, s=0.5, alpha=0.5,
             label=label)
  return fig