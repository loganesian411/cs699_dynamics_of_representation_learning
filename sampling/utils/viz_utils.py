"""Visualization utils."""

import matplotlib.pyplot as plt

def plot_samples(samples, color='g', label=None, density=None, fig=None):
  if fig is None:
    fig = plt.figure(figsize=(4, 4))
  if density is not None:
    plt.imshow(density, alpha=0.3)
  plt.scatter(samples[:, 1], samples[:, 0], color=color, s=0.5, alpha=0.5,
              label=label)
  return fig