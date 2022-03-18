## TODO(loganesian): update the file docstring.
"""Script to generate different test density functions.

The implementation is taken from rtqichen's ffjord GitHub repository and from 
necludov's continuous-gibbs GitHub repository with only minor modifications to
work with our pipeline (for example, porting to jax).

Source code: https://github.com/rtqichen/ffjord/blob/master/lib/toy_data.py
Associated Paper: Will Grathwohl*, Ricky T. Q. Chen*, Jesse Bettencourt,
    Ilya Sutskever, David Duvenaud. "FFJORD: Free-form Continuous Dynamics for
    Scalable Reversible Generative Models." International Conference on Learning
    Representations (2019).

Source code: https://github.com/necludov/continuous-gibbs/blob/main/notebooks/2d-illustrations.ipynb
Associated Paper: Kirill Neklyudov, Roberto Bondesan, Max Welling. "Deterministic
    Gibbs Sampling via Ordinary Differential Equations." arXiv (2021).
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import sklearn.datasets
import skimage
import os

def plot_target(probs):
  """Taken from necludov's repository and modified slightly."""
  fig = plt.figure()
  plt.imshow(probs, interpolation='bilinear', origin='lower')
  plt.xticks(np.arange(probs.shape[1])-0.5, labels=np.arange(probs.shape[1]))
  plt.yticks(np.arange(probs.shape[0])-0.5, labels=np.arange(probs.shape[0]))
  plt.box(False)
  return fig

# TODO(loganesian): num_samples should eventually be removed.
def generate_density(data, num_samples=100, rng=None, nx=10, ny=12):
  """Generates an image to use as a toy density.

  Args:
    data: str. Name of the dataset to generate. One of:
      ""
    rng:

  Returns:

  """
  if rng is None:
    rng = np.random.RandomState()

  dirout = os.getcwd()

  if data == 'rings':
    center = (0.0, 0.0)
    radii = [0.25, 0.5, 0.75, 1.]
    linewidth = 5
    fig, axes = plt.subplots()
    for r in radii:
      axes.add_artist(plt.Circle(center, r, fill=False, lw=linewidth))
    axes.set_aspect('equal')
    axes.set_xlim([-1.05, 1.05])
    axes.set_ylim([-1.05, 1.05])
    plt.axis('off')
    plt.box(False)
    fig.set_size_inches(5, 5)
    fig.savefig(os.path.join(dirout, 'rings.png'), dpi=100) # bbox_inches='tight')

    # https://stackoverflow.com/questions/7821518/matplotlib-save-plot-to-numpy-array
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # 1 minus to invert --> rings are high density regions
    data = 1 - skimage.color.rgb2gray(data)
    return data

  elif data == 'moons':
    # TODO(loganesian): Convert this to a continuous image that we can sample.
    data = sklearn.datasets.make_moons(n_samples=num_samples, noise=0.1)[0]
    data = data.astype("float32")
    data = data * 2 + np.array([-1, -0.2])
    return data

  elif data == 'chelsea':
    data = skimage.data.chelsea()
    fig = plt.figure()
    plt.imshow(data, cmap=plt.cm.gray)
    plt.axis('off')
    plt.box(False)
    fig.savefig(os.path.join(dirout, 'chelsea.png'), bbox_inches='tight')
    return data

  elif data == 'checkerboard':
    data = skimage.data.checkerboard()
    fig = plt.figure()
    plt.imshow(data, cmap=plt.cm.gray)
    plt.axis('off')
    plt.box(False)
    fig.savefig(os.path.join(dirout, 'checkerboard.png'), bbox_inches='tight')
    return data

  elif data == 'mixture_of_2gaussians':
    # TODO(loganesian): debug thiiiiiiis; would be better to save out image like
    # rings to use and to also make the covariance adjustable???
    x, y = np.arange(nx), np.arange(ny)
    x_grid, y_grid = np.meshgrid(x, y)
    stacked_grid = np.stack([x_grid, y_grid], axis=2)
    probs = 0.5 * scipy.stats.multivariate_normal.pdf(stacked_grid,
                                                      mean=[nx/5, ny/1.4],
                                                      cov=[[1., 0.7], [0.7, 1.]]) +\
            0.5 * scipy.stats.multivariate_normal.pdf(stacked_grid,
                                                      mean=[nx/1.4, ny/5],
                                                      cov=[[1., 0.7], [0.7, 1.]])
    probs /= np.sum(probs)
    fig = plot_target(probs)
    fig.savefig(os.path.join(dirout, 'mixture_of_2gaussians.png'),
                bbox_inches='tight')
    return probs

  # Default
  return generate_density('mixture_of_2gaussians', num_samples, rng, nx, ny)