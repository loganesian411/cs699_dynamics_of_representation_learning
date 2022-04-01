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
import jax
import matplotlib.image as plt_im
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import sklearn.datasets
import skimage
import os

def convert_image_to_array(fig):
  """Routine to convert figure object to an array."""
  # https://stackoverflow.com/questions/7821518/matplotlib-save-plot-to-numpy-array
  data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
  return data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

# TODO(loganesian): num_samples should eventually be removed.
def generate_density(data, num_samples=100000, nx=10, ny=12):
  """Generates an image to use as a toy density.

  Args:
    data: str. Name of the dataset to generate. One of:
      ""
    rng:

  Returns:

  """
  dirout = os.path.dirname(__file__)

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
    # Have to save image in order to convert to an array afterwards.
    fig.savefig(os.path.join(dirout, 'rings.png'), dpi=100)
    plt.close()
    return convert_image_to_array(fig)

  elif data == 'moons':
    data, labels = sklearn.datasets.make_moons(n_samples=num_samples, noise=0.1)
    data = jax.numpy.asarray(data.astype('float32'))
    data = data * 2 + jax.numpy.array([-1, -0.2])
    
    fig = plt.figure()
    colors = np.array(['#377eb8', "#ff7f00"]) # blue and orange
    # Making the markers super large so the image is sort of continuous & smooth.
    plt.scatter(data[:, 0], data[:, 1], s=20, color=colors[labels])
    plt.axis('off')
    plt.box(False)
    # Have to save image in order to convert to an array afterwards.
    fig.savefig(os.path.join(dirout, 'moons.png'), dpi=100)
    plt.close()
    return convert_image_to_array(fig)

  elif data == 'chelsea':
    data = skimage.data.chelsea()
    fig = plt.figure()
    plt.imshow(data, cmap=plt.cm.gray)
    plt.axis('off')
    plt.box(False)
    # Have to save image in order to convert to an array afterwards.
    fig.savefig(os.path.join(dirout, 'chelsea.png'), dpi=100)
    plt.close()
    return convert_image_to_array(fig)

  elif data == 'checkerboard':
    data = skimage.data.checkerboard()
    fig = plt.figure()
    plt.imshow(data, cmap=plt.cm.gray)
    plt.axis('off')
    plt.box(False)
    # Have to save image in order to convert to an array afterwards.
    fig.savefig(os.path.join(dirout, 'checkerboard.png'), dpi=100)
    plt.close()
    return convert_image_to_array(fig)

  elif data == 'labrador':
    return plt_im.imread(os.path.join(dirout, 'labrador.jpg'))

  elif data == 'mixture_of_2gaussians':
    def plot_target(probs):
      """Taken from necludov's repository and modified slightly."""
      fig = plt.figure()
      plt.imshow(probs, interpolation='bilinear', origin='lower')
      plt.xticks(np.arange(probs.shape[1])-0.5, labels=np.arange(probs.shape[1]))
      plt.yticks(np.arange(probs.shape[0])-0.5, labels=np.arange(probs.shape[0]))
      plt.axis('off')
      plt.box(False)
      return fig

    # TODO(loganesian): make the covariance adjustable???
    x, y = jax.numpy.arange(nx), jax.numpy.arange(ny)
    x_grid, y_grid = jax.numpy.meshgrid(x, y)
    stacked_grid = jax.numpy.stack([x_grid, y_grid], axis=2)
    probs = 0.5 * jax.scipy.stats.multivariate_normal.pdf(
                    stacked_grid,
                    # mean and cov
                    jax.numpy.array([nx/5, ny/1.4]),
                    jax.numpy.array([[1., 0.7], [0.7, 1.]])
                  ) +\
            0.5 * jax.scipy.stats.multivariate_normal.pdf(
                    stacked_grid,
                    # mean and cov
                    jax.numpy.array([nx/1.4, ny/5]),
                    jax.numpy.array([[1., 0.7], [0.7, 1.]])
                  )

    fig = plot_target(probs)
    # Have to save image in order to convert to an array afterwards.
    fig.savefig(os.path.join(dirout, 'mixture_of_2gaussians.png')) #, dpi=100)
    plt.close()
    return convert_image_to_array(fig)

  # Default
  return generate_density('labrador', num_samples, nx, ny)