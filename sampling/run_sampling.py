"""Samples a given density function with a specified method."""

import argparse
import hamiltonian_mcmc.hamiltonian_mcmc as hmc
import jax
import matplotlib.image as plt_im
import matplotlib.pyplot as plt
import numpy as np
import os

import NPEET.npeet.entropy_estimators
from utils.metrics import get_discretized_tv_for_image_density
from utils.density import continuous_energy_from_image, prepare_image, sample_from_image_density

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--result_folder', type=str, default='results')
  # parser.add_argument('--density', type=str, default="results")
  parser.add_argument('--sampling_method', type=str, default='hmc')
  parser.add_argument('--K', type=int, default=100)
  parser.add_argument('--eps', type=float, default=0.1)
  parser.add_argument('--num_iter', type=int, default=500)

  args = parser.parse_args()

  key = jax.random.PRNGKey(0)
  os.makedirs(f"{args.result_folder}", exist_ok=True)

  # load some image
  img = plt_im.imread('./data/labrador.jpg')

  # plot and visualize
  # fig = plt.figure(figsize=(10, 10))
  # ax = fig.add_subplot(1, 1, 1)
  # ax.imshow(img)
  # plt.show()

  # convert to energy function
  # first we get discrete energy and density values
  density, energy = prepare_image(
      img, crop=(10, 710, 240, 940), white_cutoff=225, gauss_sigma=3, background=0.01
  )

  # fig = plt.figure(figsize=(10, 10))
  # ax = fig.add_subplot(1, 1, 1)
  # ax.imshow(density)
  # ax.set_title('density')
  # plt.show()
  # fig.savefig(f"{args.result_folder}/labrador_density.png")

  # fig = plt.figure(figsize=(10, 10))
  # ax = fig.add_subplot(1, 1, 1)
  # ax.imshow(energy)
  # ax.set_title('energy')
  # plt.show()
  # fig.savefig(f"{args.result_folder}/labrador_energy.png")

  # create energy fn and its grad
  x_max, y_max = density.shape
  xp = jax.numpy.arange(x_max)
  yp = jax.numpy.arange(y_max)
  zp = jax.numpy.array(density)

  # You may use fill value to enforce some boundary conditions or some other way to enforce boundary conditions
  energy_fn = lambda coord: continuous_energy_from_image(coord, xp, yp, zp, fill_value=0)
  energy_fn_grad = jax.grad(energy_fn)

  # NOTE: JAX makes it easy to compute fn and its grad, but you can use any other framework.

  num_samples, K = 100000, args.K
  qx = jax.numpy.zeros(density.shape) # density estimate

  # uniformly spread out the initialized samples
  # subkey_x, key = jax.random.split(key)
  # subkey_y, key = jax.random.split(key)
  # X = jax.numpy.hstack([jax.random.choice(subkey_x, x_max, (num_samples, 1)),
  #                       jax.random.choice(subkey_y, y_max, (num_samples, 1))])
  # X = jax.numpy.asarray(X, dtype='float32')

  # Start with all particles at the center of the image.
  X = jax.numpy.tile(jax.numpy.array([x_max // 2, y_max // 2], dtype='float'), (num_samples, 1))

  fig = plt.figure(figsize=(10, 10))
  ax = fig.add_subplot(1, 1, 1)
  ax.scatter(np.array(X)[:, 1], np.array(X)[:, 0], s=0.5, alpha=0.5)
  ax.imshow(density, alpha=0.3)
  plt.show()
  
  for i in range(args.num_iter):
    subkey, key = jax.random.split(key)
    X, _ = hmc.hamiltonian_mcmc(X, energy_fn, K, eps=args.eps, key=subkey,
                                hamilton_ode=hmc.symplectic_integration)
    # import ipdb; ipdb.set_trace()
    # qx.at[X.at[:, 0].get(), X.at[:, 1].get()].add(1)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(np.array(X)[:, 1], np.array(X)[:, 0], s=0.5, alpha=0.5)
    ax.imshow(density, alpha=0.3)
    plt.show()
    import ipdb; ipdb.set_trace()

  # fig = plt.figure(figsize=(10, 10))
  # ax = fig.add_subplot(1, 1, 1)
  # ax.imshow(qx)
  # plt.show()
  # fig.savefig(f"{args.result_folder}/labrador_qx.png")

  # ig = plt.figure(figsize=(10, 10))
  # ax = fig.add_subplot(1, 1, 1)
  # ax.imshow(qx / qx.sum())
  # plt.show()
  # fig.savefig(f"{args.result_folder}/labrador_qx_normalized.png")

  # subkey, key = jax.random.split(key)
  # true_samples = sample_from_image_density(num_samples, density, subkey)
  # subkey, key = jax.random.split(key)
  # est_samples = sample_from_image_density(num_samples, qx, subkey)

  # fig = plt.figure(figsize=(10, 10))
  # ax = fig.add_subplot(1, 1, 1)
  # ax.scatter(np.array(samples)[:, 1], np.array(samples)[:, 0], s=0.5, alpha=0.5)
  # ax.imshow(density, alpha=0.3)
  # plt.show()
  # fig.savefig(f"{args.result_folder}/labrador_sampled_true.png")

  # fig = plt.figure(figsize=(10, 10))
  # ax = fig.add_subplot(1, 1, 1)
  # ax.scatter(np.array(est_samples)[:, 1], np.array(est_samples)[:, 0], s=0.5, alpha=0.5)
  # ax.imshow(density, alpha=0.3)
  # plt.show()
  # fig.savefig(f"{args.result_folder}/labrador_sampled_qx.png")

  # qx sample_from_image_density(num_samples, qx, key)
  # for each iterations:
  ## pass in the point of the current x
  ## let hmcmc run and then take that x and empiricially increment its
  ## coordinate in q(x)
  ## after iterations, normalize q(x) into a distribution and sample to
  ## generate the image.

