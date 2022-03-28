"""Learn a given distribution using with a specified method."""

import argparse
from collections import defaultdict
import data.toy_data as toy_data
import hamiltonian_mcmc.hamiltonian_mcmc as hmc
import jax
import matplotlib.image as plt_im
import matplotlib.pyplot as plt
import NPEET.npeet.entropy_estimators
import numpy as np
import os
from utils.density import continuous_energy_from_image, prepare_image, sample_from_image_density
import utils.lsd_wrapper as lsd_wrapper
from utils.metrics import get_discretized_tv_for_image_density
from tqdm import tqdm

_SNAPSHOT_FREQUENCY = 1

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--result_folder', type=str, default='results')
  parser.add_argument('--data', type=str,
                      choices=['moons', 'checkerboard', 'rings', 'labrador',
                              'chelsea', 'mixture_of_2gaussians'],
                     default='labrador')
  parser.add_argument('--algorithm', type=str,
                      choices=['hmc', 'lsd'],
                      default='hmc')
  parser.add_argument('--num_samples', type=int, default=10000)
  parser.add_argument('--K', type=int, default=100)
  parser.add_argument('--eps', type=float, default=0.3)
  parser.add_argument('--num_iter', type=int, default=15)
  parser.add_argument('--save_figures', type=bool, default=True)
  parser.add_argument('--density_initialization',
                      choices=['uniform', 'gaussian', 'constant'],
                      default='uniform')
  
  args = parser.parse_args()

  key = jax.random.PRNGKey(0)
  os.makedirs(f"{args.result_folder}", exist_ok=True)
  if args.save_figures:
    os.makedirs(f"{args.result_folder}/snapshot_ims", exist_ok=True)

  # #### Load some image.
  # img = plt_im.imread('./data/labrador.jpg')

  # # plot and visualize
  # fig = plt.figure(figsize=(10, 10))
  # ax = fig.add_subplot(1, 1, 1)
  # ax.imshow(img)
  # ax.set_title('density source image')
  # plt.show()

  img = toy_data.generate_density(args.data)

  if args.data == 'labrador':
    crop = (10, 710, 240, 940)
  else:
    crop = (50, 450, 150, 500)

  #### Convert to energy function.
  ## First we get discrete energy and density values
  density, energy = prepare_image(
    np.array(img), crop=crop, white_cutoff=225,
    gauss_sigma=3, background=0.01
  )

  if args.save_figures:
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(density)
    ax.set_title('density')
    fig.savefig(f"{args.result_folder}/{args.data}_density.png")

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(energy)
    ax.set_title('energy')
    fig.savefig(f"{args.result_folder}/{args.data}_energy.png")

  #### Initialize the samples.
  x_max, y_max = density.shape
  num_samples = args.num_samples

  if args.density_initialization == 'uniform':
    # Uniformly spread out the initialized samples
    subkey_x, key = jax.random.split(key)
    subkey_y, key = jax.random.split(key)
    X = jax.numpy.hstack([jax.random.uniform(subkey_x, minval=0, maxval=x_max,
                                            shape=(num_samples, 1),
                                            dtype='float64'),
                          jax.random.uniform(subkey_y, minval=0, maxval=y_max,
                                            shape=(num_samples, 1),
                                            dtype='float64')
                          ])
  elif args.density_initialization == 'gaussian':
    # Gaussian spread out the initialized samples
    subkey_x, key = jax.random.split(key)
    subkey_y, key = jax.random.split(key)
    X = jax.numpy.hstack([jax.random.normal(subkey_x, shape=(num_samples, 1),
                                            dtype='float64'),
                          jax.random.normal(subkey_y, shape=(num_samples, 1),
                                            dtype='float64')
                          ])
  else: # constant, start at center
    # Start with all particles at the center of the image.
    X = jax.numpy.tile(jax.numpy.array([x_max // 2, y_max // 2], dtype='float'),
                       (num_samples, 1))

  # Create energy fn and its grad
  xp = jax.numpy.arange(x_max)
  yp = jax.numpy.arange(y_max)
  zp = jax.numpy.array(density)

  # You may use fill value to enforce some boundary conditions or some other way to enforce boundary conditions
  energy_fn = lambda coord: continuous_energy_from_image(coord, xp, yp, zp, fill_value=0)
  energy_fn_grad = jax.grad(energy_fn)
  
  # Metrics: KL divergence, total variation, stein discrepancy (TODO).
  metrics = defaultdict(list)
  for i in tqdm(range(args.num_iter)):
  # for i in range(args.num_iter):
    subkey, key = jax.random.split(key)
    X, _ = hmc.hamiltonian_mcmc(X, energy_fn, args.K, eps=args.eps, key=subkey,
                                hamilton_ode=hmc.symplectic_integration)

    key, subkey = jax.random.split(key)
    true_samples = sample_from_image_density(num_samples, density, subkey)
    metrics['kldiv'].append(NPEET.npeet.entropy_estimators.kldiv(true_samples, X))
    metrics['tv'].append(get_discretized_tv_for_image_density(
                           np.asarray(density), np.asarray(X), bin_size=[7, 7]))
    
    if i % _SNAPSHOT_FREQUENCY == 0:
      print('Iteration: {0}, KL: {1}, TV: {2}'.format(i, metrics['kldiv'][-1],
                                                         metrics['tv'][-1]))
      if args.save_figures:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(np.array(X)[:, 1], np.array(X)[:, 0], s=0.5, alpha=0.5)
        ax.imshow(density, alpha=0.3)
        ax.set_title('Samples snapshot {0}'.format(i))
        fig.savefig(f"{args.result_folder}/snapshot_ims/samples_iter_{i}.png")
        plt.close()

  hyperparameters = {'ODE_solver': hmc.symplectic_integration, 'K': args.K, 'num_iter': args.num_iter,
                     'eps': args.eps, 'initialization': args.density_initialization}
  training_results = {'final_samples': X, 'density': density, 'energy': energy}
  np.savez(f'{args.result_folder}/final_res.npy',
    method_name='HMC', hyperparameter=hyperparameters,
    metrics=metrics, training_results=training_results)

  if args.save_figures:
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(np.array(X)[:, 1], np.array(X)[:, 0], s=0.5, alpha=0.5)
    ax.imshow(density, alpha=0.3)
    ax.set_title('Final samples distribution')
    fig.savefig(f'{args.result_folder}/snapshot_ims/samples_final.png')
    plt.close()

  ## TODO(loganesian): What if we treated samples as a density to sample from
  ## further using the continuous energy_from_image_function???
