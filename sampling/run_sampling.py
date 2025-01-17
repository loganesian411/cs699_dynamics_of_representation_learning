"""Run HMC sampling with a user-specified energy function."""

import argparse
from collections import defaultdict
import data.toy_data as toy_data
import hamiltonian_mcmc.hamiltonian_mcmc as hmc
import jax
import matplotlib.pyplot as plt
import NPEET.npeet.entropy_estimators
import numpy as np
import os
from utils.density import continuous_energy_from_image, prepare_image, sample_from_image_density
from utils.metrics import get_discretized_tv_for_image_density
from tqdm import tqdm

_SNAPSHOT_FREQUENCY = 1

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--result_folder', type=str, default='./results')
  parser.add_argument('--data', type=str,
                      choices=['moons', 'checkerboard', 'rings', 'labrador',
                              'chelsea', 'mixture_of_2gaussians'],
                     default='labrador')
  parser.add_argument('--num_samples', type=int, default=10000)
  parser.add_argument('--K', type=int, default=100,
                      help='HMC chain length before injecting velocity noise.')
  parser.add_argument('--eps', type=float, default=0.3,
                      help='Learning rate for HMC.')
  parser.add_argument('--num_iter', type=int, default=15,
                      help='Number of outside iterations, each with K steps.')
  parser.add_argument('--precondition', type=bool, default=False,
                      help='Optional preconditioning for the HMC sampler.')
  parser.add_argument('--include_kinetic', type=bool, default=False,
                      help='Include kinetic energy in the HMC energy function.')
  parser.add_argument('--use_adaptive_eps', type=bool, default=False,
                      help='Include a really basic adaptive epsilon adjustment '\
                           'step for the HMC sampling.')
  parser.add_argument('--save_figures', type=bool, default=True)
  parser.add_argument('--density_initialization',
                      choices=['uniform', 'gaussian', 'constant'],
                      default='uniform',
                      help='Density to sample for the initial set of samples.')
  
  args = parser.parse_args()

  key = jax.random.PRNGKey(0)
  os.makedirs(f"{args.result_folder}", exist_ok=True)
  if args.save_figures:
    os.makedirs(f"{args.result_folder}/snapshot_ims", exist_ok=True)

  #### Load some image.
  img = toy_data.generate_density(args.data)

  # # Plot and visualize
  # fig = plt.figure(figsize=(10, 10))
  # ax = fig.add_subplot(1, 1, 1)
  # ax.imshow(img)
  # ax.set_title('density source image')
  # plt.show()

  if args.data == 'labrador':
    crop = (10, 710, 240, 940)
  elif args.data == 'checkerboard':
    crop = (250, 450, 300, 500)
  elif args.data == 'moons':
    crop = (50, 450, 100, 550)
  else:
    crop = None

  #### Convert to energy function.
  ## First we get discrete energy and density values
  density, energy = prepare_image(
    img, crop=crop, white_cutoff=225,
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
    # Gaussian distributed initial samples.
    subkey, key = jax.random.split(key)
    some_samples = sample_from_image_density(num_samples, density, subkey)
    mean = some_samples.mean(0)
    std = some_samples.std(0, ddof=1)
    subkey_x, key = jax.random.split(key)
    subkey_y, key = jax.random.split(key)
    X = jax.numpy.hstack([jax.random.normal(subkey_x, shape=(num_samples, 1),
                                            dtype='float64') * std[0] + mean[0],
                          jax.random.normal(subkey_y, shape=(num_samples, 1),
                                            dtype='float64') * std[0] + mean[1]
                          ])
  else: # Constant, start at center. Is bad -- do not use.
    # Start with all particles at the center of the image.
    X = jax.numpy.tile(jax.numpy.array([x_max // 2, y_max // 2], dtype='float'),
                       (num_samples, 1))

  if args.save_figures:
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(np.array(X)[:, 1], np.array(X)[:, 0], s=0.5, alpha=0.5)
    ax.imshow(density, alpha=0.3)
    ax.set_title('Samples snapshot initial')
    fig.savefig(f"{args.result_folder}/snapshot_ims/samples_iter_initial.png")
    plt.close()

  # Create energy fn and its grad
  xp = jax.numpy.arange(x_max)
  yp = jax.numpy.arange(y_max)
  zp = jax.numpy.array(density)

  # You may use fill value to enforce some boundary conditions or some other way to enforce boundary conditions
  energy_fn = lambda coord: continuous_energy_from_image(coord, xp, yp, zp, fill_value=0)
  energy_fn_grad = jax.grad(energy_fn)

  if args.precondition:
    # Fit a gaussian to a random sampling of the data.
    subkey, key = jax.random.split(key)
    init_samples = sample_from_image_density(num_samples, density, subkey)
    M = jax.numpy.cov(init_samples.T)
    # M = jax.numpy.diag(jax.numpy.diag(jax.numpy.cov(init_samples.T)))
  else:
    M = None
  
  # Metrics: KL divergence, total variation.
  metrics = defaultdict(list)
  for i in tqdm(range(args.num_iter)):
    subkey, key = jax.random.split(key)
    X, _ = hmc.hamiltonian_mcmc(X, energy_fn, args.K, eps=args.eps, key=subkey,
                                hamilton_ode=hmc.symplectic_integration, M=M,
                                include_kinetic=args.include_kinetic,
                                use_adaptive_eps=args.use_adaptive_eps)

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

  hyperparameters = {'ODE_solver': hmc.symplectic_integration, 'K': args.K,
                     'num_iter': args.num_iter, 'eps': args.eps,
                     'initialization': args.density_initialization}
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
    fig.savefig(f'{args.result_folder}/snapshot_ims/samples_iter_final.png')
    plt.close()
