"""Wrapper around LSD training logic found in LSD.lsd_toy.py.

Effectively moved the main() of LSD.lsd_toy.py to a method with minor
modifications to run the same tests as the HMC sampler.
"""
from collections import defaultdict
import hamiltonian_mcmc.hamiltonian_mcmc_forebm as hmc
import jax
import LSD.lsd_toy as lsd_core
import LSD.networks as networks
import LSD.toy_data as LSD_toy_data
import LSD.utils as LSD_utils
from LSD.visualize_flow import visualize_transform
import logging
import matplotlib.pyplot as plt
import NPEET.npeet.entropy_estimators
import numpy as np
import os
import time
import torch
from tqdm import tqdm
import utils.density as density_utils
import utils.metrics as density_metrics
import utils.viz_utils as viz_utils

device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')

def sample_data(density, batch_size, key=jax.random.PRNGKey(0), demean=False):
  """Samples the given density image.
  
  Args:
    density: np.array. Density image to sample.
    batch_size: int. Number of samples to sample.
    key: int. Optional. jax PRNGKey
    demean: bool. Default False. Demean the samples.

  Returns:
    Samples as (batch_size, 2) np.array.
  """
  key, subkey = jax.random.split(key)
  x = density_utils.sample_from_image_density(batch_size, density, subkey)
  # Unfortunately JAX does not play nice with pytorch cuda yet!
  # https://github.com/google/jax/issues/1100
  x = torch.from_numpy(np.array(x)).type(torch.float32).to(device)
  if demean: x -= x.mean(0)
  return x

def sample_data_orig(data, batch_size, demean=False):
  """Sample from given dataset.

  Args:
    data: string. One of the datasets in LSD.toy_data.
    batch_size: int. Number of samples to sample.
    demean: bool. Default False. Demean the samples.

  Returns:
    Samples as (batch_size, 2) np.array.
  """
  x = LSD_toy_data.inf_train_gen(data, batch_size=batch_size)
  x = torch.from_numpy(x).type(torch.float32).to(device)
  if demean: x -= x.mean(0)
  return x

def lsd(density, energy, out_dir,
        exact_trace=True, niters=10000, c_iters=5, # critic fn inner loops.
        lr=1e-3, weight_decay=0, critic_weight_decay=0, l2=10.,
        batch_size=5000, test_nsteps=10, K=5, eps=0.3,
        density_initialization='gaussian', precondition_HMC=False,
        key=jax.random.PRNGKey(0), log_freq=100, save_freq=10000, viz_freq=100,
        logger=None, debug_mode=False):
  """Runs the default LSD+EBM training procedure as defined in LSD.lsd_toy.py.

  Args:
    density: np.ndarray. True density to sample.
    energy: np.ndarray. Energy corresponding to true density.
    out_dir: string. Directory to save images and training/evaluation outputs to.
    exact_trace: bool. See the original Grathwohl et al. paper for the difference
      between trace approximation or exact computation.
    niters: int. Training iterations.
    c_iters: int. Inner loops for training the critic function network.

    ## Optimizer parameters for critic and EBM networks. See torch Adam optimizer
    ## documentation.
    lr, (critic_)weight_decay

    ## HMC sampler parameters
    test_nsteps: int. Number of iterations the sampler will have internally
    K: int. Chain length before random velocity injection
    eps: float. Learning rate.
    density_initialization: str. Specify distribution for initializing samples.
      Default "gaussian". Support: "gaussian" and "uniform".
    precondition_HMC: bool. Use the covariance for the initial set of samples as
      a preconditioning mass matrix in HMC sampling.

    key: int. Jax PRNGKey.

    ## Logging and snapshotting frequency specification.
    log_freq: frequency at which intermittent metrics are computed
    save_freq: frequency at which data is saved out (i.e., written to disk)
    viz_freq: frequency at which intermittent learned density is visualized by
      sampling the learned energy
    logger: optional python logging object.
    debug_mode: bool. Will force creation of logger object if not provided.
  """
  if logger is None and debug_mode:
    logger = LSD_utils.get_logger(logpath=os.path.join(out_dir, 'logs'),
                                  filepath=os.path.abspath(__file__))

  init_batch = sample_data(density, batch_size, key=key).requires_grad_()
  mu, std = init_batch.mean(0), init_batch.std(0)
  low = init_batch.min(0).values
  high = init_batch.max(0).values

  extent = [low[0].detach().numpy(), high[0].detach().numpy(),
            low[1].detach().numpy(), high[1].detach().numpy()]
  fig = viz_utils.plot_samples(init_batch.detach().numpy(), color='g',
                               density=density, extent=extent)
  fig_filename = os.path.join(out_dir, 'figs', 'starter.png')
  LSD_utils.makedirs(os.path.dirname(fig_filename))
  plt.savefig(fig_filename)
  plt.close()

  # Define a base distribution
  if density_initialization not in ['gaussian', 'uniform']:
    density_initialization = 'gaussian'

  if density_initialization == 'gaussian':
    # Fit a gaussian to the training data.  
    base_dist = torch.distributions.Normal(mu, std)
  elif density_initialization == 'uniform':
    # Fit a uniform to the training data.
    base_dist = torch.distributions.Uniform(low, high)

  # Create critic and EBM neural nets.
  critic = networks.SmallMLP(2, n_out=2)
  net = networks.SmallMLP(2)
  if density_initialization != 'uniform':
    ebm = lsd_core.EBM(net, base_dist)
  else:
    ebm = lsd_core.EBM(net)
  
  ebm.to(device)
  critic.to(device)

  # For sampling.
  init_fn = lambda: base_dist.sample_n(batch_size)
  if precondition_HMC:
    cov = LSD_utils.cov(init_batch)
  else:
    cov = torch.eye(init_batch.shape[1])

  # I think what's going on here if we define a HMC samples that initializes
  # samples with init_fn (typically Gaussian), uses the EBM to define the
  # the potential energy and internally has a kinetic energy that is Gauss
  # distributed with some sort of covariance options to precondition -- based on
  # the true data distribution (assuming based on the cov variables above). Then
  # the energy driving the sampling is potential + kinetic.
  sampler = LSD_utils.HMCSampler(ebm, eps, K, init_fn, device=device,
                                 covariance_matrix=cov)

  if logger is not None:
    logger.info(ebm)
    logger.info(critic)

  # Optimizers.
  optimizer = torch.optim.Adam(ebm.parameters(), lr=lr,
                               weight_decay=weight_decay, betas=(.0, .999))
  critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr,
                                      weight_decay=critic_weight_decay,
                                      betas=(.0, .999))

  time_meter = LSD_utils.RunningAverageMeter(0.98)
  loss_meter = LSD_utils.RunningAverageMeter(0.98)
  metrics = defaultdict(list)

  ebm.train()
  end = time.time()
  for itr in tqdm(range(niters)):

    optimizer.zero_grad()
    critic_optimizer.zero_grad()

    x = sample_data(density, batch_size, key=key)
    x.requires_grad_()

    # Compute dlogp(x)/dx
    logp_u = ebm(x)

    sq = lsd_core.keep_grad(logp_u.sum(), x)
    fx = critic(x)

    # Compute (dlogp(x)/dx)^T * f(x)
    sq_fx = (sq * fx).sum(-1)

    # Compute/estimate Tr(df/dx)
    if exact_trace:
      tr_dfdx = lsd_core.exact_jacobian_trace(fx, x)
    else:
      tr_dfdx = lsd_core.approx_jacobian_trace(fx, x)

    stats = (sq_fx + tr_dfdx)
    loss = stats.mean()  # estimate of S(p, q)
    l2_penalty = (fx * fx).sum(1).mean() * l2  # Penalty to enforce f \in F.

    # Adversarial: update the critic function more frequently before updating
    # ebm.
    if c_iters > 0 and itr % (c_iters + 1) != 0:
      (-1. * loss + l2_penalty).backward()
      critic_optimizer.step()
    else:
      loss.backward()
      optimizer.step()

    loss_meter.update(loss.item())
    time_meter.update(time.time() - end)

    if itr % log_freq == 0:
      metrics['loss'].append(loss.item())
      metrics['l2_penalty'].append(l2_penalty.item())

      key, subkey = jax.random.split(key)
      p_samples = sample_data(density, batch_size, subkey)
      q_samples = sampler.sample(test_nsteps)
      metrics['kldiv'].append(NPEET.npeet.entropy_estimators.kldiv(p_samples, q_samples))
      metrics['tv'].append(density_metrics.get_discretized_tv_for_image_density(
                           np.asarray(density), np.asarray(q_samples), bin_size=[7, 7]))

      if logger is not None:
        log_message = (
            'Iter {:04d} | Time {:.4f}({:.4f}) | Loss {:.4f}({:.4f})'.format(
                itr, time_meter.val, time_meter.avg, loss_meter.val, loss_meter.avg
            )
        )
        logger.info(log_message)

    if itr % save_freq == 0 or itr == niters:
      ebm.cpu()
      LSD_utils.makedirs(out_dir)
      settings = {
        'niters': niters, 'citers': c_iters, 'test_nsteps': test_nsteps,
        'l2': l2, 'lr': lr, 'weight_decay': weight_decay,
        'critic_weight_decay': critic_weight_decay, 'batch_size': batch_size,
        'density_initialization': density_initialization,
        'sampler_settings': {
          'K': K, 'eps': eps, 'base_dist': base_dist, 'covariance_matrix': cov
        },
      }
      training_results = {
        'state_dict': ebm.state_dict(),
        'critic_dict': critic.state_dict(),
        'density': density,
        'energy': energy,
      }
      np.savez(os.path.join(out_dir, 'LSD_ckpt_{0}'.format(itr)),
          method_name='lsd_ebm',
          hyperparameters=settings,
          metrics=metrics,
          training_results=training_results,
      )
      ebm.to(device)

    if itr % viz_freq == 0:
      # Plot data.
      plt.clf()
      key, subkey = jax.random.split(key)
      p_samples = sample_data(density, batch_size, subkey)
      q_samples = sampler.sample(test_nsteps)

      ## Code to swap out with our hmc sampler. make sure dependencies are
      ## imported.
      #
      # q_samples = jax.numpy.array(init_fn().detach().numpy())
      # energy_fn = lambda x: jax.numpy.array(ebm(torch.tensor(np.array(x))).detach().numpy())
      # def grad_fn(x):
      #   x = torch.tensor(np.array(x)).requires_grad_()
      #   grad = torch.autograd.grad(-ebm(x).sum(), x, create_graph=True)[0]
      #   return jax.numpy.array(grad.detach().numpy())

      # for _ in tqdm(range(test_nsteps)):
      #   key, subkey = jax.random.split(key)
      #   q_samples, _ = hmc.hamiltonian_mcmc(q_samples, energy_fn, K,
      #                                       eps=eps, key=subkey,
      #                                       M=cov.detach().numpy(),
      #                                       hamilton_ode=hmc.symplectic_integration,
      #                                       include_kinetic=False,
      #                                       use_adaptive_eps=False,
      #                                       grad_func=grad_fn,
      #                                       apply_along_axis=False)

      ## Visualization code used by Grathwohl et al.
      #
      # ebm.cpu()
      # x_enc = critic(x)
      # xes = x_enc.detach().cpu().numpy()
      # trans = xes.min()
      # scale = xes.max() - xes.min()
      # xes = (xes - trans) / scale * 8 - 4
      #
      # visualize_transform([p_samples, q_samples.detach().cpu().numpy(), xes],
      #                     ['data', 'model', 'embed'],
      #                     [ebm], ['model'], npts=batch_size)
      # ebm.to(device)

      fig = viz_utils.plot_samples(p_samples, density=density, extent=extent,
                                   color='g')
      fig = viz_utils.plot_samples(q_samples.detach().numpy(), color='r', fig=fig)
      fig_filename = os.path.join(out_dir, 'figs', '{:04d}.png'.format(itr))
      LSD_utils.makedirs(os.path.dirname(fig_filename))
      plt.savefig(fig_filename)
      plt.close()

    end = time.time()

  if logger is not None: logger.info('Training has finished.')