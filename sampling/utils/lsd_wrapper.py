"""Wrapper around LSD training logic.

Effectively moved the main() of LSD.lsd_toy.py to a method with minor
modifications to run the same tests as the HMC sampler.
"""
from collections import defaultdict
import jax
import LSD.lsd_toy as lsd_core
import LSD.networks as networks
import LSD.utils as LSD_utils
from LSD.visualize_flow import visualize_transform
import matplotlib.pyplot as plt
import NPEET.npeet.entropy_estimators
import numpy as np
import os
import time
import torch
from tqdm import tqdm
import utils.density as density_utils
import utils.metrics as density_metrics

device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')

def sample_data(density, batch_size, key=jax.random.PRNGKey(0)):
  key, subkey = jax.random.split(key)
  x = density_utils.sample_from_image_density(batch_size, density, subkey)
  # Unfortunately JAX does not play nice with pytorch cuda yet!
  # https://github.com/google/jax/issues/1100
  x = torch.from_numpy(np.array(x)).type(torch.float32).to(device)
  return x

def lsd(density, energy, out_dir,
        exact_trace=True,  niters=10000, c_iters=5, # critic fn inner loops.
        lr=1e-3, weight_decay=0, critic_weight_decay=0, l2=10.,
        batch_size=1000, density_initialization='gaussian',
        test_nsteps=10, key=jax.random.PRNGKey(0),
        log_freq=100, save_freq=10000, viz_freq=100, logger=None):
  init_batch = sample_data(density, batch_size, key=key).requires_grad_()

  # Define a base distribution
  if density_initialization not in ['gaussian', 'uniform']:
    density_initialization = 'gaussian'

  if density_initialization == 'gaussian':
    # Fit a gaussian to the training data.  
    mu, std = init_batch.mean(0), init_batch.std(0)
    base_dist = torch.distributions.Normal(mu, std)
  elif density_initialization == 'uniform':
    # Fit a uniform to the training data.
    low, high = init_batch.min(); init_batch.max()
    base_dist = torch.distributions.Uniform(low, high)

  # Create critic and EBM neural nets.
  critic = networks.SmallMLP(2, n_out=2)
  net = networks.SmallMLP(2)
  ebm = lsd_core.EBM(net, base_dist)
  ebm.to(device)
  critic.to(device)

  # For sampling.
  init_fn = lambda: base_dist.sample_n(batch_size)
  cov = LSD_utils.cov(init_batch)

  # I think what's going on here if we define a HMC samples that initializes
  # samples with init_fn (typically Gaussian), uses the EBM to define the
  # the potential energy and internally has a kinetic energy that is Gauss
  # distributed with some sort of covariance options to precondition -- based on
  # the true data distribution (assuming based on the cov variables above). Then
  # the energy driving the sampling is potential + kinetic.
  sampler = LSD_utils.HMCSampler(ebm, .3, 5, init_fn, device=device,
                                 covariance_matrix=cov)

  if logger is not None:
    logger.info(ebm)
    logger.info(critic)

  # Optimizers.
  optimizer = torch.optim.Adam(ebm.parameters(), lr=lr, weight_decay=weight_decay,
                               betas=(.0, .999))
  critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr, betas=(.0, .999),
                                      weight_decay=critic_weight_decay)

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

    # compute dlogp(x)/dx
    logp_u = ebm(x)
    sq = lsd_core.keep_grad(logp_u.sum(), x)
    fx = critic(x)
    # compute (dlogp(x)/dx)^T * f(x)
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
    metrics['loss'].append(loss.item())
    metrics['l2_penalty'].append(l2_penalty.item())

    key, subkey = jax.random.split(key)
    p_samples = density_utils.sample_from_image_density(batch_size, density, subkey)
    q_samples = sampler.sample(test_nsteps)
    metrics['kldiv'].append(NPEET.npeet.entropy_estimators.kldiv(p_samples, q_samples))
    metrics['tv'].append(density_metrics.get_discretized_tv_for_image_density(
                         np.asarray(density), np.asarray(q_samples), bin_size=[7, 7]))

    if logger is not None and itr % log_freq == 0:
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
          'f': 3, 'eps': 5, 'base_dist': base_dist, 'covariance_matrix': cov
        },
      }
      training_results = {
        'state_dict': ebm.state_dict(),
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
      # Plot data
      plt.clf()
      key, subkey = jax.random.split(key)
      p_samples = density_utils.sample_from_image_density(batch_size, density, subkey)
      q_samples = sampler.sample(test_nsteps)

      ebm.cpu()

      x_enc = critic(x)
      xes = x_enc.detach().cpu().numpy()
      trans = xes.min()
      scale = xes.max() - xes.min()
      xes = (xes - trans) / scale * 8 - 4

      plt.figure(figsize=(4, 4))
      visualize_transform([p_samples, q_samples.detach().cpu().numpy(), xes],
                          ['data', 'model', 'embed'],
                          [ebm], ['model'], npts=batch_size)

      fig_filename = os.path.join(out_dir, 'figs', '{:04d}.png'.format(itr))
      LSD_utils.makedirs(os.path.dirname(fig_filename))
      plt.savefig(fig_filename)
      plt.close()

      ebm.to(device)
    end = time.time()

  if logger is not None: logger.info('Training has finished.')