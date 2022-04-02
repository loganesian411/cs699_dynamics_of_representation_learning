"""Hamiltonian MCMC module.

Following implementations are heavily based on the CS699 Feb 2
lecture demo code and Chen, T. et. al. (2014).
"""

import jax
import numpy as np
from tqdm import tqdm

_SEED = 901

def euler_integration(x, v, E, M=None, eps=0.1):
  """Performs first order Euler integration.

  Args:
    x: np.array of shape (num_samples, 2)
    v: np.array of shape (num_samples, 2)
    E: function handle. Can compute the energy associated with a particular x
      by calling E(x), where x is of shape (1, 2).
    M: np.array of size (x_dim, x_dim), that is (2, 2). Preconditioner "mass"
      matrix. Default is identity.
    eps: float. Learning rate used in the Euler integration. Default is 0.1.

  Returns:
    Tuple (x_new, v_new) at the next timestep. x_new and v_new are both of shape
    (num_samples, 2).
  """
  if M is None: M = jax.numpy.identity(x.shape[1])

  E_grad = jax.grad(E)
  x_new = x + eps * v @ jax.numpy.linalg.pinv(M) # check this 
  v_new = v - eps * jax.numpy.apply_along_axis(E_grad, 1, x)
  return x_new, v_new

def symplectic_integration(x, v, E, M=None, eps=0.1):
  """Performs symplectic (or leap-frog) integration.

  Args:
    x: np.array of shape (num_samples, 2)
    v: np.array of shape (num_samples, 2)
    E: function handle. Can compute the energy associated with a particular x
      by calling E(x), where x is of shape (1, 2)
    M: np.array of size (x_dim, x_dim), that is (2, 2). Preconditioner "mass"
      matrix. Default is identity.
    eps: float. Learning rate used in the Euler integration. Default 0.1.

  Returns:
    Tuple (x_new, v_new) at the next timestep.
  """
  if M is None: M = jax.numpy.identity(x.shape[1])

  E_grad = jax.grad(E)
  v_half_new = v - eps / 2.0 * jax.numpy.apply_along_axis(E_grad, 1, x)
  x_new = x + eps * v_half_new @ jax.numpy.linalg.pinv(M) # check this
  v_new = v_half_new  - eps / 2.0 * jax.numpy.apply_along_axis(E_grad, 1, x_new)
  return x_new, v_new

def metropolis_hastings_adjustment(E, x_prev, x_new,
                                   key=jax.random.PRNGKey(_SEED),
                                   v_prev=None, v_new=None, M=None):
  """Performs Metropolis-Hastings accept/reject step.

  If v_prev, v_new, and M are all provided, the kinetic energy term will also
  be included in the MHA comparison.

  Args:
    E: function handle. Will return the energy distribution associated with an
      input position x, of size (1, 2), computed as E(x).
    x_prev: np.array of shape (num_samples, 2). Current locations for each
      sample.
    x_new: np.array of shape (num_samples, 2). Proposed new locations for each
      sample.
    key: jax.random.PRNGKey() output. Used to seed jax random number generators.
    v_prev: np.array of shape (num_samples, 2). Current velocity values for each
      sample.
    v_new: np.array of shape (num_samples, 2). Proposed new velocity values for
      each sample.
    M: np.array of shape (2, 2). Mass term associated with the samples.

  Returns:
    (x_new, acceptance_rate) where x_new is the new state after performing MHA &
    acceptance_rate reflects what percentage of x_new is actually new.
  """
  E_prev = jax.numpy.apply_along_axis(E, 1, x_prev)
  E_new = jax.numpy.apply_along_axis(E, 1, x_new)

  # Optionally add kinetic energy.
  if v_prev is not None and v_new is not None and M is not None:
    v_mean = jax.numpy.zeros(M.shape[0])
    K_prev = -jax.scipy.stats.multivariate_normal.logpdf(v_prev, v_mean, M)
    E_prev += K_prev
    K_new = -jax.scipy.stats.multivariate_normal.logpdf(v_new, v_mean, M)
    E_new += K_new

  accept_inds = np.exp(E_prev - E_new) > jax.random.uniform(key, shape=E_new.shape)
  acceptance_rate = np.sum(accept_inds) / E_prev.shape[0]
  x_new.at[~accept_inds, :].set(x_prev[~accept_inds, :])
  return x_new, acceptance_rate

def hamiltonian_mcmc(x, E, K, eps=0.1, key=jax.random.PRNGKey(_SEED),
                     M=None, hamilton_ode=symplectic_integration,
                     lograte=10, include_kinetic=False, use_adaptive_eps=False):
  """Hamiltonian Monte Carlo Markov Chain sampler.

  Example usage:
    # Define energy function E.
    key = jax.random.PRNGKey(0)
    x_curr = jax.random.uniform(key, shape=(num_samples))
    K = 100
    for _ in range(num_iterations):
      # Internally hamiltonian_mcmc() will randomly initialize v and run the
      # sampler K iterations with MHA before returning the final values of x 
      # and v.
      key, subkey = jax.random.split(key)
      x_curr, v_curr = hamiltonian_mcmc(x_curr, E, K, key=subkey)
    # do something with resulting samples

  Args:
    x: np.array of shape (num_samples, 2). Starting position of the samples.
    E: function handle. Will return the energy distribution associated with an
      input position x, of size (1, 2), computed as E(x).
    K: int. The step size for the sampler, that is the number of iterations the
      sampler will run before returning the sample positions. This corresponds
      to the number of steps before random velocity noise is added to the system.
    eps: float. Learning rate to use in integration. Default is 0.1.
    key: jax.random.PRNGKey() output. Used to seed jax random number generators.
    M: np.array of size (x_dim, x_dim), that is (2, 2). Preconditioner "mass"
      matrix. Default is identity.
    hamilton_ode: function handle. The integration method used to propogate the
      state space. Default is symplectic_integration.
    lograte: int. The frequency at which to display debug logs. Default 10.
    include_kinetic: bool. Include the kinetic energy in the HMC energy function.
      Otherwise, it will only use the potential energy defined by E. Default False.
    use_adaptive_eps: bool. Use a basic adaptive epsilon adjustment algorithm
      based on MHA acceptance rate.

  Returns:
    Tuple (x, v) after K iterations of sampling. x and v are both of shape
    (num_samples, 2).
  """
  if M is None: M = jax.numpy.identity(x.shape[1])

  v = jax.random.multivariate_normal(key,
                                     jax.numpy.zeros(M.shape[0]), # zero mean
                                     M, # covariance
                                     shape=(x.shape[0],))

  x_orig, v_orig = x, v
  prev_acceptance_rate, accept_thresh = 0.5, 0.05
  for i in tqdm(range(K)):
    x, v = hamilton_ode(x, v, E, M=M, eps=eps)
    key, subkey = jax.random.split(key)
    if include_kinetic:
      x, acceptance_rate = metropolis_hastings_adjustment(E, x_orig, x, key=subkey,
        v_prev=v_orig, v_new=v, M=M)
    else:
      x, acceptance_rate = metropolis_hastings_adjustment(E, x_orig, x, key=subkey)

    if i % lograte == 0:
      print('\nAcceptance rate: ', acceptance_rate, '\n')
    
    # This is from LSD.utils.HMCSample.sample() implementation.
    if use_adaptive_eps:
      if acceptance_rate < 0.4:
        eps *= .67
        print('Decreasing eps.')
      elif acceptance_rate > 0.9:
        eps *= 1.33
        print('Increasing eps.')
    
    prev_acceptance_rate = acceptance_rate
    x_orig, v_orig = x, v

  return x, v