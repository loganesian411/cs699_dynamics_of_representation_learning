"""Hamiltonian MCMC module.

Following implementations are heavily based on the CS699 Feb 2
lecture demo code.
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
                                   key=jax.random.PRNGKey(_SEED)):
  """Performs Metropolis-Hastings accept/reject step.

  Args:
    E: function handle. Will return the energy distribution associated with an
      input position x, of size (1, 2), computed as E(x).
    x_prev: np.array of shape (num_samples, 2). Current locations for each
      sample.
    x_new: np.array of shape (num_samples, 2). Proposed new locations for each
      sample.
    key: jax.random.PRNGKey() output. Used to seed jax random number generators.
  """
  E_prev = jax.numpy.apply_along_axis(E, 1, x_prev)
  E_new = jax.numpy.apply_along_axis(E, 1, x_new)
  reject_inds = np.exp(-(E_prev - E_new)) > jax.random.uniform(key, shape=E_new.shape)
  print('Acceptance rate: ' 1 - np.sum(reject_inds) / E_prev.shape[0])
  x_new.at[reject_inds, :].set(x_prev[reject_inds, :])
  return x_new

# K is the number of steps before selecting random velocity term.
def hamiltonian_mcmc(x, E, K, eps=0.1, key=jax.random.PRNGKey(_SEED),
                     M=None, hamilton_ode=symplectic_integration):
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
      sampler will run before returning the sample positions.
    eps: float. Learning rate to use in integration. Default is 0.1.
    key: jax.random.PRNGKey() output. Used to seed jax random number generators.
    M: np.array of size (x_dim, x_dim), that is (2, 2). Preconditioner "mass"
      matrix. Default is identity.
    hamilton_ode: function handle. The integration method used to propogate the
      state space. Default is symplectic_integration.

  Returns:
    Tuple (x, v) after K iterations of sampling. x and v are both of shape
    (num_samples, 2).
  """
  if M is None: M = jax.numpy.identity(x.shape[1])

  v = jax.random.normal(key, shape=x.shape)
  x_orig = x
  for i in tqdm(range(K)):
    x, v = hamilton_ode(x, v, E, M=M, eps=eps)
    key, subkey = jax.random.split(key)
    x = metropolis_hastings_adjustment(E, x_orig, x, key=subkey)
    x_orig = x
  return x, v