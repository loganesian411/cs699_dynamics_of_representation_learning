"""Hamiltonian MCMC module.

Following implementations are heavily based on the CS699 Feb 2
lecture demo code.
"""

import jax
import numpy as np

_SEED = 901

def euler_integration(x, v, E, eps=0.1):
  E_grad = jax.grad(E)
  x_new = x + eps * v
  v_new = v - eps * E_grad(x)
  return x_new, v_new

def symplectic_integration(x, v, E, eps=0.1):
  E_grad = jax.grad(E)
  v_half_new = v - eps / 2.0 * jax.numpy.apply_along_axis(E_grad, 1, x)
  x_new = x + eps * v_half_new
  v_new = v_half_new  - eps / 2.0 * jax.numpy.apply_along_axis(E_grad, 1, x_new)
  return x_new, v_new

def metropolis_hastings_adjustment(E, x_prev, x_new):
  E_prev = jax.numpy.apply_along_axis(E, 1, x_prev)
  E_new = jax.numpy.apply_along_axis(E, 1, x_new)
  reject_inds = np.exp(-(E_prev - E_new)) > 1
  # import ipdb; ipdb.set_trace()
  x_new.at[reject_inds, :].set(x_prev[reject_inds, :])
  return x_new

# K is the number of steps before selecting random velocity term.
def hamiltonian_mcmc(x, E, K, eps=0.1, key=jax.random.PRNGKey(_SEED),
                     hamilton_ode=symplectic_integration):
  v = jax.random.normal(key, shape=x.shape)
  x_orig = x
  for i in range(K):
    x, v = hamilton_ode(x, v, E, eps)
    x = metropolis_hastings_adjustment(E, x_orig, x)
    x_orig = x
  return x, v