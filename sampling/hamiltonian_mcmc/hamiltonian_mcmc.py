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
  v_half_new = v - eps / 2.0 * E_grad(x)
  x_new = x + eps * v_half_new
  v_new = v_half_new  - eps / 2.0 * E_grad(x_new)
  return x_new, v_new

def metropolis_hastings_adjustment(E, x_prev, x_new):
	if np.exp(-(E(x_prev) - E(x_new))) > 1:
		return x_prev
	return x_new

# K is the number of steps before selecting random velocity term.
def hamiltonian_mcmc(x, v, E, K, eps=0.1, key=jax.random.PRNGKey(_SEED),
										 hamilton_ode=symplectic_integration):
	subkey, key = jax.random.split(key)
  v = jax.random.normal(subkey, shape=x.shape)
  x_orig = x
  for i in range(K):
  	x, v = hamilton_ode(x, v, E, eps)
  	x = metropolis_hastings_adjustment(E, x_orig, x)
  	x_orig = x
  return x, v