"""Stochastic Gradient Langevin sampling."""

import torch
import torch.optim as TorchOptimizer

# E(w) = - \Sigma_{i} log p(y_i | x_i, w) - log p(w)

class SGLD(TorchOptimizer):
	"""Naive SGLD."""

	def __init__(self, params, lr=1e-2,
							 preconditioner=None, precondition_decay_rate=None):
		# Custom settings for this optimizer.
		defaults = dict(lr=lr,
										preconditioner=preconditioner,
										precondition_decay_rate=precondition_decay_rate)

		super(SGLD, self).__init__(params, defaults)

	def step(self, closure=None):
		"""https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.step.html#torch.optim.Optimizer.step"""
		loss = None
		if closure is not None:
			loss = closure()

		for group in self.param_groups:
			for parameter in group['params']:

				if parameter.grad is None: continue # parameter doesn't store gradient.
				grad = parameter.grad.data

				## TODO(loganesian): num pseudo batches to get N/n???

				## If we wanted momentum, weight decay or other niceties we would add it
				## here. But for not just doing the normal step.
				parameter.data.add_(grad, alpha=-group['lr'])
			  
			  ## Langevin noise
			  noise_std = torch.Tensor(np.sqrt(2*group['lr']))
			  noise = torch.randn_like(parameter.data) * noise_std
			  parameter.data.add_(noise)

class LangevinDynamics():
	def __init__(self, x, energy_func, lr=1e-2, lr_final=1e-4, max_iter=1e3,
							 device='cpu', lr_scheduler=None):
		self.optim = SGLD(params, lr=lr)

		self.lr_scheduler = lr_scheduler
		if self.lr_scheduler is None:
			self.lr_scheduler = self.decay_fn(lr_init, lr_final, max_iter)

	def _decay_func(self, lr_init, lr_final, max_iter):
		pass

	def _sample_posterior(self):
		pass

	def posterior_samples(self):
		pass

	def sample(self, epoch):
		self.

	### scheduler nees to step at the end of the epoch not during batches

## compare this again swa averaging:
#### https://pytorch.org/docs/stable/optim.html#stochastic-weight-averaging
##### vs averaging predictions
##### vs weighted averaging of predictions