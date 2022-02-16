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
	def __init__(self, model, loss_func, lr=1e-2, lr_final=1e-4, max_iter=int(1e3),
							 num_burn_in_steps=300, device='cpu', lr_scheduler=None):
		# TODO(loganesian): need to dynamically compute the number of steps after which to start
		# sampling posterior.

		self.lr = lr
		self.lr_final = lr_final
		self.max_iter = max_iter

		self.optim = SGLD(model.parameters(), lr=lr)
		self.lr_scheduler = lr_scheduler
		if self.lr_scheduler is None:
			self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer=self.optim,
        lr_lambda=_decay_func(lr_init, lr_final, max_iter),
	    )

		self.loss_func = loss_func
		self.num_burn_in_steps = num_burn_in_steps
		self.posterior_samples = [] * (max_iter - num_burn_in_steps) # initialize

		self.model = model

	def _decay_func(self, lr_init, lr_final, max_iter):
		def lr_lambda(epoch):
			return a * ((b + epoch) ** gamma)
		return lr_lambda

	def _sample_posterior(self, epoch):
		return epoch >= self.num_burn_in_steps

	def sample(self, epoch, batch_data, batch_labels):
		self.model.train()

		outputs = self.model(batch_data)
		logLL = torch.nn.functional.cross_entropy(output, batch_labels)

		self.optim.zero_grad()
		logLL.backward()
		self.optim.step()

		if self._sample_posterior():
			self.posterior_samples[epoch - self.num_burn_in_steps] = model.state_dict()

	def state_dict(self):
		return self.__dict__

	def load_state_dict(self, state_dict):
		for k, v in state_dict.items():
			setattr(self, k, v)

	### scheduler nees to step at the end of the epoch not during batches

## compare this again swa averaging:
#### https://pytorch.org/docs/stable/optim.html#stochastic-weight-averaging
##### vs averaging predictions
##### vs weighted averaging of predictions