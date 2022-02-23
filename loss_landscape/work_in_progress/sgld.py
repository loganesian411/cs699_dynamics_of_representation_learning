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

		self.loss_func = loss_func # unused?

		self.num_burn_in_steps = num_burn_in_steps
		self.posterior_samples = [] * (max_iter - num_burn_in_steps) # initialize
		self.posterior_lr_weights = [] * (max_iter - num_burn_in_steps) # initialize
		self.aved_wts = {}
		self.posterior_lr_weights = {}

		self.num_steps = 0
		self.model = model

	def _decay_func(self, lr_init, lr_final, max_iter):
		def lr_lambda(step):
			# 1/n decay for learning rate -> convergence of second order, but not first.
			# so it'll keep moving but the variance of movement will be limited.
			return lr_init * (1. / step) + lr_final
		return lr_lambda

	def _sample_posterior(self):
		return self.num_steps >= self.num_burn_in_steps

	def sample(self, train_loader, test_loader, summary_writer=None):
		if self.num_steps >= self.max_iter: return self.model

		self.model.train()
		for data, _ in train_loader:
			# outputs are -E(x) ~ posterior P(X | Y, theta)
			outputs = self.model(batch_data)

			self.optim.zero_grad()
			outputs.backward() # this is -d_E(x)/d_theta
			self.optim.step()

			self.num_steps += 1

			if self._sample_posterior(): # Store this posterior if in Langevin mode.
				def average_weights(wtsA, wtsB, w):
					aved_Wts = {}
					for k in wtsA.keys():
						aved_Wts[k] = wtsA + w * wtsB
					return aved_Wts

				self.posterior_samples[epoch - self.num_burn_in_steps] = self.model.state_dict()
				self.posterior_lr_weights[epoch - self.num_burn_in_steps] = self.lr_scheduler.get_lr()

				# self.aved_wts = average_weights(self.aved_wts, self.model.state_dict(), self.lr_scheduler.get_lr())
				# self.posterior_lr_weights += self.lr_scheduler.get_lr()

			self.lr_scheduler.step() # inside or outside fold?
		return self.model

	def state_dict(self):
		return self.__dict__

	def load_state_dict(self, state_dict):
		for k, v in state_dict.items():
			setattr(self, k, v)

	def predict(self, loader, device='cpu'):
		self.model.eval()
		losses, accuracies = [], []
		with torch.no_grad():
			for images, labels in loader:
				images = images.to(device)
				labels = labels.to(device)

				if not self._sample_posterior():
					# Not in Langevin dynamics yet.... so just do a predict + eval without
					# averaging.
          outputs = model(images)

          import ipdb; ipdb.set_trace()

          loss = torch.nn.functional.cross_entropy(outputs, labels, reduce=None).detach()
          losses.append(loss.reshape(-1))

          import ipdb; ipdb.set_trace()

          acc = (torch.argmax(outputs, dim=1) == labels).float().detach()
          accuracies.append(acc.reshape(-1))
          torch.nn.functional.cross_entropy
        
        else: # self._sample_posterior()
        	self.posterior_samples

      import ipdb; ipdb.set_trace()
      losses = torch.cat(losses, dim=0).mean().cpu().data.numpy()
      accuracies = torch.cat(accuracies, dim=0).mean().cpu().data.numpy()
      return losses, accuracies


# grand average will be E_{X | Y, theta}[X]
### scheduler nees to step at the end of the epoch not during batches

## compare this again swa averaging:
#### https://pytorch.org/docs/stable/optim.html#stochastic-weight-averaging
##### vs averaging predictions
##### vs weighted averaging of predictions