"""Wrapper around LSD training logic.

Effectively moved the main() of LSD.lsd_toy.py to a method.
"""

import LSD.lsd_toy as lsd_core
import torch

def lsd():
	# fit a gaussian to the training data
  init_size = 1000
  init_batch = sample_data(args, init_size).requires_grad_()
  mu, std = init_batch.mean(0), init_batch.std(0)
  base_dist = distributions.Normal(mu, std)

  # neural netz
  critic = networks.SmallMLP(2, n_out=2)
  net = networks.SmallMLP(2)

  ebm = EBM(net, base_dist if args.base_dist else None)
  ebm.to(device)
  critic.to(device)

  # for sampling
  init_fn = lambda: base_dist.sample_n(args.test_batch_size)
  cov = utils.cov(init_batch)
  sampler = HMCSampler(ebm, .3, 5, init_fn, device=device, covariance_matrix=cov)

  logger.info(ebm)
  logger.info(critic)

  # optimizers
  optimizer = optim.Adam(ebm.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(.0, .999))
  critic_optimizer = optim.Adam(critic.parameters(), lr=args.lr, betas=(.0, .999),
                                weight_decay=args.critic_weight_decay)

  time_meter = utils.RunningAverageMeter(0.98)
  loss_meter = utils.RunningAverageMeter(0.98)

  ebm.train()
  end = time.time()
  for itr in range(args.niters):

      optimizer.zero_grad()
      critic_optimizer.zero_grad()

      x = sample_data(args, args.batch_size)
      x.requires_grad_()

      if args.mode == "lsd":
          # our method

          # compute dlogp(x)/dx
          logp_u = ebm(x)
          sq = keep_grad(logp_u.sum(), x)
          fx = critic(x)
          # compute (dlogp(x)/dx)^T * f(x)
          sq_fx = (sq * fx).sum(-1)

          # compute/estimate Tr(df/dx)
          if args.exact_trace:
              tr_dfdx = exact_jacobian_trace(fx, x)
          else:
              tr_dfdx = approx_jacobian_trace(fx, x)

          stats = (sq_fx + tr_dfdx)
          loss = stats.mean()  # estimate of S(p, q)
          l2_penalty = (fx * fx).sum(1).mean() * args.l2  # penalty to enforce f \in F

          # adversarial!
          if args.c_iters > 0 and itr % (args.c_iters + 1) != 0:
              (-1. * loss + l2_penalty).backward()
              critic_optimizer.step()
          else:
              loss.backward()
              optimizer.step()

      elif args.mode == "sm":
          # score matching for reference
          fx = ebm(x)
          dfdx = torch.autograd.grad(fx.sum(), x, retain_graph=True, create_graph=True)[0]
          eps = torch.randn_like(dfdx)  # use hutchinson here as well
          epsH = torch.autograd.grad(dfdx, x, grad_outputs=eps, create_graph=True, retain_graph=True)[0]

          trH = (epsH * eps).sum(1)
          norm_s = (dfdx * dfdx).sum(1)

          loss = (trH + .5 * norm_s).mean()
          loss.backward()
          optimizer.step()
      else:
          assert False

      loss_meter.update(loss.item())
      time_meter.update(time.time() - end)

      if itr % args.log_freq == 0:
          log_message = (
              'Iter {:04d} | Time {:.4f}({:.4f}) | Loss {:.4f}({:.4f})'.format(
                  itr, time_meter.val, time_meter.avg, loss_meter.val, loss_meter.avg
              )
          )
          logger.info(log_message)

      if itr % args.save_freq == 0 or itr == args.niters:
          ebm.cpu()
          utils.makedirs(args.save)
          torch.save({
              'args': args,
              'state_dict': ebm.state_dict(),
          }, os.path.join(args.save, 'checkpt.pth'))
          ebm.to(device)

      if itr % args.viz_freq == 0:
          # plot dat
          plt.clf()
          npts = 100
          p_samples = toy_data.inf_train_gen(args.data, batch_size=npts ** 2)
          q_samples = sampler.sample(args.n_steps)

          ebm.cpu()

          x_enc = critic(x)
          xes = x_enc.detach().cpu().numpy()
          trans = xes.min()
          scale = xes.max() - xes.min()
          xes = (xes - trans) / scale * 8 - 4

          plt.figure(figsize=(4, 4))
          visualize_transform([p_samples, q_samples.detach().cpu().numpy(), xes], ["data", "model", "embed"],
                              [ebm], ["model"], npts=npts)

          fig_filename = os.path.join(args.save, 'figs', '{:04d}.png'.format(itr))
          utils.makedirs(os.path.dirname(fig_filename))
          plt.savefig(fig_filename)
          plt.close()

          ebm.to(device)
      end = time.time()

  logger.info('Training has finished, can I get a yeet?')