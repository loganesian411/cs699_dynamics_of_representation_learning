"""Wrapper around LSD training logic.

Effectively moved the main() of LSD.lsd_toy.py to a method with minor
modifications to run the same tests as the HMC sampler.
"""

import LSD.lsd_toy as lsd_core
import LSD.networks as networks
import LSD.utils as LSD_utils
import torch
import utils.density as density
# import continuous_energy_from_image, prepare_image, sample_from_image_density

device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')

# def sample_data_orig(args, batch_size):
#   x = toy_data.inf_train_gen(args.data, batch_size=batch_size)
#   x = torch.from_numpy(x).type(torch.float32).to(device)
#   return x

def sample_data(density, num_samples, key=jax.random.PRNGKey(0)):
  key, subkey = jax.random.split(key)
  x = density.sample_from_image_density(num_samples, density, subkey)
  # Unfortunately JAX does not play nice with pytorch cuda yet!
  # https://github.com/google/jax/issues/1100
  x = torch.from_numpy(np.array(x)).type(torch.float32).to(device)
  return x

def lsd(density, niters, num_samples,
        exact_trace=True,
        lr=1e-3, weight_decay=0, critic_weight_decay=0,
        init_size=1000, test_batch_size=1000,
        key=jax.random.PRNGKey(0), logger=None):
  # TODO(loganesian): why???????!?!!?!?!
	# Fit a gaussian to the training data.
  init_batch = sample_data(density, init_size, key=key).requires_grad_()
  mu, std = init_batch.mean(0), init_batch.std(0)
  base_dist = distributions.Normal(mu, std)

  # Create critic and EBM neural nets.
  critic = networks.SmallMLP(2, n_out=2)
  net = networks.SmallMLP(2)

  ebm = EBM(net, base_dist)
  ebm.to(device)
  critic.to(device)

  # for sampling
  init_fn = lambda: base_dist.sample_n(test_batch_size)
  cov = LSD_utils.cov(init_batch)
  # TODO(loganesian): WHAT IS THIS FOR??????????????
  sampler = LSD_utils.HMCSampler(ebm, .3, 5, init_fn, device=device,
                                 covariance_matrix=cov)

  if logger is not None:
    logger.info(ebm)
    logger.info(critic)

  # Optimizers.
  optimizer = optim.Adam(ebm.parameters(), lr=lr, weight_decay=weight_decay,
                         betas=(.0, .999))
  critic_optimizer = optim.Adam(critic.parameters(), lr=lr, betas=(.0, .999),
                                weight_decay=critic_weight_decay)

  time_meter = utils.RunningAverageMeter(0.98)
  loss_meter = utils.RunningAverageMeter(0.98)

  ebm.train()
  end = time.time()
  for itr in range(niters):

    optimizer.zero_grad()
    critic_optimizer.zero_grad()

    x = sample_data(density, num_samples, key=key)
    x.requires_grad_()

    # compute dlogp(x)/dx
    logp_u = ebm(x)
    sq = keep_grad(logp_u.sum(), x)
    fx = critic(x)
    # compute (dlogp(x)/dx)^T * f(x)
    sq_fx = (sq * fx).sum(-1)

    # Compute/estimate Tr(df/dx)
    if exact_trace:
        tr_dfdx = LSD_toy.exact_jacobian_trace(fx, x)
    else:
        tr_dfdx = LSD_toy.approx_jacobian_trace(fx, x)

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