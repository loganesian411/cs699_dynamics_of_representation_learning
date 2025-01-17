"""
Script to run Langevin Sampling to learn a posterior distribution.
"""

import argparse
import logging
import os
import pprint
import time

import dill
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

from utils.evaluations import get_loss_value
import utils.input_output as io
from utils.linear_algebra import FrequentDirectionAccountant
from utils.nn_manipulation import count_params, flatten_grads
from utils.reproducibility import set_seed
from utils.resnet import get_resnet

# "Fixed" hyperparameters
NUM_EPOCHS = 200
# In the resnet paper they train for ~90 epoch before reducing LR, then 45 and 45 epochs.
# We use 100-50-50 schedule here.
LR = 0.1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-D", "--debug", action='store_true')
    parser.add_argument("--seed", required=False, type=int, default=0)
    parser.add_argument(
        "--device", required=False, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--result_folder", "-r", required=True)
    parser.add_argument(
        "--mode", required=False, nargs="+", choices=["test", "train"], default=["test", "train"]
    )

    # model related arguments
    parser.add_argument("--statefile", "-s", required=False, default=None)
    parser.add_argument(
        "--model", required=True, choices=["resnet20", "resnet32", "resnet44", "resnet56"]
    )
    parser.add_argument("--remove_skip_connections", action="store_true", default=False)
    parser.add_argument(
        "--skip_bn_bias", action="store_true",
        help="whether to skip considering bias and batch norm params or not, Li et al do not consider bias and batch norm params"
    )

    parser.add_argument("--batch_size", required=False, type=int, default=128)
    parser.add_argument("--add_noise", required=False, type=float, default=0)
    parser.add_argument("--drop_pixels", required=False, type=float, default=0)
    parser.add_argument("--shuffle_pixels", required=False, type=int, default=0)
    parser.add_argument(
        "--save_strategy", required=False, nargs="+", choices=["epoch", "init"],
        default=["epoch", "init"]
    )

    args = parser.parse_args()

    # set up logging
    os.makedirs(f"{args.result_folder}/ckpt", exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger = logging.getLogger()
    if args.debug:
        logger.setLevel(logging.DEBUG)
    summary_writer = SummaryWriter(log_dir=args.result_folder)
    logger.info("Config:")
    logger.info(pprint.pformat(vars(args), indent=4))

    set_seed(args.seed)

    # get dataset
    train_loader, test_loader = io.get_dataloader(args.batch_size,
	                                               add_noise=args.add_noise,
	                                               drop_pixels=args.drop_pixels,
	                                               shuffle_pixels=args.shuffle_pixels)

    
    optimizer = sgld.SGLD(params, lr=1e-2, preconditioner=None,
                          precondition_decay_rate=None)

    # get model

    # model = get_resnet(args.model)(
    #     num_classes=10, remove_skip_connections=args.remove_skip_connections
    # )
    # model.to(args.device)
    # logger.info(f"using {args.model} with {count_params(model)} parameters")

    # logger.debug(model)

    # # use the same setup as He et al., 2015 (resnet)
    # optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)
    # total_params = count_params(model, skip_bn_bias=args.skip_bn_bias)
    # if args.statefile:
    #     checkpoint = torch.load(args.statefile)
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     start_epoch = checkpoint['epoch'] + 1
    #     loss = checkpoint['loss']
    #     fd = checkpoint['fd']
    #     fd_last_1 = checkpoint['fdl1']
    #     fd_last_10 = checkpoint['fdl10']
    #     step = checkpoint['step']
    # else: # args.statefile
    #     start_epoch = 0
    #     step = 0

    #     # we can try computing principal directions from some specific training rounds only
    #     fd = FrequentDirectionAccountant(k=2, l=10, n=total_params, device=args.device)
    #     # frequent direction for last 10 epoch
    #     fd_last_10 = FrequentDirectionAccountant(k=2, l=10, n=total_params, device=args.device)
    #     # frequent direction for last 1 epoch
    #     fd_last_1 = FrequentDirectionAccountant(k=2, l=10, n=total_params, device=args.device)

    #     if "init" in args.save_strategy:
    #         torch.save({'epoch': start_epoch, # init
    #                     'step': step, # init
    #                     'model_state_dict': model.state_dict(),
    #                     'optimizer_state_dict': optimizer.state_dict(),
    #                     'loss': None, # init
    #                     'fd': fd,
    #                     'fdl1': fd_last_1,
    #                     'fdl10': fd_last_10,
    #                     },
    #                     f"{args.result_folder}/ckpt/init_model.pt",
    #                     pickle_module=dill)

    # scheduler = torch.optim.lr_scheduler.LambdaLR(
    #     optimizer=optimizer, last_epoch=start_epoch-1,
    #     lr_lambda=lambda x: 1 if x < 100 else (0.1 if x < 150 else 0.01),
    # )

    # # training loop
    # # we pass flattened gradients to the FrequentDirectionAccountant before clearing the grad buffer
    # total_step = len(train_loader) * NUM_EPOCHS
    # direction_time = 0
    # for epoch in range(start_epoch, NUM_EPOCHS):
    #     model.train()
    #     for i, (images, labels) in enumerate(train_loader):
    #         images = images.to(args.device)
    #         labels = labels.to(args.device)

    #         # Forward pass
    #         outputs = model(images)
    #         loss = torch.nn.functional.cross_entropy(outputs, labels)

    #         # Backward and optimize
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #         # get gradient and send it to the accountant
    #         start = time.time()
    #         fd.update(flatten_grads(model, total_params, skip_bn_bias=args.skip_bn_bias))
    #         direction_time += time.time() - start

    #         if epoch >= NUM_EPOCHS - 10:
    #             fd_last_10.update(
    #                 flatten_grads(model, total_params, skip_bn_bias=args.skip_bn_bias)
    #             )
    #         if epoch >= NUM_EPOCHS - 1:
    #             fd_last_1.update(
    #                 flatten_grads(model, total_params, skip_bn_bias=args.skip_bn_bias)
    #             )

    #         summary_writer.add_scalar("train/loss", loss.item(), step)
    #         step += 1

    #         if step % 100 == 0:
    #             logger.info(
    #                 f"Epoch [{epoch}/{NUM_EPOCHS}], Step [{step}/{total_step}] Loss: {loss.item():.4f}"
    #             )

    #     scheduler.step()

    #     # Save the model checkpoint
    #     if "epoch" in args.save_strategy:
    #         torch.save({
    #             'epoch': epoch,
    #             'step': step,
    #             'model_state_dict': model.state_dict(),
    #             'optimizer_state_dict': optimizer.state_dict(),
    #             'fd': fd,
    #             'loss': loss,
    #             'fdl1': fd_last_1,
    #             'fdl10': fd_last_10,
    #             },
    #             f'{args.result_folder}/ckpt/{epoch + 1}_model.pt',
    #             pickle_module=dill
    #         )

    #     # TODO(loganesian): implement early stopping based on loss for last three.
    #     loss, acc = get_loss_value(model, test_loader, device=args.device)
    #     logger.info(f'Accuracy of the model on the test images: {100 * acc}%')
    #     summary_writer.add_scalar("test/acc", acc, step)
    #     summary_writer.add_scalar("test/loss", loss, step)

    # logger.info(f"Time to computer frequent directions {direction_time} s")

    # logger.info(f"fd was updated for {fd.step} steps")
    # logger.info(f"fd_last_10 was updated for {fd_last_10.step} steps")
    # logger.info(f"fd_last_1 was updated for {fd_last_1.step} steps")

    # # save the frequent_direction buffers and principal directions
    # buffer = fd.get_current_buffer()
    # directions = fd.get_current_directions()
    # directions = directions.cpu().data.numpy()

    # numpy.savez(
    #     f"{args.result_folder}/buffer.npy",
    #     buffer=buffer.cpu().data.numpy(), direction1=directions[0], direction2=directions[1]
    # )

    # # save the frequent_direction buffer
    # buffer = fd_last_10.get_current_buffer()
    # directions = fd_last_10.get_current_directions()
    # directions = directions.cpu().data.numpy()

    # numpy.savez(
    #     f"{args.result_folder}/buffer_last_10.npy",
    #     buffer=buffer.cpu().data.numpy(), direction1=directions[0], direction2=directions[1]
    # )

    # # save the frequent_direction buffer
    # buffer = fd_last_1.get_current_buffer()
    # directions = fd_last_1.get_current_directions()
    # directions = directions.cpu().data.numpy()

    # numpy.savez(
    #     f"{args.result_folder}/buffer_last_1.npy",
    #     buffer=buffer.cpu().data.numpy(), direction1=directions[0], direction2=directions[1]
    # )
