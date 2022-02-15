"""Compute accuracy curves for train and test datasets."""

import argparse
import dill
import logging
import numpy as np
import os
import sys
import torch
from tqdm import tqdm

from train import get_dataloader
from utils.evaluations import get_loss_value
from utils.nn_manipulation import count_params
from utils.reproducibility import set_seed
from utils.resnet import get_resnet

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-D", "--debug", action='store_true')
    parser.add_argument("--seed", required=False, type=int, default=0)
    parser.add_argument("--device", required=False,
    	default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch_size", required=False, type=int, default=1000)

    parser.add_argument("--result_folder", "-r", required=True)
    parser.add_argument("--accuracy_file", type=str, required=True,
        help="filename to store evaluation results"
    )

    parser.add_argument("--statefile_folder", "-s", required=True, default=None)
    
    parser.add_argument("--model", required=True,
    	choices=["resnet20", "resnet32", "resnet44", "resnet56"]
    )
    parser.add_argument("--remove_skip_connections", action="store_true",
									    	default=False)
    
    args = parser.parse_args()

    # set up logging
    os.makedirs(f"{args.result_folder}", exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger = logging.getLogger()
    if args.debug:
        logger.setLevel(logging.DEBUG)
    set_seed(args.seed)

    if os.path.exists(f"{args.result_folder}/{args.accuracy_file}"):
        logger.error(f"{args.accuracy_file} exists, so we will exit")
        sys.exit()

    # get dataset
    # using training dataset and only 10000 examples for faster evaluation
    train_loader, test_loader = get_dataloader(args.batch_size, train_size=10000)

    # setup model class
    model = get_resnet(args.model)(
        num_classes=10, remove_skip_connections=args.remove_skip_connections
    )
    model.to(args.device)
    total_params = count_params(model)
    logger.info(f"using {args.model} with {total_params} parameters")

    # load all model checkpoints.
    state_files = [f"{args.statefile_folder}/init_model.pt"]
    all_chkpts = [fname for fname in os.listdir(args.statefile_folder) if 'init' not in fname]
    all_chkpts = sorted(all_chkpts, key=lambda v: int(v.split("_")[0]))
    for fname in all_chkpts:
      state_files.append(os.path.join(args.statefile_folder, fname))

    train_losses, test_losses = np.zeros(len(state_files)), np.zeros(len(state_files))
    train_accuracies, test_accuracies = np.zeros(len(state_files)), np.zeros(len(state_files))
    with tqdm(total=len(state_files)) as pbar:
    	for state_ind, state_file in enumerate(state_files):
		    logger.info(f"Loading model from {state_file}")
		    state_dict = torch.load(state_file, pickle_module=dill, map_location=args.device)

		    model.load_state_dict(state_dict['model_state_dict'])
		    train_losses[state_ind], train_accuracies[state_ind] = get_loss_value(
																				    	model, train_loader, args.device)
		    test_losses[state_ind], test_accuracies[state_ind] = get_loss_value(
																				    	model, test_loader, args.device)

		    pbar.set_description(f"ckpt:{state_ind}")
		    pbar.update(1)

    # save losses and accuracies evaluations
    logger.info("Saving results")
    np.savez(
        f"{args.result_folder}/{args.accuracy_file}",
        train_losses=train_losses, train_accuracies=train_accuracies,
        test_losses=test_losses, test_accuracies=test_accuracies
    )    
    