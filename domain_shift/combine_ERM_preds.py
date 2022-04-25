"""Combine predictions over multiple prediction outputs."""

import argparse
from collections import defaultdict
import numpy as np
import os
import scipy.stats

# Even though weights are different seeds, all evaluated on the same ordering
# of tests so we can population vote on prediction.
_PRED_FNAME = 'camelyon17_split:{0}_seed:0_epoch:best_pred.csv'

def main():
  """Arg defaults are filled in according to examples/configs/."""
  parser = argparse.ArgumentParser()

  # Required arguments
  parser.add_argument('--seeds_to_load', nargs='*', type=int, required=True)
  parser.add_argument('--pretrained_ERM_dir', type=str, required=True)
  parser.add_argument('--combined_log_dir', type=str, required=True)
  parser.add_argument('--splits_to_combine', nargs='+', default=['test'])

  args = parser.parse_args()

  combined_preds = defaultdict(list)
  for split in args.splits_to_combine:
    for seed in args.seeds_to_load:
      log_dir = os.path.join(args.pretrained_ERM_dir,
        f'camelyon17_erm_densenet121_seed{seed}')
      fname = os.path.join(log_dir, _PRED_FNAME.format(split))

      seed_split_preds = np.loadtxt(fname, delimiter=',')
      combined_preds[split].append(seed_split_preds)

    combined_preds[split] = np.array(combined_preds[split]).T
    population_vote, counts = scipy.stats.mode(combined_preds[split], axis=1)
    
    fout = os.path.join(args.combined_log_dir,
      'seeds_'+'_'.join([str(seed) for seed in args.seeds_to_load]))
    if not os.path.exists(fout):
      os.makedirs(fout)

    np.savetxt(os.path.join(fout, _PRED_FNAME.format(split, '0')),
      population_vote, delimiter=',', fmt='%d')

if __name__=='__main__':
  main()