"""Perform model ensembling by combining predictions from multiple models.

Currently supports the following ensembling approaches:
  1) population voting
  2) unweighted logit averaging
  3) maximum softmax probability (maxprob) per test prediction weighted logit averaging
  4) average maxprob over all predictions weighted logit averaging
"""
import argparse
import numpy as np
import os
import scipy.special
import scipy.stats
import torch
import wilds.common.metrics.all_metrics as wilds_metrics

# Even though weights are from different seeds, all were evaluated on the same seed,
# that is ordering of data, so we can ensemble predictions.
_PRED_FNAME = 'camelyon17_split:test_seed:0_epoch:best_pred.csv' # categorical predication
_RAW_PRED_FNAME = 'camelyon17_split:{0}_seed:0_raw_y_pred.csv' # logit predictions

def main():
  """Arg defaults are filled in according to examples/configs/."""
  parser = argparse.ArgumentParser()
  parser.add_argument('--seeds_to_load', nargs='*', type=int, required=True,
                      help='Which seeded models to ensemble.')
  parser.add_argument('--pretrained_ERM_dir', type=str, required=True,
                      help='Directory where seeded models live.')
  parser.add_argument('--combined_log_dir', type=str, required=True,
                      help='Where to save the combined/ensembled predictions.')
  parser.add_argument('--method', type=str, default='population_vote',
    choices=['population_vote', 'average', 'wt_ave_maxprob', 'wt_per_maxprob'],
    help='Ensembling method.')

  args = parser.parse_args()

  combined_preds = []
  for seed in args.seeds_to_load:
    log_dir = os.path.join(args.pretrained_ERM_dir,
      f'camelyon17_erm_densenet121_seed{seed}')

    if args.method == 'population_vote':
      fname = os.path.join(log_dir, _PRED_FNAME)
    else: # Load raw preds for everything else.
      fname = os.path.join(log_dir, _RAW_PRED_FNAME.format('test'))
    combined_preds.append(np.loadtxt(fname, delimiter=','))

  if args.method == 'wt_ave_maxprob' or args.method == 'wt_per_maxprob':
    # Compute confidence per prediction.
    confidence_vals = [np.amax(scipy.special.softmax(p, axis=1), axis=1) for p in combined_preds]
    if args.method == 'wt_ave_maxprob':
      weights = np.array([np.mean(x) for x in confidence_vals])
    else: # method == wt_per_maxprob
      weights = np.array(confidence_vals) # models-by-predictions

  combined_preds = np.array(combined_preds) # models-by-predictions
  if args.method == 'population_vote':
    ensembled_pred, counts = scipy.stats.mode(combined_preds.T, axis=1)
  else: # Averaging raw predictions before classification.
    if args.method == 'average' or args.method == 'wt_ave_maxprob':
      if args.method == 'average': weights = np.ones(combined_preds.shape)
      average_prediction = np.average(combined_preds, weights=weights, axis=0)
    else: # Averaging using confidence per prediction.
      average_prediction = np.zeros(combined_preds.shape[1:])
      for i in range(average_prediction.shape[-1]):
        this_average = np.sum(combined_preds[..., i] * weights, axis=0)
        this_average /= np.sum(weights, axis=0)
        average_prediction[:, i] = this_average

    ensembled_pred = wilds_metrics.multiclass_logits_to_pred(torch.from_numpy(average_prediction))
    ensembled_pred = ensembled_pred.detach().numpy()
  
  fout = os.path.join(args.combined_log_dir,
    f'{args.method}_seeds_'+'_'.join([str(seed) for seed in args.seeds_to_load]))
  print('Saving to: ', fout)
  if not os.path.exists(fout):
    os.makedirs(fout)
  np.savetxt(os.path.join(fout, _PRED_FNAME), ensembled_pred, delimiter=',', fmt='%d')

if __name__=='__main__':
  main()