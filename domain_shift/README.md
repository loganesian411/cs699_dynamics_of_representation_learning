## Bayesian Ensembling Modifications

1. Copy eval_wts.py to wilds/examples
2. Copy evaluate.py to wilds/examples
3. Copy train.py to wilds/examples

Recommended run configuration
```
python3 wilds/examples/eval_wts.py --dataset camelyon17 --algorithm ERM --root_dir <data_dir> \
    --log_dir <dir_where_all_model_weights_stored> --progress_bar --eval_only --frac 0.25
```
The frac argument allows to run eval on a smaller fraction of the data.

The log_dir should be formatted as such:
```
├── camelyon17_erm_densenet121_seed0
├── camelyon17_erm_densenet121_seed1
├── camelyon17_erm_densenet121_seed2
├── camelyon17_erm_densenet121_seed3
├── camelyon17_erm_densenet121_seed4
├── camelyon17_erm_densenet121_seed5
├── camelyon17_erm_densenet121_seed6
├── camelyon17_erm_densenet121_seed7
├── camelyon17_erm_densenet121_seed8
└── camelyon17_erm_densenet121_seed9
```

## Using Tent modifications

1. Copy run_expt_test.py to wilds/examples/
2. Copy train.py to wilds/examples
3. Copy initializer.py to wilds/examples/models
4. Copy single_model_algorithm to wilds/examples/algorithms

Recommended run configuration
```
python3 wilds/examples/run_expt_tent.py --dataset camelyon17 --algorithm ERM --root_dir <data_dir> \
    --eval_only --frac 0.1 --progress_bar --tent_model \
    --pretrained_model_path <path/to/camelyon17_erm_densenet121_seedX/best_model.pth>
```

