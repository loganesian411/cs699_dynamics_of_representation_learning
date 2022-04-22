Copy the file eval_wts.py to wilds/examples/. This will allow the script to actually be run.

Recommended run configuration
```
python3 examples/eval_wts.py --dataset camelyon17 --algorithm ERM --root_dir <data_dir> \
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