# Hierarchical State Abstraction Based on Structural Information Principles

## Installation

All of the dependencies are in the `conda_env.yml` file. They can be installed manually or with the following command:

```
conda env create -f conda_env.yml
```

If that doesn't work, use pip to install either of the requirements files.


## Instructions
To reproduce our results use the following commands.

### SISA
```sh
# for seed in {1..10}:
#   for domain, task in [
#     ('ball_in_cup', 'catch'),
#     ('cartpole', 'swingup'),
#     ('cheetah', 'run'),
#     ('finger', 'spin'),
#     ('reacher', 'easy'),
#     ('walker', 'walk),
#   ]:
#     OTHER_SETTINGS = [check_appendix]
python -m train --replicate --sisa --sisa_pretrain_batch_size 512 --init_steps [INIT_STEPS] --sisa_catchup_steps [CATCHUP_STEPS] --sisa_pretrain_steps 100000 --sisa_inv_coef [COEF_INV] --sisa_smoothness_coef [COEF_SMOOTH] --sisa_smoothness_max_dz 0.01 --domain_name [domain] --task_name [task] --sisa_lr [LR] --seed [seed] --tag sisa-agent --work_dir ./tmp/sisa
```

Explanation of important, non-obvious args:
```
  `--replicate` - ensures that the RAD settings replicate the original RAD paper
  `--sisa`    - enables the SISA abstraction objectives. It is intended to be used
                  in conjunction with the other SISA hyperparameters.
  `--work_dir`  - where to store learning performance results
  `--tag`       - a tag to use for each experiment (mainly used during hyperparameter
                  tuning)
```
