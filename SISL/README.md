## Requirements

- python 3.7+
- mujoco 2.1
- Ubuntu 18.04

## Installation Instructions

To install MuJoCo follow the instructions [here](https://github.com/openai/mujoco-py).

```
conda env create -f environment.yml
conda activate sisl
pip install -e .
```

## Data Collection and Training
To collect a dataset using the scripted controllers run the following command:
```
python data/collect_demos.py --num_trajectories 40000 --subseq_len 10 --task block
```
There are two sets of tasks `block` and `hook`
The dataset collected for the `block` tasks can be used to train a downstream RL agent in the `FetchPyramidStack-v0`, `FetchCleanUp-v0` and `FetchSlipperyPush-v0` environments.
The dataset collected for the `hook` task is used to train the downstream RL agent in the `FetchComplexHook-v0` environment.
We collect the demonstration data for the hook and block based environments in the `FetchHook-v0` and `FetchPlaceMultiGoal-v0` environments respectively.

To train the skill modules on the collected dataset run the following command:
```
python train_skill_modules.py --config_file block/config.yaml --dataset_name fetch_block_40000
```
To visualise the performance of the trained skill module run the following command:
```
python utils/test_skill_modules.py --dataset_name fetch_block_40000 --task block --use_skill_prior True
```

To train the SISL agent using the trained skill modules, run the following command:

```
python train_sisl_agent.py --config_file table_cleanup/config.yaml --dataset_name fetch_block_40000
```
  
## Logging
  
All results are logged using [Weights and Biases](https://wandb.ai). An account and initial login is required to initialise logging as described on thier website.
