# Structural Information prinples-based Skill Learning (SISL)

This repository contains the source code for SISL mechanism, including:

- Source code: for pre-training and hierarchical learning with SISL
- Experimental Results
- Video Demo

For a dedicated benchmark suite, refer to [facebookresearch/bipedal-skills](https://github.com/facebookresearch/bipedal-skills).

## Setup

1. Install PyTorch based on the [official instructions](https://pytorch.org/get-started). This code has been tested with PyTorch versions 1.8 and 1.9.

2. Install remaining dependencies with:
    ```sh
    pip install -r requirements.txt
    ```

3. Optional: For optimal performance, install NVidia's [PyTorch extensions](https://github.com/NVIDIA/apex).

## Usage

The framework uses [Hydra](https://hydra.cc) for managing training configurations.

### Pre-training SISL Skills

To pre-train skill policies, use the `pretrain.py` script (requires a machine with 2 GPUs):
```sh
# Walker robot
python pretrain.py -cn walker_pretrain
```

### SISL Control

Train high-level policy with SISL as follows:
```sh
# Walker robot
python train.py -cn walker_sisl
```

Pre-trained skill policies can be used by pointing to their location as follows:
```sh
# Walker robot
python train.py -cn walker_sisl agent.lo.init_from=$PWD/pretrained-skills/walker.pt
```

### Benchmark Baselines

To run individual baselines, pass the relevant configuration name as the `-cn` argument to `train.py`. 
