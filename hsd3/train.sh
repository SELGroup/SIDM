#!/bin/bash

# Download the pre-trained skills
wget https://dl.fbaipublicfiles.com/hsd3/pretrained-skills.tar.gz

# Unpack the archive
tar -xzvf pretrained-skills.tar.gz

# Train the Walker robot with the pre-trained skills
python train.py -cn walker_hsd3 agent.lo.init_from=$PWD/pretrained-skills/walker.pt

# Train the Humanoid robot with the pre-trained skills
#python train.py -cn humanoid_hsd3 agent.lo.init_from=$PWD/pretrained-skills/humanoidpc.pt
