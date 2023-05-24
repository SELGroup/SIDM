#!/bin/bash



# Train the Walker robot with the pre-trained skills
nohup python pretrain2.py -cn walker_pretrain estimate_joint_spaces=null checkpoint_path=checkpoint-lo-null.pt >null.log &
nohup python pretrain2.py -cn walker_pretrain estimate_joint_spaces=kmeans checkpoint_path=checkpoint-lo-kmeans.pt > kmeans.log &
nohup python pretrain2.py -cn walker_pretrain estimate_joint_spaces=sep checkpoint_path=checkpoint-lo-sep.pt > sep.log &
nohup python pretrain2.py -cn walker_pretrain estimate_joint_spaces=hac checkpoint_path=checkpoint-lo-hac.pt > hac.log &
# Train the Humanoid robot with the pre-trained skills
#python train.py -cn humanoid_hsd3 agent.lo.init_from=$PWD/pretrained-skills/humanoidpc.pt
# estimate_joint_spaces: sep
# init_model_from:
# checkpoint_path: checkpoint-lo-sep.pt