# Homework.py

# Hey there, future master of Reinforcement Learning! Ready to code your way to glory once again?
# This time, we're pulling out the big guns. More imports, more variables, and a lot more fun!

# First, as usual, let's import some modules. Don't worry, they won't bite.
import itertools
import json
import logging
import os
import shutil
from copy import copy
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List

import gym
import hydra
import numpy as np
import torch as th
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from visdom import Visdom

# Remember hucc, our best friend from the last homework? Yeah, it's back!
import hucc
from hucc.agents.utils import discounted_bwd_cumsum_
from hucc.hashcount import HashingCountReward
from hucc.spaces import th_flatten

log = logging.getLogger(__name__)

# Here's our class again, this time with some new friends. 
# I'll leave you to figure out their purpose. Remember, curiosity is the key to success!
### Class: TrainingSetup

#Your class should have the following attributes:

# * `cfg`: A DictConfig object. This is your configuration dictionary.
# * `agent`: An instance of the hucc.Agent class.
# * `model`: A nn.Module object. This is your model.
# * `tbw`: A SummaryWriter instance.
# * `viz`: A Visdom instance. This will be used for visualization.
# * `rq`: An instance of hucc.RenderQueue.
# * `envs`: An instance of hucc.VecPyTorch. This is your environment.
# * `eval_envs`: Another instance of hucc.VecPyTorch. This is your evaluation environment.
# * `eval_fn`: A Callable object. It's a function that takes in a TrainingSetup and an integer and returns nothing.
# * `n_samples`: An integer. This keeps count of the samples.
# * `replaybuffer_checkpoint_path`: A string, default is 'replaybuffer.pt'. This is the path for the replay buffer checkpoint.
# * `training_state_path`: A string, default is 'training\_state.json'. This is the path for the training state.
# * `hcr_checkpoint_path`: A string, default is 'hashcounts.pt'. This is the path for the hash counts reward checkpoint.
# * `hcr_space`: For this assignment, leave this as None.
# * `hcr`: An instance of HashingCountReward. For this assignment, leave this as None.

# ### Method: close

# Your `close` method should do the following:

# * Close `rq`, `envs`, and `eval_envs`.
# * Try to unlink (delete) the file at the path `replaybuffer_checkpoint_path`. If the file doesn't exist, don't do anything.



# Class we're going to complete:
class TrainingSetup(SimpleNamespace):
    # cfg: This variable will hold a configuration dictionary. 
    # It's like a treasure map, but instead of leading us to a chest full of gold, it guides us through our training process.
    cfg: DictConfig

    # agent: This is our hero, the one who's going to learn and explore the environment.
    # It's like a small child, eager to learn and discover the world.
    agent: hucc.Agent

    # model: This is the brain of our agent. It's going to make our agent smarter... or dumber, depending on how well we train it.
    model: nn.Module

    # tbw: This will be used for TensorBoard to visualize our training progress.
    # It's like our agent's diary, where it writes down everything it learns.
    tbw: SummaryWriter

    # And here come the rest of our adventure tools! I'll leave you to find out what they do.
    viz: hucc.Visdom
    rq: hucc.RenderQueue
    envs: hucc.VecPyTorch
    eval_envs: hucc.VecPyTorch
    eval_fn: Callable  # Callable[[TrainingSetup, int], None]
    n_samples: int = 0
    replaybuffer_checkpoint_path: str = 'replaybuffer.pt'
    training_state_path: str = 'training_state.json'
    hcr_checkpoint_path: str = 'hashcounts.pt'
    hcr_space = None
    hcr: hucc.HashingCountReward = None

    # Now, here's the first challenge for you!
    # Complete this close function to close all the necessary resources.
    # Remember, just like you need to turn off the lights before leaving a room,
    # you need to close everything before ending your training.
    def close(self):
        # TODO: Use the hucc documentation and your brilliant mind to figure out how to close the resources.
        # Don't forget to handle exceptions! They're like the monsters in your adventure, and you're the hero who's going to defeat them.
        pass

# Welcome back, brave adventurer! Now, it's time to setup training. 
# Like a boss preparing for an epic quest, we need to ensure our gear (code) is ready for the journey ahead.


# Let's create a function to set up the training. 
# This function will take a configuration dictionary as an argument and return a TrainingSetup object.
def setup_training(cfg: DictConfig) -> TrainingSetup:
    # TODO: Check if CUDA is available. If it's not, then let's use the CPU.
    # Don't forget to set the seed for PyTorch to ensure the results are reproducible.
    # Tip: You might want to use 'th.manual_seed()' for setting the seed.

    # TODO: Setup the Visdom visualization tool.
    # It's like a magical mirror that lets us see how our agent is performing.
    # Remember to use the configuration parameters from 'cfg'!

    # TODO: Create the environment for training and evaluation using hucc.
    # It's like the world where our agent will live and learn. Make sure it's a nice place!

    # TODO: Setup the observation and action spaces.
    # This defines what our agent can see and do in its world.

    # TODO: Create the model. You'll need to recursively create models for each key if the observation and action spaces are dicts.

    # TODO: Use the 'hucc.make_agent()' function to create the agent.
    # It's like bringing our hero to life!

    # TODO: Setup TensorBoard SummaryWriter.
    # It's like a diary where our agent will write down its thoughts and feelings (well, more like performance metrics, but you get the idea).

    # TODO: Setup the HashingCountReward (hcr), if required by the config.
    # It's like a special reward that our agent can get for exploring new things.

    # TODO: Finally, return a TrainingSetup object with all the components we've created.
    # It's like the backpack filled with all the gear our agent will need for its epic quest!

    return TrainingSetup()


# Let's create a function to evaluate our agent's performance. 
# This function will take a TrainingSetup object and the number of samples to test on.

def eval(setup: TrainingSetup, n_samples: int = -1):
    # TODO: Extract the agent, request queue, and evaluation environments from the setup.

    # TODO: Reset the environments and initialize variables for collecting rewards, 
    # dones, request queue inputs, images, and metrics.

    # TODO: Now, let's step into the coliseum! Run a loop until all environments have completed.

        # TODO: If video collection is enabled, collect and annotate the video frames.

        # TODO: Get the agent's action and apply it to the environments.

        # TODO: If entropy is being tracked, append it to the entropy_ds list.

        # TODO: If certain metrics are being tracked, append them to the metrics_v dictionary.

        # TODO: Append the reward and done tensors to their respective lists.

        # TODO: If all environments are done, break the loop. Otherwise, reset the done environments.

    # TODO: Compute the undiscounted and discounted returns, and the episode length.

    # TODO: Update the metrics_v dictionary with the computed returns and episode lengths.

    # TODO: Log the metrics to TensorBoard using the agent's summary writer.

    # TODO: Log the average episode length and return.

    # TODO: If entropy was being tracked, log its mean and histogram to TensorBoard.

    # TODO: If video was being collected, annotate the frames with the accumulated reward 
    # and push them to the request queue for display.
    pass

# Now, over to you! Strap on your armor, and let's step into the coliseum!


#This time, we're diving into the training loop, the heart of any machine learning algorithm.
# It's where the magic happens: 
# the agent will interact with the environment, learn from its experiences, and gradually improve its performance.
def train_loop(setup: TrainingSetup):
    # TODO: Extract the agent, request queue, and environments from the setup.

    # TODO: Turn on the agent's training mode.

    # TODO: Set up some variables for the loop. These include the number of environments, 
    # the path for checkpointing, video recording settings, and the maximum steps for training.

    # TODO: Get the initial observation from the environments.

    # TODO: Here we go! Start the training loop.

        # TODO: If it's time for evaluation, save a checkpoint of the agent and evaluate it.

        # TODO: If video recording is enabled and it's time to record a video, 
        # collect the video frames and push them to the request queue.

        # TODO: Get the agent's action and apply it to the environments.

        # TODO: Let the agent learn from the step.

        # TODO: If state count dumping is enabled and it's time to dump, 
        # compute the unique state count and log it to TensorBoard.

        # TODO: If any environments are done, reset them.

        # TODO: Increment the number of samples.

    # TODO: After the training loop, save a final checkpoint and evaluate the agent one last time.
    pass
# That's it! You're ready to take on the training loop. Remember, practice makes perfect!

# In this function, we're creating a series of checkpoints for our AI agent's training process. 
# The function is named checkpoint and it's responsible for saving the current state of the agent and several other objects. 
# Checkpointing is important because it allows you to pause and resume training, and also provides a way to recover if something goes wrong.
# The function checkpoint is taking one argument, setup, which is an instance of the TrainingSetup class.

# Some Python functions that will be helpful for this task are:

# with open(file_path, mode) as f:: It's used to open a file. 'wb' stands for 'write binary' and 'wt' for 'write text'.
# object.save(file): It's used to save a checkpoint of an object to a file.
# json.dump(object, file): It's used to write a JSON object to a file.
# log.exception(message): It's used to log an exception along with a custom message.

#The try-except blocks are there to catch and log any errors that might happen during the checkpointing process. 
# It's a good practice to use them when dealing with file I/O operations because many things can go wrong 
# (e.g., disk full, no write permissions, etc.).
def checkpoint(setup: TrainingSetup):
    # TODO: Log that we are starting the checkpointing process.

    # TODO: Extract the configuration from the setup.

    # TODO: Try to open the checkpoint file in write-binary mode.

        # TODO: If successful, save the agent's checkpoint to the file.

    # TODO: If there's an error during this process, log the exception.

    # TODO: If the agent has a replay buffer, try to save its checkpoint.

    # TODO: If there's an error during this process, log the exception.

    # TODO: If the setup has a hashcount replay (hcr), try to save its checkpoint.

    # TODO: If there's an error during this process, log the exception.

    # TODO: Try to open the training state file in write-text mode.

        # TODO: If successful, dump the number of samples to the file as a JSON object.

    # TODO: If there's an error during this process, log the exception.


# restore(setup: TrainingSetup): This function restores the state of the training process from saved checkpoint files. 
# The setup argument is an instance of the TrainingSetup class.
# Key Python functions and methods for this task include:

# Path(file_path).is_file(): Checks if a file exists at the given path.
# with open(file_path, mode) as f:: Opens a file. 'rt' stands for 'read text' and 'rb' for 'read binary'.
# object.load(file): Loads a checkpoint from a file into an object.
# json.load(file): Reads a JSON object from a file.
# log.exception(message): Logs an exception along with a custom message.
def restore(setup: TrainingSetup):
    # TODO: Check if the training state file exists. If it does, open it and load the number of samples.

    # TODO: If there's an error during this process, log the exception.

    # TODO: Extract the configuration from the setup.

    # TODO: Check if the agent's checkpoint file exists. If it does, open it and load the checkpoint into the agent.

    # TODO: If the checkpoint file doesn't exist, raise an error.

    # TODO: If the agent has a replay buffer and its checkpoint file exists, open it and load the checkpoint into the replay buffer.

    # TODO: If there's an error during this process, log the exception.

    # TODO: If the setup has a hashcount replay (hcr) and its checkpoint file exists, open it and load the checkpoint into the hcr.

    # TODO: If there's an error during this process, log the exception.


# auto_adapt_config(cfg: DictConfig) -> DictConfig: This function adapts the configuration based on the specific environment.
# Key Python functions and methods for this task include:

# str.startswith(substring): Checks if a string starts with the given substring.
# OmegaConf.set_struct(cfg, bool): Sets the struct flag of a DictConfig object. If the flag is set to True, the object becomes read-only.
# OmegaConf.to_yaml(cfg, resolve=True): Converts a DictConfig object to a YAML string.

def auto_adapt_config(cfg: DictConfig) -> DictConfig:
    # TODO: If the environment is 'BiskStairs', append '-relz' to the goal_space configuration.

    # TODO: If the environment is 'BiskPoleBalance', set the action_interval configuration to 1.

    # TODO: Return the adapted configuration.

# main(cfg: DictConfig): This function sets up and starts the training loop.

# Key Python functions and methods for this task include:

# hydra.utils.get_original_cwd(): Gets the original working directory before Hydra changed it.
# setup_training(cfg): Sets up
def main(cfg: DictConfig):
    # TODO: Log the original working directory.

    # TODO: If auto_adapt is enabled in the configuration, adapt the configuration.

    # TODO: Log the final configuration.

    # TODO: Set up the training process.

    # TODO: Set the checkpoint function.

    # TODO: Restore the state of the training process from checkpoints.

    # TODO: Start the training loop.

    # TODO: Close the setup after the training loop has finished.
