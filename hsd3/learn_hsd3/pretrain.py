# Copyright (c) 2021-present, Facebook, Inc.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import getpass
import importlib
import json
import logging
import os
import shutil
import uuid
from collections import defaultdict
from copy import copy, deepcopy
from itertools import combinations
from pathlib import Path
from typing import Dict, Optional, cast

import gym
import hydra
import numpy as np
import torch as th
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch import multiprocessing as mp

import hucc
from hucc.agents.sacmt import SACMTAgent
from hucc.agents.utils import discounted_bwd_cumsum_
from hucc.envs.ctrlgs import CtrlgsPreTrainingEnv
from hucc.envs.goal_spaces import g_goal_spaces
from hucc.spaces import th_flatten
from train import TrainingSetup, checkpoint, restore, setup_training
log = logging.getLogger(__name__)

# This function is for evaluating the agent's performance across multiple tasks. It returns a dictionary mapping task identifiers
# to performance metrics. The function collects several episodes of interaction data for each task and computes several statistics,
# such as the total reward and the proportion of goals reached.

def eval_mfdim(setup, n_samples: int) -> Dict[str, float]:

    # Extract important objects from the setup. The `cfg` object contains the configuration settings for the experiment.
    # The `agent` object is the learning agent. The `rq` object is used for rendering and recording videos of the agent's behavior.
    # The `envs` object is a set of environments used for evaluation. All these objects are part of the `setup` object passed to the function.
    # Your code here:

    # ...

    # Map from task identifiers to task indices. This is used to map the one-hot encoded task identifiers in the observations
    # to the corresponding task indices.
    # Your code here:

    # ...

    # Set the random seed for the environments to ensure that the evaluation results are reproducible. Then, reset the environments
    # to get the initial observations.
    # Your code here:

    # ...

    # Initialize several variables used for recording the agent's behavior and computing the evaluation statistics.
    # `reached_goala` is a dictionary that maps task identifiers to a list of boolean values indicating whether the agent reached
    # the goal for each episode of the task. `reward` is a tensor that will hold the rewards received by the agent. `rewards` is a list
    # that will hold tensors of rewards for each time step. `dones` is a list that will hold tensors of boolean values indicating whether
    # each episode has ended. `rq_in` is a list of lists that will hold the data for rendering and recording videos.
    # Your code here:

    # ...

    # The main evaluation loop. This loop collects several episodes of interaction data for each task and computes several statistics.
    # The loop continues until a certain number of episodes (`n_episodes`) have been collected.
    # Your code here:

    # ...

        # Compute the task abstractions for the current observations. This involves finding the indices of the one-hot encoded task identifiers
        # in the observations and mapping them to the corresponding task indices.
        # Your code here:

        # ...

        # If video recording is enabled, record frames of the agent's behavior. This involves rendering the environments and storing the rendered
        # images along with some additional information, such as the current time step, the features of the current state, and the current reward.
        # Your code here:

        # ...

        # Compute the agent's action based on the current observations. Then, execute the action in the environments to get the next observations,
        # the rewards, the done flags, and the info dictionaries. The `done` flags indicate whether each episode has ended, and the `info` dictionaries
        # contain additional information about the environments' states.
        # Your code here:

        # ...

        # Record whether the agent reached the goal for each episode that has ended. This involves extracting the `reached_goal` flag from the info
        # dictionaries and storing it in the `reached_goala` dictionary under the key corresponding to the task identifier.
        # Your code here:

        # ...

        # If the total number of completed episodes has reached `n_episodes`, break the loop. Otherwise, reset the environments that have ended
        # and continue the loop
                # Your code here:

        # ...

    # Compute the discounted and undiscounted returns for each episode. The `reward` tensor is reshaped and copied to two new tensors,
    # `r_discounted` and `r_undiscounted`. Then, the `discounted_bwd_cumsum_` function is called to compute the cumulative sum of the rewards
    # in reverse order (from the end of the episode to the beginning). This function multiplies each reward by a discount factor (`cfg.agent.gamma`)
    # raised to the power of the time step, which results in more recent rewards having a greater influence on the sum.
    # Your code here:

    # ...

    # Compute the proportion of goals reached for each task. This involves iterating over the `reached_goala` dictionary, computing the proportion
    # of true values in the list for each key, and storing the results in the `goalsa_reached` dictionary. The overall proportion of goals reached
    # is then computed by taking a weighted average of the proportions for each task, with the weights being the number of episodes for each task.
    # Your code here:

    # ...

    # Log the evaluation statistics to the tensorboard writer if it is enabled. This includes the discounted and undiscounted returns, the proportion
    # of goals reached, and the number of trials for each task.
    # Your code here:

    # ...

    # Log a summary of the evaluation results to the console. This includes the average discounted and undiscounted returns, the minimum and maximum
    # undiscounted returns, and the overall proportion of goals reached.
    # Your code here:

    # ...

    # If video recording is enabled and there are frames to record, display the cumulative reward in the video and push the frames to the rendering queue.
    # This involves computing the cumulative reward for each time step, appending it to the data for each frame, and pushing the data to the rendering queue.
    # Then, call the `plot` method of the rendering queue to generate the video.
    # Your code here:

    # ...

    # Return the dictionary mapping task identifiers to the proportion of goals reached for each task.
    # Your code here:

    # ...
# END_OF_CODE


# This function is responsible for the learner part of the training loop in a distributed setup.
# It receives transitions from the queue, and uses them to update the agent. It doesn't interact with the environments directly.
def train_loop_mfdim_learner(setup: TrainingSetup, queue: mp.Queue):

    # Retrieve the configuration, the agent, the environment, and the number of environments from the setup object.
    # Insert your code here:

    # ...

    # Log a debug message indicating that the learner has started.
    # This is useful for debugging and understanding the flow of the program.
    # Insert your code here:

    # ...

    # Set the agent to training mode.
    # This is important because some components of the agent may behave differently during training and evaluation.
    # Insert your code here:

    # ...

    # While the total number of samples is less than the maximum number of steps...
    # Insert your code here:

    # ...

        # ...get a transition from the queue.
        # Insert your code here:

        # ...

        # ...use this transition to update the agent.
        # Insert your code here:

        # ...

        # ...delete the transition to free up memory.
        # Insert your code here:

        # ...

        # ...increase the total number of samples by the number of environments.
        # This is because each environment generates one sample per step.
        # Insert your code here:

        # ...

# END OF FUNCTION DEFINITION

# This function is shorter than the actor function because it only has to handle the learning part of the training loop.
# The actor function, on the other hand, has to handle both the interaction with the environments and the learning.
# Moreover, the actor function also has to handle logging, checkpointing, and other tasks that are not part of the core learning process.




# FUNCTION DEFINITION
# The function 'train_loop_mfdim_actor' is designed to run the training loop for a multi-feature dimension actor.
# The training loop involves collecting samples from the environment, updating the agent based on these samples, and saving performance metrics.
def train_loop_mfdim_actor(setup: TrainingSetup):

    # Extract necessary components from the setup object for ease of use.
    # Insert your code here:

    # ...

    # Set the agent into training mode. This affects certain behaviours of the agent, like dropout and batch normalization.
    # Insert your code here:

    # ...

    # Create a deep copy of the model and move it to the CPU for use in the environment.
    # Also, set the 'requires_grad' attribute of the model's parameters to False, because we don't need gradients for the target network.
    # Insert your code here:

    # ...

    # Set the shared model in the environments and initialize variables for the training loop.
    # Insert your code here:

    # ...

    # The training loop starts here. It continues until the total number of collected samples exceeds the maximum number of steps.
    # The main responsibilities inside this loop include collecting new samples, updating the agent, and performing evaluations at regular intervals.
    while setup.n_samples < max_steps:

        # Every certain number of steps (as determined by the eval.interval configuration), perform an evaluation and save a checkpoint.
        # The evaluation involves running the policy without any exploration noise and checking its performance.
        # Checkpoints allow us to resume training from this point if it gets interrupted.
        # Insert your code here:

        # ...

        # Estimate the controllability of the agent using the collected samples.
        # Controllability is a measure of how well the agent can move to different states in the environment.
        # Insert your code here:

        # ...

        # Determine the method of evaluation based on the configuration.
        # The available options are 'rollouts', 'running_avg', 'q_value', and 'reachability'.
        # Insert your code here:

        # ...

        # Fix up goal keys to match '+' syntax and log the performance metrics to TensorBoard.
        # TensorBoard allows us to visualize these metrics in real time as the agent is training.
        # Insert your code here:

        # ...

        # Save the current abstraction information, which includes the task map, goal dimensions, and performance metrics.
        # This information can be useful for debugging and understanding the learned policy.
        # Insert your code here:

        # ...

        # Update the feature distribution based on the new performance metrics.
        # This allows the agent to focus more on tasks that it is currently bad at.
        # Insert your code here:

        # ...

        # Every certain number of steps (as determined by the video.interval configuration), record a video of the agent's performance.
        # This provides a visual way to understand the behavior of the agent.
        # Insert your code here:

        # ...

        # Collect a new sample from each environment by first selecting an action according to the agent's policy, then applying this action to the environment.
        # The resulting new states, rewards, and done flags are stored in the respective variables.
        # Insert your code here:

        # ...

        # Split the collected samples into chunks and put each chunk into a separate queue.
        # These queues are used by the learner processes to update the agent.
        # Insert your code here:

        # ...

        # Perform an update step on the agent.
        # This involves computing the gradient of the loss function with respect to the agent's parameters and taking a
                # step in the direction of negative gradient. This is how the agent learns from the samples.
        # Insert your code here:

        # ...

        # Reset environments that have reached their terminal state.
        # This is important to ensure that each environment is always active and generating new samples.
        # Insert your code here:

        # ...

        # Maintain a running average of the controllability during training.
        # This is a metric that measures how well the agent can reach different states in the environment.
        # Insert your code here:

        # ...

        # If the agent has been updated, copy the updated model parameters into the shared model.
        # This is important to ensure that the shared model, which is used by the environments, is always up-to-date.
        # Insert your code here:

        # ...

    # After the training loop is finished, save one last checkpoint.
    # This allows us to resume training from the very end if we want to train the agent for more steps.
    # Insert your code here:

    # ...

    # Perform one last evaluation to get the final performance of the agent.
    # Then, estimate the final controllability of the agent.
    # Insert your code here:

    # ...

    # Determine the method of evaluation based on the configuration, as done previously inside the training loop.
    # Insert your code here:

    # ...

    # Log the final performance of the agent for each feature.
    # This gives us an idea of what the agent has learned to do well and what it still struggles with.
    # Insert your code here:

    # ...

    # Save the final abstraction information, which includes the task map, goal dimensions, and performance metrics.
    # This is the final summary of what the agent has learned.
    # Insert your code here:

    # ...

# END OF FUNCTION DEFINITION


def setup_training_mfdim(cfg: DictConfig):
    # Ensure the feature dimensions configuration is a string.
    if not isinstance(cfg.feature_dims, str):
        cfg.feature_dims = str(cfg.feature_dims)

    # Get goal spaces for a particular robot and features.
    gs = g_goal_spaces[cfg.features][cfg.robot]
    n = len(gs['str'])

    # Based on the configuration, select the appropriate feature dimensions.
    # This section supports some special names like 'all', 'torso' for convenience.
    # It also handles a custom configuration where feature dimensions are specified.
    # If the configuration doesn't match these special cases, it treats the config as a regular expression and selects the matching features.

    # ...

    # Check each selected feature dimension. If a feature is not controllable, it's removed from the list. 
    # This section also logs a warning whenever it removes a feature.

    # ...

    # If the number of selected feature dimensions is less than the requested rank (number of features to control), raise an error.

    # ...

    # Update the environment arguments with the selected robot and other parameters.

    # ...

    # Calculate the distribution of features for the training environment.
    # If the task weighting configuration starts with 'lp', normalize the distribution so that the sum of all weights equals 1.

    # ...

    # Create a mapping from features to tasks.
    # This section also updates the environment arguments with the feature distribution and task map.

    # ...

    # If the discount factor gamma is set to 'auto_horizon', calculate it based on the horizon (the number of steps in each episode).

    # ...

    # Call the generic setup_training function with the updated configuration.
    # Also, prepare the goal dimensions and task map for use in the training loop.

    # ...

    if not isinstance(cfg.feature_dims, str):
        cfg.feature_dims = str(cfg.feature_dims)
    gs = g_goal_spaces[cfg.features][cfg.robot]
    n = len(gs['str'])
    # Support some special names for convenience
    if cfg.feature_dims == 'all':
        dims = [str(i) for i in range(n)]
    elif cfg.feature_dims == 'torso':
        dims = [
            str(i)
            for i in range(n)
            if gs['str'][i].startswith(':')
            or gs['str'][i].startswith('torso:')
            or gs['str'][i].startswith('root')
        ]
    else:
        try:
            for d in cfg.feature_dims.split('#'):
                _ = map(int, d.split('+'))
            dims = [d for d in cfg.feature_dims.split('#')]
        except:
            dims = [
                str(i)
                for i in range(n)
                if re.match(cfg.feature_dims, gs['str'][i]) is not None
            ]
    uncontrollable = set()
    for dim in dims:
        for d in map(int, dim.split('+')):
            if not CtrlgsPreTrainingEnv.feature_controllable(
                cfg.robot, cfg.features, d
            ):
                uncontrollable.add(dim)
                log.warning(f'Removing uncontrollable feature {dim}')
                break
    cfg.feature_dims = '#'.join([d for d in dims if not d in uncontrollable])
    if cfg.feature_rank == 'max':
        cfg.feature_rank = len(cfg.feature_dims.split('#'))
    if len(cfg.feature_dims) < int(cfg.feature_rank):
        raise ValueError('Less features to control than the requested rank')

    # Setup custom environment arguments based on the selected robot
    prev_args: Dict[str, Any] = {}
    if isinstance(cfg.env.args, DictConfig):
        prev_args = dict(cfg.env.args)
    cfg.env.args = {
        **prev_args,
        'robot': cfg.robot,
    }
    fdist = {
        ','.join(d): 1.0
        for d in combinations(cfg.feature_dims.split('#'), cfg.feature_rank)
    }
    if cfg.task_weighting.startswith('lp'):
        for k, v in fdist.items():
            fdist[k] = v / len(fdist)
    feats: Set[int] = set()
    task_map: Dict[str, int] = {}
    for fs in fdist.keys():
        for f in map(int, fs.replace('+', ',').split(',')):
            feats.add(f)
    for f in sorted(feats):
        task_map[str(f)] = len(task_map)
    cfg.env.args = {
        **cfg.env.args,
        'feature_dist': fdist,
        'task_map': task_map,
    }
    if cfg.agent.gamma == 'auto_horizon':
        cfg.agent.gamma = 1 - 1 / cfg.horizon
        log.info(f'gamma set to {cfg.agent.gamma}')

    setup = setup_training(cfg)
    if 'goal_dims' in cfg.env.args:
        setup.goal_dims = dict(cfg.env.args.goal_dims)
    else:
        setup.goal_dims = dict(cfg.env.args.feature_dist)
    setup.task_map = dict(cfg.env.args.get('task_map', {}))
    return setup



def worker(rank, role, queues, bcast_barrier, cfg: DictConfig):
    # If a GPU is available, this process will use the GPU with the same ID as the process's rank.
    if th.cuda.is_available():
        th.cuda.set_device(rank)

    # Logging info about process group creation.
    log.info(
        f'Creating process group of size {cfg.distributed.size} via {cfg.distributed.init_method} [rank={rank}]'
    )

    # Initializing a process group for distributed training. 
    # The process group is a set of processes that can communicate with each other.
    dist.init_process_group(
        backend='nccl' if th.cuda.is_available() else 'gloo',  # 'nccl' backend is used if GPU is available, else 'gloo' is used.
        rank=rank,  # The rank (ID) of the current process in the process group.
        world_size=cfg.distributed.size,  # Total number of processes in the group (actors + learners).
        init_method=cfg.distributed.init_method,  # Method used to setup the distributed environment.
    )

    # Storing the role ('learner' or 'actor') of the current process in the configuration.
    cfg.distributed.role = role

    # Adjusting the configuration based on the role of the current process.
    if role == 'learner':
        # OmegaConf is a library for handling hierarchical configurations.
        # Here, it's used to allow modifications to the 'env' part of the configuration.
        OmegaConf.set_struct(cfg.env, False)
        cfg.env.args.fork = False  # Disabling forking in the environment.
        cfg.env.eval_procs = 1  # Setting the number of evaluation processes to 1.
        cfg.env.train_procs //= cfg.distributed.num_learners  # Reducing the number of training processes in proportion to the number of learners.
        cfg.agent.batch_size //= cfg.distributed.num_learners  # Reducing the batch size in proportion to the number of learners.
        cfg.agent.samples_per_update //= cfg.distributed.num_learners  # Reducing the number of samples per update in proportion to the number of learners.
        cfg.agent.warmup_samples //= cfg.distributed.num_learners  # Reducing the number of warmup samples in proportion to the number of learners.

    # Setting up the training environment and agent.
    try:
        setup = setup_training_mfdim(cfg)
    except:
        log.exception('Error in training loop')
        raise

    # Storing the queues (for inter-process communication) in the setup.
    setup.queues = queues
    agent = setup.agent
    agent.bcast_barrier = bcast_barrier

    # All processes wait at the barrier until all of them have reached this point.
    bcast_barrier.wait()

    # If there's more than one learner, a new group is created for the learners.
    if cfg.distributed.num_learners > 1:
        learner_group = dist.new_group(
            [i for i in range(cfg.distributed.num_learners)]
        )
        agent.learner_group = learner_group

    # Loading a pretrained model if specified in the configuration.
    if cfg.init_model_from:
        log.info(f'Initializing model from checkpoint {cfg.init_model_from}')
        with open(cfg.init_model_from, 'rb') as fd:
            data = th.load(fd)
            setup.model.load_state_dict(data['_model'])
            agent._log_alpha.clear()
            # Here, the log_alpha values from the loaded model are copied into the agent's log_alpha.
            for k, v in data['_log_alpha'].items():
                agent._log_alpha[k] = v

    # Restoring any previous training state.
    restore(setup)

    log.debug(f'broadcast params {rank}:{role}')

    # All processes wait at the barrier until all of them have reached this point.
    bcast_barrier.wait()

    # The parameters of the model are broadcasted from the learner process (src) to all other processes.
    for p in setup.model.parameters():
        dist.broadcast(p, src=cfg.distributed.num_learners)

    # All processes wait until all other processes have reached this point.
    dist.barrier()

    log.debug('done')

    # The function for evaluating the model is stored in the setup.
    setup.eval_fn = eval_mfdim

    # The role of the agent is stored.
    agent.role = role

    try:
        # If this process is an actor, it starts the actor training loop.
        if role == 'actor':
            hucc.set_checkpoint_fn(checkpoint, setup)
            train_loop_mfdim_actor(setup)
        else:
            # If this process is a learner, it starts the learner training loop.
            log.debug(f'start leaner with queue {rank}')
            train_loop_mfdim_learner(setup, setup.queues[rank])
    except:
        log.exception('Error in training loop')
        raise

    # After training, any resources used in the setup are cleaned up.
    setup.close()

