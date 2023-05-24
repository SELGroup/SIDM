

import logging
import math
from collections import defaultdict
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import gym
import torch as th
from omegaconf import DictConfig
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from hucc.agents.utils import batch_2to1  # Make sure you have this module available or you'll need to implement this function too!

log = logging.getLogger(__name__)

class Agent:
    '''
    Welcome students to the wonderful world of Reinforcement Learning (RL)! 
    Today, we are going to implement some key functions of an RL agent.
    
    In this exercise, you're going to complete an Agent class for an RL 
    application. The Agent class has been started for you, but you'll need to 
    fill in the rest.
    
    This agent will interact with an environment, learn from its actions, and 
    hopefully get better over time. Your task will be to implement key methods 
    of this agent, namely `action()`, `step()`, `_update()`, and `_update_v()`.
    
    This exercise will test your understanding of RL concepts, your ability 
    to work with PyTorch, and your ability to follow the structure of a complex class.
    
    Remember, the goal is not just to write code that works (although that is important), 
    but to write code that your fellow students can read, understand, and learn from. 
    So be sure to include clear comments explaining what your code is doing and why.
    '''
    
    # ... (Rest of the provided code)

    def action(self, env, obs) -> Tuple[th.Tensor, Any]:
        '''
        This function will decide which action to take, given an observation.
        
        Parameters:
        - env: The environment instance.
        - obs: Previous observation used to determine the action.
        
        Returns:
        - action: The action that the agent decided to take.
        - extra: Any extra information that you think will be useful. 
                 This will be passed back to you in the step() function.

        TODO: Your task is to implement this function. 
        You will need to determine what action to take based on the observation `obs`.
        '''
        raise NotImplementedError()

    def step(
        self,
        env,
        obs,
        action: th.Tensor,
        extra: Any,
        result: Tuple[Any, th.Tensor, th.Tensor, List[Dict]],
    ) -> None:
        '''
        This function will be called after the agent takes an action, 
        and the environment returns the result.
        
        Parameters:
        - env: The environment instance.
        - obs: Previous observation used to determine the action.
        - action: Action that was taken.
        - extra: The extra value returned from the previous action() call.
        - result: Return value of env.step(), i.e., (next_obs, reward, done, info).
        
        TODO: Your task is to implement this function. 
        You will need to update the agent's state based on the result of its action.
        '''
        raise NotImplementedError()

    def _update(self) -> None:
        '''
        This function is called to update the agent. 
        It's where the learning happens!
        
        TODO: Your task is to implement this function. 
        Use this function to update the
        agent's internal state based on what it has learned so far. 
        You might want to use the PyTorch optimization functions here.
        '''
        raise NotImplementedError()

    def _update_v(
        self,
        model: nn.Module,
        optim: SimpleNamespace,
        obs: th.Tensor,
        ret: th.Tensor,
        train_v_iters: int,
        max_grad_norm: float = math.inf,
    ) -> float:
        '''
        This function will update the value function of the agent, 
        which is a prediction of future rewards.
        
        Parameters:
        - model: The model that the agent is using.
        - optim: The optimizer that the agent is using.
        - obs: The observation that the agent has made.
        - ret: The return that the agent has received.
        - train_v_iters: The number of training iterations.
        - max_grad_norm: The maximum gradient norm.
        
        Returns:
        - loss: The loss value after updating the value function.
        
        TODO: Your task is to implement this function. 
        You need to update the value function of the agent based on the received return.
        '''
        # Assume non-recurrent model and hence flatten observations and returns
        # into a single batch dimension
        obs, ret = batch_2to1(obs), batch_2to1(ret)

        # TODO: Custom baseline update
        # If the model has a 'fit' method, use it to fit the model to the observation and return.
        # After fitting, compute and return the mean squared error loss between model predictions and actual returns.

        ret = ret.detach()
        # TODO: For `train_v_iters` iterations, update the model's value function to minimize the mean squared error loss.
        # Don't forget to zero out the gradients before each step and clip the gradients if `max_grad_norm` is not infinity.
        
        # Return the final loss value
        raise NotImplementedError()
