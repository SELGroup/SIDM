"""
## Homework: Reinforcement Learning Utilities

Welcome to this week's programming assignment! It's time to get your hands dirty 
and practice writing some utility functions for Reinforcement Learning (RL) models.
This homework is designed to be fun and challenging. Remember, it's not about the destination; 
it's about the journey. So, let's get started!

Also, remember the first rule of coding: "Always comment your code" (Actually, 
this might not be the first rule, but it's still pretty important!)

I've provided some starter code below. Your job is to fill in the gaps, following the prompts in the comments.

Please submit your completed .py file by next week. And don't hesitate to ask questions if you're confused. 
After all, the only bad question is the one that isn't asked!

Alright, folks! Let's get coding!
"""

import logging
import re
import signal
from collections import defaultdict
from types import FrameType, SimpleNamespace
from typing import Dict

import torch as th
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.optim import Optimizer

log = logging.getLogger(__name__)

def make_optim(optim_cfg: DictConfig, model: nn.Module) -> SimpleNamespace:
    """
    TODO: Complete this function to create an optimizer based on the provided configuration 
    and model parameters. This function should be able to handle nested optimizers.
    The 'optim_cfg' parameter is a dictionary-like configuration object (from Hydra/OmegaConf).
    'model' is a PyTorch module, which can have nested submodules.
    The function should return a SimpleNamespace object containing created optimizers.
    """

    # Define a helper function to create a single optimizer
    def make(ocfg: DictConfig, model: nn.Module):
        # TODO: Implement this function
        pass

    # Define a recursive function to handle nested optimizers
    def recurse(ocfg: DictConfig, model: nn.Module):
        # TODO: Implement this function
        pass

    # TODO: Use the recursive function to handle the optim_cfg and model
    pass


def set_checkpoint_fn(fn, *args, **kwargs):
    """
    TODO: This function should set a checkpoint handler that will be called when a SIGUSR1 signal is received.
    'fn' is the function to call when the signal is received. '*args' and '**kwargs' are the arguments to pass to 'fn'.
    You should use the 'signal' module for this.
    """

    # TODO: Define a signal handler function
    def sigusr1(signum: signal.Signals, frame: FrameType):
        # TODO: Implement this function
        pass

    # TODO: Set the signal handler for SIGUSR1
    pass


def dim_select(input: th.Tensor, dim: int, index: th.Tensor):
    """
    TODO: This function should select specific dimensions from an input tensor based on an index tensor.
    'input' is the input tensor, 'dim' is the dimension to select from, and 'index' is the index tensor.
    The function should return a new tensor with the selected dimensions.
    """

    # TODO: Implement this function
    pass


def sorted_nicely(l):
    """
    TODO: This function should sort the elements of a list in a way that humans would expect.
    For example, ['item10', 'item2', 'item1'] should be sorted as ['item1', 'item2', 'item10'].
    """

    # TODO: Implement this function
    pass


def sorted_nicely_sep(l, sep=','):
        """
    TODO: This function should sort the elements of a list similarly to 'sorted_nicely', 
    but it should handle strings containing separators (like commas). 
    'l' is the list to sort, and 'sep' is the separator character.
    You might find it helpful to use the 'split' string method and the 'defaultdict' class from the 'collections' module.
    """
    
    # TODO: Implement this function
    pass

"""
### Hints and information

1. **Optimizers in PyTorch**:
    PyTorch provides several optimization algorithms that you can use to train your models. 
    These are implemented in the `torch.optim` module. To create an optimizer, you need to 
    pass the parameters of the model that the optimizer should update, and the learning rate.

2. **Signal handling in Python**:
    Python's `signal` module provides mechanisms to handle various types of system signals, 
    including the SIGUSR1 signal. You can use the `signal` function from this module to register 
    a handler function that will be called when a specific signal is received.

3. **Tensor indexing in PyTorch**:
    PyTorch provides several ways to index tensors. The simplest way is to use square brackets and 
    indices. However, if you want to perform more complex indexing, you can use the `index_select` 
    or `gather` methods. You will need to use one of these methods in the `dim_select` function.

4. **Sorting strings naturally**:
    Python's built-in `sorted` function sorts strings lexicographically, which might not be what 
    you want if your strings contain numbers. The `sorted_nicely` and `sorted_nicely_sep` functions 
    should sort strings in a way that humans would expect, e.g., "2" should come before "10".

5. **Recursive functions**:
    A recursive function is a function that calls itself during its execution. This allows the function 
    to be written in a more readable and elegant manner, at the expense of potentially higher memory usage. 
    You will need to write a recursive function to handle the potentially nested optimizers in the 
    `make_optim` function.

Good luck with your assignment! Remember, the key to success is to never stop asking questions.
"""
