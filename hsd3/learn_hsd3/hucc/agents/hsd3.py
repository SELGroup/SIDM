# homework.py

import json
import logging
from copy import copy, deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import gym
import hydra
import numpy as np
import torch as th
import torch.distributions as D
from omegaconf import DictConfig
from torch import nn
from torch.nn import functional as F

from hucc import ReplayBuffer
from hucc.agents import Agent
from hucc.envs.ctrlgs import CtrlgsPreTrainingEnv
from hucc.envs.goal_spaces import subsets_task_map
from hucc.models import TracedModule
from hucc.utils import dim_select, sorted_nicely_sep

log = logging.getLogger(__name__)

def _parse_list(s, dtype):
    '''
    This function takes a string and a data type as input and returns a list
    of values of that data type.
    
    Parameters:
    - s: A string of values separated by '#'.
    - dtype: The data type to which the values should be converted.
    
    Returns:
    - A list of values of the specified data type.
    
    TODO: Your task is to implement this function. Make sure to handle the case
    where s is None (in which case the function should return an empty list).
    '''
    raise NotImplementedError()

class DeterministicLo(nn.Module):
    '''
    This class is a wrapper for a policy model that makes it deterministic 
    by always choosing the action with the highest mean.
    '''
    
    def __init__(self, pi: nn.Module):
        '''
        Constructor for the DeterministicLo class.
        
        Parameters:
        - pi: The policy model to be made deterministic.
        
        TODO: Your task is to implement this constructor. You should store the 
        policy model in an instance variable for use in the forward() method.
        '''
        raise NotImplementedError()

    def forward(self, x):
        '''
        This method is called to choose an action for the given observation.
        
        Parameters:
        - x: The observation for which an action should be chosen.
        
        Returns:
        - The action with the highest mean according to the policy model.
        
        TODO: Your task is to implement this method. You should use the policy 
        model stored in the instance variable to choose and return an action.
        '''
        raise NotImplementedError()

# __init__(self, env, cfg: DictConfig): The constructor of the class that takes in the environment and a configuration dictionary as arguments. It prepares all the necessary variables, configurations, and spaces based on the input configuration. It also sets up goal spaces and task maps.

# parse_lo_info(cfg): This static method loads low-level information from a provided configuration. It returns subsets and task maps.

# action_mask_hi(self): This method generates a mask for high-level actions based on subsets of features in the goal space.

# gs_obs(self, obs): This method returns the goal space observations from the overall observations.

# translate(self, gs_obs, task, subgoal, delta_gs_obs=None): This method translates the goal space observations, task and subgoal into the low-level policy's action space.

# update_bp_subgoal(self, gs_obs, next_gs_obs, action_hi): This method updates the backprojected subgoal based on the current and next goal space observations, and the high-level action.

# observation_lo(self, o_obs, action_hi): This method constructs a low-level observation from the overall observation and the high-level action.

# dist_lo(self, gs_obs, task, subgoal): This method calculates the distance between the projected current state and the subgoal in the low-level policy's action space.

# reward_lo(self, gs_obs, next_gs_obs, task, subgoal): This method calculates a potential-based reward for the low-level policy.

# log_goal_spaces(self): This method logs the information about the goal spaces.
class HiToLoInterface:
    '''
    This describes the interface between the high- and low-level sub-agents used
    in HSD3Agent. The goal space handling is a bit involved but needs to stay
    close to the pre-training setup. This interface generally works with a
    number of linear combinations of features; in practice, we manually
    construct these combinatins from a number of specified features.
    '''

    def __init__(self, env, cfg: DictConfig):
        gscfg = cfg.goal_space
        lo_subsets: Optional[List[str]] = None
        lo_task_map: Optional[Dict[str, int]] = None
        try:
            lo_subsets, lo_task_map = self.parse_lo_info(cfg)
        except FileNotFoundError:
            pass

        if gscfg.subsets == 'from_lo':
            subsets, task_map = lo_subsets, lo_task_map
        else:
            subsets, task_map = subsets_task_map(
                features=gscfg.features,
                robot=gscfg.robot,
                spec=gscfg.subsets,
                rank_min=gscfg.rank_min,
                rank_max=gscfg.rank_max,
            )
            if lo_task_map is not None:
                task_map = lo_task_map
        if subsets is None or task_map is None or len(subsets) == 0:
            raise ValueError('No goal space subsets selected')

        self.task_map = task_map
        self.subsets = [s.replace('+', ',') for s in subsets]
        # XXX Unify
        for i in range(len(self.subsets)):
            su = []
            for f in self.subsets[i].split(','):
                if not f in su:
                    su.append(f)
            self.subsets[i] = ','.join(su)
        self.robot = gscfg.robot
        self.features = gscfg.features
        self.delta_actions = bool(gscfg.delta_actions)
        self.mask_gsfeats = _parse_list(gscfg.mask_feats, int)
        n_subsets = len(self.subsets)

        n_obs = env.observation_space['observation'].shape[0]
        self.max_rank = max((len(s.split(',')) for s in self.subsets))
        ng = max(max(map(int, s.split(','))) for s in self.subsets) + 1
        task_space = gym.spaces.Discrete(n_subsets)
        subgoal_space = gym.spaces.Box(
            low=-1, high=1, shape=(ng,), dtype=np.float32
        )
        self.action_space_hi = gym.spaces.Dict(
            [('task', task_space), ('subgoal', subgoal_space)]
        )
        self.action_space_hi.seed(gscfg.seed)
        self.task = th.zeros((n_subsets, len(self.task_map)), dtype=th.float32)
        for i, s in enumerate(self.subsets):
            for j, dim in enumerate(s.split(',')):
                self.task[i][self.task_map[dim]] = 1

        # XXX A very poor way of querying psi etc -- unify this.
        fdist = {a: 1.0 for a in self.subsets}
        dummy_env = CtrlgsPreTrainingEnv(
            gscfg.robot,
            gscfg.features,
            feature_dist=fdist,
            task_map=self.task_map,
        )
        self.gobs_space = dummy_env.observation_space.spaces['gs_observation']
        self.gobs_names = dummy_env.goal_featurizer.feature_names()
        self.goal_space = dummy_env.observation_space.spaces['desired_goal']
        self.delta_feats = dummy_env.goal_space['delta_feats']
        self.twist_feats = [
            self.task_map[str(f)] for f in dummy_env.goal_space['twist_feats']
        ]
        self.psi = dummy_env.psi
        self.offset = dummy_env.offset
        self.psi_1 = dummy_env.psi_1
        self.offset_1 = dummy_env.offset_1
        self.obs_mask = dummy_env.obs_mask
        self.task_idx = dummy_env.task_idx
        gsdim = self.psi.shape[0]
        dummy_env.close()

        self.observation_space_lo = gym.spaces.Dict(
            {
                'observation': gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(len(self.obs_mask),),
                    dtype=np.float32,
                ),
                'desired_goal': self.goal_space,
                'task': gym.spaces.Box(
                    low=0, high=1, shape=(len(self.task_map),), dtype=np.float32
                ),
            }
        )
        spaces = copy(env.observation_space.spaces)
        # Ignore the goal space in both policies
        self.gs_key = gscfg.key
        del spaces[self.gs_key]
        self.observation_space_hi = gym.spaces.Dict(spaces)

        # Inverse psi matrix indexed by available subsets
        self.psi_1_by_ss = th.zeros(
            (n_subsets, self.max_rank, gsdim), dtype=th.float32
        )
        self.psi_by_ss = th.zeros(
            (n_subsets, self.max_rank, gsdim), dtype=th.float32
        )
        self.offset_by_ss = th.zeros(
            (n_subsets, self.max_rank), dtype=th.float32
        )
        for i, s in enumerate(self.subsets):
            for j, dim in enumerate(s.split(',')):
                self.psi_1_by_ss[i][j] = th.tensor(self.psi_1[int(dim)])
                self.psi_by_ss[i][j] = th.tensor(self.psi[int(dim)])
                self.offset_by_ss[i][j] = self.offset[int(dim)]

        device = cfg.device
        self.psi_1_by_ss = self.psi_1_by_ss.to(device)
        self.psi_by_ss = self.psi_by_ss.to(device)
        self.offset_by_ss = self.offset_by_ss.to(device)
        self.offset_1 = th.tensor(
            self.offset_1, device=device, dtype=th.float32
        )
        self.task = self.task.to(device)
        self.subgoal_idxs = []
        for s in self.subsets:
            self.subgoal_idxs.append([self.task_map[f] for f in s.split(',')])

    @staticmethod
    def parse_lo_info(cfg):
        p = Path(cfg.lo.init_from)
        abs_path = str(p.with_name(p.stem + '_abs.json'))
        log.debug(f'Loading subset information from checkpoint {abs_path}')
        with open(abs_path, 'rt') as ft:
            d = json.load(ft)
            if 'task_map' in d:
                task_map = d['task_map']
            cperf = d['cperf']
            if 'total' in cperf:
                del cperf['total']
        eps = cfg.goal_space.from_lo_eps
        subsets = sorted_nicely_sep(
            list(set([k for k, v in cperf.items() if v >= 1.0 - eps]))
        )
        if cfg.goal_space.rank_min > 0:
            subsets = [
                a
                for a in subsets
                if len(a.split(',')) >= cfg.goal_space.rank_min
            ]
        if cfg.goal_space.rank_max > 0:
            subsets = [
                a
                for a in subsets
                if len(a.split(',')) <= cfg.goal_space.rank_max
            ]
        return subsets, task_map

    def action_mask_hi(self):
        nt = self.action_space_hi['task'].n
        ng = self.action_space_hi['subgoal'].shape[0]
        mask = th.zeros(nt, ng)

        for i, feats in enumerate(self.subgoal_idxs):
            mask[i, feats] = 1
        for f in self.mask_gsfeats:
            mask[:, self.task_map[str(f)]] = 0

        return mask.to(self.task.device)

    def gs_obs(self, obs):
        return obs[self.gs_key]

    def translate(self, gs_obs, task, subgoal, delta_gs_obs=None):
        # Copy subgoal features to the front for compatibility with projections.
        subgoal_s = th.zeros(
            (subgoal.shape[0], self.max_rank), device=subgoal.device
        )
        for i in range(task.shape[0]):
            sg = subgoal[i, self.subgoal_idxs[task[i].item()]]
            subgoal_s[i, : sg.shape[0]] = sg

        if self.delta_actions:
            # Subgoal is projected current state plus the specified action
            proj_obs = (
                th.bmm(
                    gs_obs.unsqueeze(1),
                    self.psi_by_ss.index_select(0, task).transpose(1, 2),
                ).squeeze(1)
                + self.offset_by_ss.index_select(0, task)
            )
            subgoal_s = proj_obs + subgoal_s

        # Backproject absolute subgoal into observation space
        bproj_goal = (
            th.bmm(
                subgoal_s.unsqueeze(1), self.psi_1_by_ss.index_select(0, task)
            ).squeeze(1)
            + self.offset_1
        )
        # Add delta features from current state; with delta actions, this has
        # already been taken care of via proj_obs.
        if delta_gs_obs is not None:
            raise RuntimeError('Umm no idea what this should do??')
            bproj_goal[:, self.delta_feats] += delta_gs_obs[:, self.delta_feats]
        elif not self.delta_actions:
            bproj_goal[:, self.delta_feats] += gs_obs[:, self.delta_feats]
        task_rep = self.task.index_select(0, task)
        # Give desired delta to ground truth
        goal = bproj_goal[:, self.task_idx] - gs_obs[:, self.task_idx]
        if len(self.twist_feats) > 0:
            twf = self.twist_feats
            goal[:, twf] = (
                th.remainder(
                    (
                        bproj_goal[:, self.task_idx][:, twf]
                        - gs_obs[:, self.task_idx][:, twf]
                    )
                    + np.pi,
                    2 * np.pi,
                )
                - np.pi
            )
        goal = goal * task_rep
        return {
            'desired_goal': goal,
            'task': task_rep,
        }

    # Update backprojected subgoal
    def update_bp_subgoal(self, gs_obs, next_gs_obs, action_hi):
        upd = (
            gs_obs[:, self.task_idx]
            - next_gs_obs[:, self.task_idx]
            + action_hi['desired_goal']
        ) * action_hi['task']
        if len(self.twist_feats) > 0:
            twf = self.twist_feats
            upd[:, twf] = (
                th.remainder(
                    (
                        gs_obs[:, self.task_idx][:, twf]
                        - next_gs_obs[:, self.task_idx][:, twf]
                        + action_hi['desired_goal'][:, twf]
                    )
                    + np.pi,
                    2 * np.pi,
                )
                - np.pi
            ) * action_hi['task'][:, twf]
        return upd

    def observation_lo(self, o_obs, action_hi):
        return {
            'observation': o_obs[:, self.obs_mask],
            'desired_goal': action_hi['desired_goal'],
            'task': action_hi['task'],
        }

    def observation_hi(self, obs):
        tobs = copy(obs)
        del tobs[self.gs_key]
        return tobs

    def dist_lo(self, gs_obs, task, subgoal):
        subgoal_s = th.zeros_like(subgoal)
        for i in range(task.shape[0]):
            sg = subgoal[i, self.subgoal_idxs[task[i].item()]]
            subgoal_s[i, : sg.shape[0]] = sg

        proj_obs = (
            th.bmm(
                gs_obs.unsqueeze(1),
                self.psi_by_ss.index_select(0, task).transpose(1, 2),
            ).squeeze(1)
            + self.offset_by_ss.index_select(0, task)
        )
        return th.linalg.norm(subgoal_s - proj_obs, ord=2, dim=1)

    # Potential-based reward for low-level policy
    def reward_lo(self, gs_obs, next_gs_obs, task, subgoal):
        subgoal_s = th.zeros_like(subgoal)
        for i in range(task.shape[0]):
            sg = subgoal[i, self.subgoal_idxs[task[i].item()]]
            subgoal_s[i, : sg.shape[0]] = sg

        proj_obs = (
            th.bmm(
                gs_obs.unsqueeze(1),
                self.psi_by_ss.index_select(0, task).transpose(1, 2),
            ).squeeze(1)
            + self.offset_by_ss.index_select(0, task)
        )
        proj_next_obs = (
            th.bmm(
                next_gs_obs.unsqueeze(1),
                self.psi_by_ss.index_select(0, task).transpose(1, 2),
            ).squeeze(1)
            + self.offset_by_ss.index_select(0, task)
        )
        d = th.linalg.norm(subgoal - proj_obs, ord=2, dim=1)
        dn = th.linalg.norm(subgoal - proj_next_obs, ord=2, dim=1)
        return d - dn

    def log_goal_spaces(self):
        log.info(f'Considering {len(self.subsets)} goal space subsets')
        for i, s in enumerate(self.subsets):
            name = ','.join(
                [
                    CtrlgsPreTrainingEnv.feature_name(
                        self.robot, self.features, int(f)
                    )
                    for f in s.split(',')
                ]
            )
            log.info(f'Subset {i}: {name} ({s})')


class HSD3Agent(Agent):
    '''
    A HRL agent that can leverage a low-level policy obtained via hierarchical
    skill discovery. It implements a composite action space for chosing (a) a
    goal space subset and (b) an actual goal. Soft Actor-Critic is used for
    performing policy and Q-function updates.

    The agent unfortunately depends on

    This implementation also features the following:
    - Dense high-level updates as in DynE
    '''

    import gym
import torch.nn as nn
import torch as th
from types import SimpleNamespace
from omegaconf import DictConfig
from copy import deepcopy
import hydra
import numpy as np
import logging as log

from YOUR_MODULE_PATH import HiToLoInterface, ReplayBuffer, DeterministicLo, TracedModule


class HSD3Agent(nn.Module):
    """
    Your task is to implement a Hierarchical Soft-Actor Critic with Discrete and Continuous actions (HSD3Agent)
    in Reinforcement Learning. The HSD3Agent class in this file has been started for you, but you will need
    to fill in the missing parts marked with `# TODO`.

    The HSD3Agent uses an interface between high-level and low-level sub-agents, likely within a 
    Hierarchical Reinforcement Learning (HRL) context. HRL is a type of reinforcement learning architecture 
    that allows for learning and decision-making at various levels of abstraction. It typically involves 
    high-level policies (or sub-agents) making more abstract, strategic decisions, and low-level policies 
    carrying out the specific actions to achieve those strategic goals.
    """
    def __init__(
        self,
        env: gym.Env,
        model: nn.Module,
        optim: SimpleNamespace,
        cfg: DictConfig,
    ):
        """
        The __init__ method initializes the HSD3Agent class.
        
        The parameters are:
            env: An instance of an OpenAI gym environment.
            model: The model that the agent will use for decision making.
            optim: A namespace that contains the optimizer that the model will use for learning.
            cfg: Configuration parameters that the agent needs to properly initialize.
            
        Your task: Fill in the code for the initialization. Pay attention to the TODO comments.
        """
        super().__init__(cfg)

        # TODO: Check if the model has the necessary modules. If not, raise a ValueError.
        # Here are the modules you need to check for:
        # 1. 'hi'
        # 2. 'lo'
        # 3. 'hi.pi_subgoal'
        # 4. 'hi.pi_task'
        # 5. 'hi.q'
        # 6. 'lo.pi'

        # TODO: Check if the environment's action_space is of type gym.spaces.Box, if not, raise a ValueError.

        # TODO: Check if the environment's observation_space is of type gym.spaces.Dict, if not, raise a ValueError.

        # TODO: Check if the observation_space of the environment has the following keys: 'time', 'observation', 
        # and the key from cfg.goal_space.key. If not, raise a ValueError for each missing key.

        # TODO: Initialize the HiToLoInterface with the environment and the configuration, 
        # and call the log_goal_spaces method.

        # TODO: Initialize the action spaces for the subgoal ('subgoal') and task ('task').

        # TODO: Initialize the model for the high-level policy for subgoal and task, 
        # and their respective optimizers.

        # TODO: Initialize other necessary parameters such as batch_size, gamma, polyak, rpbuf_size, 
        # samples_per_update, num_updates, warmup_samples, randexp_samples, clip_grad_norm, action_interval

    # Alright students, let's keep going! Now we'll be working on more static methods and agent action functions. 


    # TODO: Define a static method `effective_observation_space` which takes two parameters, `env` and `cfg`.
    # The purpose of this function is to return the effective observation spaces for both hi and lo levels.
    @staticmethod
    def effective_observation_space(env: gym.Env, cfg: DictConfig):  # Don't forget to import necessary libraries!
        # TODO: Instantiate a HiToLoInterface with `env` and `cfg`, assign it to `iface`.

        # TODO: Return a dictionary contains 'lo' and 'hi' keys.
        # For 'lo' key, it should return `iface.observation_space_lo`.
        # For 'hi' key, it should be a dictionary contains 'pi_task', 'pi_subgoal', 'q' keys, 
        # with corresponding values to be filled according to the context of this function.
        pass

    # TODO: Similar to the previous function, define a static method `effective_action_space` which takes two parameters, `env` and `cfg`.
    # This function should return the effective action spaces for both hi and lo levels.
    @staticmethod
    def effective_action_space(env: gym.Env, cfg: DictConfig):
        # TODO: Complete this function.
        pass

    # TODO: Define a function `action_hi_d_qinput`, it takes one parameter `action_d` and should return a tensor.
    def action_hi_d_qinput(self, action_d: th.Tensor) -> th.Tensor:
        # TODO: Compute and return the result tensor.
        pass

    # TODO: Now we will create action functions for hi and lo levels, and an overall action function.
    # Start with `action_hi_rand`, it takes two parameters `env` and `time`, and should return a dictionary.
    def action_hi_rand(self, env, time):
        # TODO: Complete this function.
        pass

    # TODO: Define another action function for hi level named `action_hi_cd`, it takes two parameters `env` and `obs`.
    def action_hi_cd(self, env, obs):
        # TODO: Complete this function.
        pass

    # TODO: Now define a function `action_hi` to select the high-level action. 
    # This function takes three parameters: `env`, `obs`, and `prev_action`.
    def action_hi(self, env, obs, prev_action):
        # TODO: Complete this function.
        pass

    # TODO: Define a function `action_lo` to select the low-level action, it takes two parameters: `env` and `obs`.
    def action_lo(self, env, obs):
        # TODO: Complete this function.
        pass

    # TODO: Finally, define a function `action` to select the overall action.
    # This function takes two parameters: `env` and `obs`, and should return a tuple.
    def action(self, env, obs) -> Tuple[th.Tensor, Any]:
        # TODO: Complete this function.
        pass

# Congratulations! You have completed the transformation of the real-world code into your homework. 
# Please try to understand the code and complete the TODOs according to the comments and your understanding.
# Don't forget to test your code after completion. Good luck!

    def step(
        self,
        env,
        obs,
        action,
        extra: Any,
        result: Tuple[th.Tensor, th.Tensor, th.Tensor, List[Dict]],
    ) -> None:
        """
        This function collects data from the environment by stepping through it with the provided action.
        The observed states, actions, and rewards are then stored in a staging buffer. If the staging buffer
        reaches its maximum capacity, the data is transferred to the main buffer. Moreover, if a certain number
        of steps have been taken, the agent's policy is updated using the collected data.
        """
        next_obs, reward, done, info = result
        action_hi = extra['action_hi']
        tr_action_hi = extra['tr_action_hi']
        obs_hi = extra['obs_hi']
        # Ignore terminal state if we have a timeout
        fell_over = th.zeros_like(done, device='cpu')
        for i in range(len(info)):
            if 'TimeLimit.truncated' in info[i]:
                # log.info('Ignoring timeout')
                done[i] = False
            elif 'fell_over' in info[i]:
                fell_over[i] = True
        fell_over = fell_over.to(done.device)

        d = dict(
            terminal=done,
            step=obs['time'].remainder(self._action_interval).long(),
        )
        for k, v in action_hi.items():
            d[f'action_hi_{k}'] = v
        for k in self._obs_keys:
            d[f'obs_{k}'] = obs_hi[k]
            if k != 'prev_task' and k != 'prev_subgoal':
                d[f'next_obs_{k}'] = next_obs[k]
        d['reward'] = reward

        self._staging.put_row(d)
        self._cur_rewards.append(reward)

        if self._staging.size == self._staging.max:
            self._staging_to_buffer()

        self._n_steps += 1
        self._n_samples += done.nelement()
        self._n_samples_since_update += done.nelement()
        ilv = self._staging.interleave
        if self._buffer.size + self._staging.size - ilv < self._warmup_samples:
            return
        if self._n_samples_since_update >= self._samples_per_update:
            self.update()
            self._cur_rewards.clear()
            self._n_samples_since_update = 0

    def _staging_to_buffer(self):
        """
        This function moves data from the staging buffer to the main buffer. It also pre-processes the data to be
        suitable for training. This includes stacking several transitions together, calculating the next high-level
        action to take, summing up discounted rewards, and creating a dictionary containing the current and next
        state, the summed reward, and whether the episode ended or not. This dictionary is then put into the main buffer.
        """
        ilv = self._staging.interleave
        buf = self._staging
        assert buf._b is not None
        c = self._action_interval
        # Stack at least two transitions because for training the low-level
        # policy we'll need the next high-level action.
        n_stack = max(c, 2)
        batch: Dict[str, th.Tensor] = dict()
        idx = (
            buf.start + th.arange(0, ilv * n_stack, device=buf.device)
        ) % buf.max
        for k in set(buf._b.keys()):
            b = buf._b[k].index_select(0, idx)
            b = b.view((n_stack, ilv) + b.shape[1:]).transpose(0, 1)
            batch[k] = b

        # c = action_freq
        # i = batch['step']
        # Next action at c - i steps further, but we'll take next_obs so
        # access it at c - i - 1
        next_action_hi = (c - 1) - batch['step'][:, 0]
        # If we have a terminal before, use this instead
        terminal = batch['terminal'].clone()
        for j in range(1, c):
            terminal[:, j] |= terminal[:, j - 1]
        first_terminal = c - terminal.sum(dim=1)
        # Lastly, the episode could have ended with a timeout, which we can
        # detect if we took another action_hi (i == 0) prematurely. This will screw
        # up the reward summation, but hopefully it doesn't hurt too much.
        next_real_action_hi = th.zeros_like(next_action_hi) + c
        for j in range(1, c):
            idx = th.where(batch['step'][:, j] == 0)[0]
            next_real_action_hi[idx] = next_real_action_hi[idx].clamp(0, j - 1)
        next_idx = th.min(
            th.min(next_action_hi, first_terminal), next_real_action_hi
        )

        # Sum up discounted rewards until next c - i - 1
        reward = batch['reward'][:, 0].clone()
        for j in range(1, c):
            reward += self._gamma ** j * batch['reward'][:, j] * (next_idx >= j)

        not_done = th.logical_not(dim_select(batch['terminal'], 1, next_idx))
        obs = {k: batch[f'obs_{k}'][:, 0] for k in self._obs_keys}
        obs['time'] = batch['step'][:, 0:1].clone()
        obs_p = {
            k: dim_select(batch[f'next_obs_{k}'], 1, next_idx)
            for k in self._obs_keys
        }
        obs_p['time'] = obs_p['time'].clone().unsqueeze(1)
        obs_p['time'].fill_(0)

        gamma_exp = th.zeros_like(reward) + self._gamma
        gamma_exp.pow_(next_idx + 1)

        db = dict(
            reward=reward,
            not_done=not_done,
            terminal=batch['terminal'][:, 0],
            gamma_exp=gamma_exp,
        )
        db[f'action_hi_{self._dkey}'] = batch[f'action_hi_{self._dkey}'][:, 0]
        db[f'action_hi_{self._ckey}'] = batch[f'action_hi_{self._ckey}'][:, 0]
        for k, v in obs.items():
            db[f'obs_{k}'] = v
        for k, v in obs_p.items():
            db[f'next_obs_{k}'] = v

        self._buffer.put_row(db)

    # The '_update' function is used to update the parameters of the models being trained.
    def _update(self):
        # Nested function that returns the action and its log probability given an observation and mask.
        def act_logp_c(obs, mask):
            # Pass the observations through the policy model to get a distribution over actions.
            dist = self._model_pi_c(obs)
            # Sample an action from the distribution.
            action = dist.rsample()
            # If mask is not None, apply it to the action and its log probability.
            if mask is not None:
                log_prob = (dist.log_prob(action) * mask).sum(dim=-1) / mask.sum(dim=-1)
                action = action * mask * self._action_factor_c
            else:
                log_prob = dist.log_prob(action).sum(dim=-1)
                action = action * self._action_factor_c
            return action, log_prob

        # Nested function that calculates the target value for the Q function.
        def q_target(batch):
            reward = batch['reward']
            not_done = batch['not_done']
            # Extract the observations from the batch.
            obs_p = {k: batch[f'next_obs_{k}'] for k in self._obs_keys}
            # The temperatures for the two entropy terms in the objective.
            alpha_c = self._log_alpha_c.detach().exp()
            alpha_d = self._log_alpha_d.detach().exp()
            bsz = reward.shape[0]
            d_batchin = self._d_batchin.narrow(0, 0, bsz * nd)
            c_batchmask = self._c_batchmask.narrow(0, 0, bsz * nd)

            # Distribution over discrete actions.
            dist_d = self._model_pi_d(obs_p)
            # Continuous action and its log probability.
            action_c, log_prob_c = act_logp_c(obs_p, self._action_c_mask)

            # If the expected number of discrete actions is -1 and there is more than one discrete action.
            if self._expectation_d == -1 and nd > 1:
                # Modify the observations to include the continuous and discrete actions.
                obs_pe = {k: v.repeat_interleave(nd, dim=0) for k, v in obs_p.items()}
                obs_pe[self._dkey] = d_batchin
                obs_pe[self._ckey] = action_c.view(d_batchin.shape[0], -1)
                # The target Q value is the minimum over the two Q functions.
                q_t = th.min(self._target.hi.q(obs_pe), dim=-1).values

                q_t = q_t.view(bsz, nd)
                log_prob_c = log_prob_c.view(bsz, nd)
                # The estimate of the state value function.
                v_est = (dist_d.probs * (q_t - log_prob_c * alpha_c)).sum(dim=-1) + alpha_d * (dist_d.entropy() - self._uniform_entropy_d)
            else:
                # Sample actions from the discrete action distribution.
                action_d = th.multinomial(dist_d.probs, nds, replacement=True)
                # Get the log probability of the sampled actions.
                log_prob_d = dist_d.logits.gather(1, action_d)

                # Modify the observations to include the continuous and discrete actions.
                obs_pe = {k: v.repeat_interleave(nds, dim=0) if nds > 1 else v for k, v in obs_p.items()}
                obs_pe[self._dkey] = self.action_hi_d_qinput(action_d).view(
                    -1, nd
                )

                action_c = dim_select(action_c, 1, action_d).view(
                    -1, action_c.shape[-1]
                )
                log_prob_c = log_prob_c.gather(1, action_d)
                obs_pe[self._ckey] = action_c

                q_t = th.min(self._target.hi.q(obs_pe), dim=-1).values.view(
                    -1, nds
                )
                log_prob_c = log_prob_c.view(-1, nds)
                # Initialize a mask for action_c if it's None
                if self._action_c_mask is not None:
                    self._c_batchmask = self._action_c_mask.index_select(
                        1, th.arange(bsz * nd, device=mdevice).remainder(nd)
                    ).squeeze(0)
                else:
                    self._c_batchmask = None

                # If the dyne updates flag is not set, assert that the buffer is either empty or full
                if not self._dyne_updates:
                    assert (
                        self._buffer.start == 0 or self._buffer.size == self._buffer.max
                    )
                    # Get indices where observation time is 0
                    indices = th.where(
                        self._buffer._b['obs_time'][: self._buffer.size] == 0
                    )[0]

                gbatch = None
                # If dyne updates is enabled and batch size is less than 512, get a batch from the buffer
                if self._dyne_updates and self._bsz < 512:
                    gbatch = self._buffer.get_batch(
                        self._bsz * self._num_updates,
                        device=mdevice,
                    )

                # Start updates
                for i in range(self._num_updates):
                    # If dyne updates are enabled, handle the batch accordingly
                    if self._dyne_updates:
                        if gbatch is not None:
                            batch = {
                                k: v.narrow(0, i * self._bsz, self._bsz)
                                for k, v in gbatch.items()
                            }
                        else:
                            batch = self._buffer.get_batch(
                                self._bsz,
                                device=mdevice,
                            )
                    else:
                        # Get batch where indices are as defined before
                        batch = self._buffer.get_batch_where(
                            self._bsz, indices=indices, device=mdevice
                        )

                    # Get observations from the batch
                    obs = {k: batch[f'obs_{k}'] for k in self._obs_keys}
                    # Calculate the exponential of alpha_c and alpha_d
                    alpha_c = self._log_alpha_c.detach().exp()
                    alpha_d = self._log_alpha_d.detach().exp()

                    # Backup for Q-Function
                    # Q-function target is calculated with no gradient tracking
                    with th.no_grad():
                        backup = q_target(batch)

                    # Q-Function update
                    # Prepare inputs for Q-function
                    q_in = copy(obs)
                    q_in[self._dkey] = self.action_hi_d_qinput(
                        batch[f'action_hi_{self._dkey}']
                    )
                    q_in[self._ckey] = batch[f'action_hi_{self._ckey}']
                    q = self._q_hi(q_in)
                    q1 = q[:, 0]
                    q2 = q[:, 1]
                    # Compute the MSE loss for both Q1 and Q2
                    q1_loss = F.mse_loss(q1, backup, reduction='none')
                    q2_loss = F.mse_loss(q2, backup, reduction='none')
                    q_loss = q1_loss.mean() + q2_loss.mean()
                    # Zero out the gradients of the Q-function optimizer
                    self._optim.hi.q.zero_grad()
                    # Backpropagate the Q-function loss
                    q_loss.backward()
                    # If gradient clipping is used, apply it before the optimization step
                    if self._clip_grad_norm > 0.0:
                        nn.utils.clip_grad_norm_(
                            self._model.q.parameters(), self._clip_grad_norm
                        )
                    # Optimization step for the Q-function
                    self._optim.hi.q.step()

                    # Policy update
                    # Prepare inputs for the policy function
                    with th.no_grad():
                        p_in = copy(obs)
                        p_in[self._ckey] = self.action_hi_c_qinput(
                            batch[f'action_hi_{self._ckey}']
                        )
                    # Get actions from the policy function
                    action = self._model.hi.p(p_in)
                    action_d = action[:, : self._d_dim]
                    action_c = action[:, self._d_dim :]
                    # Get action values from Q-function
                    q_in = copy(obs)
                    q_in[self._dkey] = action_d
                    q_in[self._ckey] = action_c
                    q = self._model.hi.q(q_in)
                    q1 = q[:, 0]
                    q2 = q[:, 1]
                    # Get the minimum of Q1 and Q2
                    q = th.min(q1, q2)
                    # Get the mean of Q-values
                    q_mean = q.mean()
                    # Compute the policy loss
                    p_loss = -q_mean
                    # Zero out the gradients of the policy optimizer
                    self._optim.hi.p.zero_grad()
                    # Backpropagate the policy loss
                    p_loss.backward()
                    # If gradient clipping is used, apply it before the optimization step
                    if self._clip_grad_norm > 0.0:
                        nn.utils.clip_grad_norm_(
                            self._model.p.parameters(), self._clip_grad_norm
                        )
                    # Optimization step for the policy
                    self._optim.hi.p.step()

                    # Temperature update
                    alpha_loss_c = -self._log_alpha_c * (
                        batch[f'action_hi_{self._ckey}_logp'] + self._target_entropy_c
                    ).detach().mean()
                    alpha_loss_d = -self._log_alpha_d * (
                        self.action_hi_d_logp(batch[f'action_hi_{self._dkey}'])
                        + self._target_entropy_d
                    ).detach().mean()
                    alpha_loss = alpha_loss_c + alpha_loss_d
                    # Zero out the gradients of the temperature optimizer
                    self._optim.alpha.zero_grad()
                    # Backpropagate the temperature loss
                    alpha_loss.backward()
                    # Optimization step for the temperature
                    self._optim.alpha.step()

                    # Log the losses
                    self._log_losses(i, q_loss, p_loss, alpha_loss, alpha_c, alpha_d)

                    # Update the target network
                    self._model.hi.q_targ.load_state_dict(self._model.hi.q.state_dict())

                return self._get_logs()

