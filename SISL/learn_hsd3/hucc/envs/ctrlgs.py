# Copyright (c) 2021-present, Facebook, Inc.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
from typing import Dict, List, Tuple

import gym
import numpy as np
import torch as th
from torch import nn

from bisk import BiskSingleRobotEnv
from bisk.features import make_featurizer
from hucc.envs.goal_spaces import g_goal_spaces, g_delta_feats

log = logging.getLogger(__name__)

# The CtrlgsPreTrainingEnv class inherits from the BiskSingleRobotEnv class.
# It represents a multi-task, goal-based pre-training environment where a single robot can be controlled.
class CtrlgsPreTrainingEnv(BiskSingleRobotEnv):
    '''
    A multi-task, goal-based pre-training environment.

    The environment is "empty" except for a single robot that can be controlled.
    The "tasks" consider the control of one or more observed features -- those
    will be sampled according to `feature_dist` (which can also be changed after
    constructing the environment). For each task (combination of features), a
    goal space is constructed using `psi` and `offset`, and goals are sampled in
    this goal space in [-1,1].

    A continual version of this environment can be obtained with a
    `hard_reset_interval` of > 1. This parameter specifices the frequency at
    which the simulation is reset to its initial state. Other resets will simply
    result in a new goal to be sampled.
    '''

    
    def __init__(
        # Constructor for the CtrlgsPreTrainingEnv class.
        # It takes several parameters that help define the environment, including the robot, the features to be controlled, and various task-specific parameters.
        self,
        robot: str,  # The robot to be controlled
        features: str,  # The features to be controlled
        feature_dist: Dict[str, float],  # The distribution for sampling the features
        task_map: Dict[str, int],  # Mapping of tasks
        precision: float = 0.1,  # Precision for controlling features
        idle_steps: int = 0,  # Number of idle steps before the robot starts moving
        max_steps: int = 20,  # Maximum number of steps the robot can take
        backproject_goal: bool = True,  # Whether to backproject the goal or not
        reward: str = 'potential',  # Reward function to use
        hard_reset_interval: int = 1,  # Frequency of hard reset
        reset_p: float = 0.0,  # Reset probability
        resample_features: str = 'hard',  # Method for resampling features
        full_episodes: bool = False,  # Whether to use full episodes or not
        allow_fallover: bool = False,  # Whether to allow the robot to fall over or not
        fallover_penalty: float = -1.0,  # Penalty if the robot falls over
        implicit_soft_resets: bool = False,  # Whether to use implicit soft resets or not
        goal_sampling: str = 'random',  # Method for goal sampling
        ctrl_cost: float = 0.0,  # Control cost
        normalize_gs_observation: bool = False,  # Whether to normalize goal space observation or not
        zero_twist_goals: bool = False,  # Whether to set twist goals to zero or not
        relative_frame_of_reference: bool = False,  # Whether to use relative frame of reference or not
    ):
        # XXX hack to have DMC robots operate with their "native" sensor input
        super().__init__(
            robot=robot,
            features='joints'
            if features not in ('sensorsnoc', 'native')
            else features,
            allow_fallover=allow_fallover,
        )

        # Construct a featurizer for the goal.
        self.goal_featurizer = make_featurizer(
            features, self.p, self.robot, 'robot'
        )

        # Dimension of goal space.
        gsdim = self.goal_featurizer.observation_space.shape[0]

        # Assign the goal space for the robot and the feature.
        self.goal_space = g_goal_spaces[features][robot]
        # Construct the abstraction matrix (psi) and offset for the goal space.
        # These are used to map between the feature space and the goal space.
        self.psi, self.offset = self.abstraction_matrix(robot, features, gsdim)
        
        # Compute the inverse of the abstraction matrix and offset, 
        # which will be used for backprojection from the goal space to the feature space.
        self.psi_1 = np.linalg.inv(self.psi)
        self.offset_1 = -np.matmul(self.offset, self.psi_1)

        # Check that the observation space is 1D and the psi and offset have the correct shapes.
        assert len(self.observation_space.shape) == 1
        assert self.psi.shape == (gsdim, gsdim)
        assert self.offset.shape == (gsdim,)

        # Assign the various parameters as instance variables.
        self.precision = precision
        self.idle_steps = idle_steps
        self.max_steps = max_steps
        self.backproject_goal = backproject_goal
        self.reward = reward
        self.hard_reset_interval = hard_reset_interval
        self.reset_p = reset_p
        self.resample_features = resample_features
        self.full_episodes = full_episodes
        self.fallover_penalty = fallover_penalty
        self.ctrl_cost = ctrl_cost
        self.implicit_soft_resets = implicit_soft_resets
        self.goal_sampling = goal_sampling
        self.normalize_gs_observation = normalize_gs_observation
        self.zero_twist_goals = zero_twist_goals
        self.relative_frame_of_reference = relative_frame_of_reference

        # Create a task index mapping.
        self.task_idx = [0] * len(task_map)
        for k, v in task_map.items():
            self.task_idx[v] = int(k)

        # Check if there are any twist features, and if so, check if their ranges are symmetric.
        # If not, raise an error, as this is currently not supported.
        if len(self.goal_space['twist_feats']) > 0:
            negpi = self.proj(
                -np.pi * np.ones(gsdim), self.goal_space['twist_feats']
            )
            pospi = self.proj(
                np.pi * np.ones(gsdim), self.goal_space['twist_feats']
            )
            if not np.allclose(-negpi, pospi):
                raise ValueError('Twist feature ranges not symmetric')
            self.proj_pi = pospi

        # Set up the goal space, either by backprojecting the goal or by creating a new goal space.
        if backproject_goal:
            all_feats = list(range(gsdim))
            gmin_back = self.backproj(-np.ones(gsdim), all_feats)
            gmax_back = self.backproj(np.ones(gsdim), all_feats)
            goal_space = gym.spaces.Box(gmin_back, gmax_back)
        else:
            max_features = max(
                (
                    len(f.replace('+', ',').split(','))
                    for f in feature_dist.keys()
                )
            )
            goal_space = gym.spaces.Box(
                low=-2, high=2, shape=(max_features,), dtype=np.float32
            )

        # Create a mapping of tasks.
        self.task_map = {int(k): v for k, v in task_map.items()}

        # Hide position-related invariant features from the observation.
        delta_feats = g_delta_feats[robot]
        self.obs_mask = list(range(self.observation_space.shape[0]))
        for d in delta_feats:
            self.obs_mask.remove(d)

        # Define the observation space as a dictionary containing the main observation, the desired goal, 
        # the task, and the
        # the goal space observation.
        self.observation_space = gym.spaces.Dict(
            {
                'observation': gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(len(self.obs_mask),),
                    dtype=np.float32,
                ),
                'desired_goal': goal_space,  # Goal space as defined earlier
                'task': gym.spaces.Box(
                    low=0, high=1, shape=(len(self.task_map),), dtype=np.float32  # Task space is defined according to the task map
                ),
                'gs_observation': self.goal_featurizer.observation_space,  # Observation space within the goal space
            }
        )

        # Initialize hard reset flag and counter.
        self._do_hard_reset = True
        self._reset_counter = 0

        # Set the feature distribution.
        self.set_feature_dist(feature_dist)

        # Initialize current features as empty.
        self._features: List[int] = []
        self._features_s = ''
        self._feature_mask = np.zeros(len(self.task_map))

        # Initialize the model (if any) and the discount factor for rewards.
        self.model = None
        self.gamma = 1.0

       
    def set_goal_dims(self, dims):
        '''
        Set the goal dimensions.

        Args:
            dims: The goal dimensions.
        '''
        self.set_feature_dist(dims)

    def set_model(self, model: nn.Module, gamma: float):
        '''
        Set the model and discount factor for the environment.

        Args:
            model (nn.Module): The model to be used.
            gamma (float): The discount factor for rewards.
        '''
        self.model = model
        self.gamma = gamma

    def set_feature_dist(self, feature_dist: Dict[str, float]):
        '''
        Set the feature distribution.

        Args:
            feature_dist (Dict[str, float]): The feature distribution as a dictionary mapping from feature strings to probabilities.
        '''
        # Deduplicate features from combinations
        fdist: Dict[str, float] = {}
        self._feature_strings = {}
        for fs, p in feature_dist.items():
            ufeats = []
            for f in fs.replace('+', ',').split(','):
                if not f in ufeats:
                    ufeats.append(f)
            fdist[','.join(ufeats)] = p
            self._feature_strings[','.join(ufeats)] = fs.replace('+', ',')

        if not self.backproject_goal:
            # Check that maximum number of features doesn't change
            max_features = max((len(fs.split(',')) for fs in fdist.keys()))
            assert (
                self.observation_space['desired_goal'].shape[0] == max_features
            )
        for fs in fdist.keys():
            for fi in map(int, fs.split(',')):
                assert fi in self.task_map
        self._feature_dist_v = [k for k, v in fdist.items()]
        s = sum([v for k, v in fdist.items()])
        self._feature_dist_p = [v / s for k, v in fdist.items()]  # Normalizing probabilities

    def proj(self, obs: np.ndarray, feats: List[int]) -> np.ndarray:
        '''
        Project the observations onto the goal space.

        Args:
            obs (np.ndarray): The observations.
            feats (List[int]): The features.

        Returns:
            The projected observations.
        '''
        return np.matmul(obs, self.psi[feats].T) + self.offset[feats]

    def backproj(self, obs_w: np.ndarray, feats: List[int]) -> np.ndarray:
        '''
        Backproject the observations from the goal space to the observation space.

        Args:
            obs_w (np.ndarray): The observations in the goal space.
            feats (List[int]): The features.

        Returns:
            The backprojected observations.
        '''
        s_p = np.matmul(obs_w, self.psi_1[feats]) + self.offset_1
        return s_p[self.task_idx]

    def seed(self, seed=None):
        '''
        Seed the environment.

        Args:
            seed (int): The seed.

        Returns:
            The seed.
        '''
        self._do_hard_reset = True
        return super().seed(seed)

    def get_observation(self):
        '''
        Get the current observation from the environment.

        Returns:
            The current observation as a dictionary.
        '''
        # Get the main observation and the goal space observation.
        obs = super().get_observation()[self.obs_mask]
        gs_obs = self.goal_featurizer()

        # Compute the relative goal.
        if self.backproject_goal:
            s = gs_obs[self.task_idx]
            bpg = self.backproj(self.goal, self._features)
            g = bpg - s
            if len(self.goal_space['twist_feats
            ) > 0:
                # If the goal space has twist features, compute the relative goal in a special way.
                twf = [self.task_map[f] for f in self.goal_space['twist_feats']]
                g[twf] = (
                    np.remainder((bpg[twf] - s[twf]) + np.pi, 2 * np.pi) - np.pi
                )
            g *= self._feature_mask
        else:
            # If the goal is not backprojected, compute the relative goal in a different way.
            if len(self.goal_space['twist_feats']) > 0:
                raise NotImplementedError()
            gs = self.proj(gs_obs, self._features)
            g = np.zeros(self.observation_space['desired_goal'].shape)
            g[0 : len(self.goal)] = self.goal - gs

        # Normalize the goal space observation, if needed.
        if self.normalize_gs_observation:
            # If the goal space is defined for fewer features than
            # gs_observation, this will yield bogus values for undefined ones.
            gs_obs = self.proj(gs_obs, np.arange(0, len(gs_obs)))

        # Return the observation.
        return {
            'observation': obs,
            'desired_goal': g,
            'task': self._feature_mask,
            'gs_observation': gs_obs,
        }



    def hard_reset(self):
        """
        Performs a hard reset on the current simulation model. 
        This involves disabling contacts, resetting the state and setting the control to zero for a specified number of idle steps.
        """
        # The 'with' keyword starts a context management protocol. 
        # Here, it is used to temporarily disable 'contact' in the model while the following code block is executed.
        with self.p.model.disable('contact'):
            # Reset the physical simulation model and its state
            self.p.reset()
            self.reset_state()

        # The simulation is advanced by a number of steps without any control input.
        # This is done to bring the model to a stable state before starting the actual simulation.
        for _ in range(self.idle_steps):
            self.p.set_control(np.zeros_like(self.p.data.ctrl))  # Set control to a zero array of similar shape as current control.
            self.step_simulation()  # Advance the simulation one step forward

        # If there are no idle steps, just advance the simulation once
        if self.idle_steps <= 0:
            self.step_simulation()

    def sample_features(self) -> List[int]:
        """
        Randomly samples a feature from the feature distribution.
        The choice is made according to the probabilities specified in self._feature_dist_p.
        
        Returns:
            List[int]: A list of integers representing the selected features.
        """
        # np_random.choice randomly selects an element from the given array according to the specified probabilities.
        fs = self.np_random.choice(
            self._feature_dist_v, 1, p=self._feature_dist_p
        )[0]

        # map() applies the int function to each element of fs, which is then converted to a list and returned
        return list(map(int, fs.split(',')))

    def sample_goals_random(self, N: int = 1) -> np.ndarray:
        """
        Samples N goals randomly.
        
        Args:
            N (int, optional): The number of goals to sample. Defaults to 1.

        Returns:
            np.ndarray: A numpy array of the sampled goals.
        """
        # Project the current goal features into a different space using self.proj
        gstate = self.proj(self.goal_featurizer(), self._features)

        # Sample N goals randomly within the range -1 to 1.
        goal = self.np_random.uniform(
            low=-1.0, high=1.0, size=(N, len(self._features))
        )

        # For each feature, if it is in 'delta_feats', add the corresponding element of gstate to it.
        # If zero_twist_goals is True and the feature is in 'twist_feats', set that feature to 0.
        for i, f in enumerate(self._features):
            if f in self.goal_space['delta_feats']:
                goal[:, i] += gstate[i]
            if self.zero_twist_goals and f in self.goal_space['twist_feats']:
                goal[:, i] = 0
        return goal

    def sample_goal_using_r(self) -> np.ndarray:
        """
        Samples a goal using the reachability model.
        
        Returns:
            np.ndarray: A numpy array of the sampled goal.
        """
        N = 128  # Number of random goals to sample initially

        # Sample random goals
        cand = self.sample_goals_random(N=N)

        # Backproject the goal if the flag is set
        if self.backproject_goal:
            s = self.goal_featurizer()[self.task_idx]  # Current goal feature for the current task

            # Calculate goal by transforming the candidate goals and adding an offset
            gb = (np.matmul(cand, self.psi_1[self._features]) + self.offset_1)[
                :, self.task_idx
            ]

            # Calculate the delta between the goal and current state
            g = gb - s
            g *= self._feature_mask  # Apply the feature mask
        else:
            # Project the current goal features
            gs = self.proj(self.goal_featurizer(), self._features)

            # Initialize an array of zeros for storing the goals
            g = np.zeros((N, self.observation_space['desired_goal'].shape[0]))

            # Update the goal array with the difference between the candidate goals and the projected features
            g[:, 0 : len(self._features)] = cand - gs

        # Prepare the observation by getting current observation and expanding its dimensions
        obs = super().get_observation()[self.obs_mask]
        inp = {
            'observation': th.tensor(obs, dtype=th.float32)
            .unsqueeze(0)
            .expand(N, obs.shape[0]),

            # Prepare the desired goal by converting the numpy array to a PyTorch tensor
            'desired_goal': th.tensor(g, dtype=th.float32),

            # Prepare the task feature mask by expanding its dimensions
            'task': th.tensor(self._feature_mask, dtype=th.float32)
            .unsqueeze(0)
            .expand(N, self._feature_mask.shape[0]),
        }

        # Calculate the action to reach the desired goal using the policy model (self.model.pi)
        with th.no_grad():  # Disable gradient calculations during this block
            action = self.model.pi(inp).mean
        inp['action'] = action

        # Calculate the reachability of the goal using the reachability model (self.model.reachability)
        with th.no_grad():  # Disable gradient calculations during this block
            r = self.model.reachability(inp).clamp(0, 1)  # Clamp the reachability between 0 and 1

        # Determine the distribution for sampling the final goal
        if self.goal_sampling in {'r2', 'reachability2'}:
            # If the goal_sampling method is 'r2' or 'reachability2', favor samples reachable with 50% probability
            dist = th.tanh(2 * (1 - th.abs(r * 2 - 1) + 1e-1))
        else:
            # Otherwise, favor unreachable samples
            dist = 1 / (r.view(-1) + 0.1)

        # Sample a goal from the distribution and return it
        return cand[th.multinomial(dist, 1).item()]

    def sample_goal_using_q(self, obs: np.ndarray) -> np.ndarray:
        """
        Samples a goal using the q-values from the trained model.
        
        Args:
            obs (np.ndarray): The current observation.

        Returns:
            np.ndarray: A numpy array of the sampled goal.
        """
        N = 128  # Number of random goals to sample initially

        # Sample random goals
        cand = self.sample_goals_random(N=N)

        # Backproject the goal if the flag is set
        if self.backproject_goal:
            s = self.goal_featurizer()[self.task_idx]  # Current goal feature for the current task

            # Calculate goal by transforming the candidate goals and adding an offset
            gb = (np.matmul(cand, self.psi_1[self._features]) + self.offset_1)[
                :, self.task_idx
            ]

            # Calculate the delta between the goal and current state
            g = gb - s
            g *= self._feature_mask  # Apply the feature mask
        else:
            # Project the current goal features
            gs = self.proj(self.goal_featurizer(), self._features)

            # Initialize an array of zeros for storing the goals
            g = np.zeros((N, self.observation_space['desired_goal'].shape[0]))

            # Update the goal array with the difference between the candidate goals and the projected features
            g[:, 0 : len(self._features)] = cand - gs

        # Prepare the observation by getting current observation and expanding its dimensions
        obs = super().get_observation()[self.obs_mask]
        inp = {
            'observation': th.tensor(obs, dtype=th.float32)
            .unsqueeze(0)
            .expand(N, obs.shape[0]),

            # Prepare the desired goal by converting the numpy array to a PyTorch tensor
            'desired_goal': th.tensor(g, dtype=th.float32),

            # Prepare the task feature mask by expanding its dimensions
            'task': th.tensor(self._feature_mask, dtype=th.float32)
            .unsqueeze(0)
            .expand(N, self._feature_mask.shape[0]),
        }

        # Calculate the action to reach the desired goal using the policy model (self.model.pi)
        with th.no_grad():  # Disable gradient calculations during this block
            action = self.model.pi(inp).mean
        inp['action'] = action

        # Calculate the Q-values for the action using the Q-function model (self.model.q)
        with th.no_grad():  # Disable gradient calculations during this block
            q = th.min(self.model.q(inp), dim=-1).values

        # Calculate the control cost
        ctrl_cost = (
            self.max_steps
            * self.ctrl_cost
            * (0.25 * self.action_space.shape[0])
        )

        # Project the current observation
        wobs = self.proj(obs, self._features)

        # Calculate the Euclidean distance between the candidate goals and the projected observation
        dist = np.linalg.norm(cand - wobs, ord=2, axis=1)

        # Calculate the minimum return
        min_ret = (dist - ctrl_cost) * self.gamma ** self.max_steps

        # Calculate the slack by subtracting the minimum return from the Q-values
        slack = q - min_ret

        # Determine the distribution for sampling the final goal
        dist = 1 / (slack - slack.min() + 1)

        # Sample a goal from the distribution and return it
        return cand[th.multinomial(dist, 1).item()]

    def reset(self):
        """
        Resets the state of the environment to its initial state. It is typically 
        called at the end of an episode when the agent is done interacting with 
        the environment.

        Returns:
            np.ndarray: The observation after resetting the environment.
        """

        # Determine if a hard reset is needed based on the hard reset interval and the reset counter
        need_hard_reset = self._do_hard_reset or (
            self.hard_reset_interval > 0
            and self._reset_counter % self.hard_reset_interval == 0
        )

        # Perform hard reset if needed
        if need_hard_reset:
            self.hard_reset()  # Reset the environment to the initial state
            self._reset_counter = 0  # Reset the counter

        # Set the frame of reference if relative frame of reference is being used
        if self.relative_frame_of_reference:
            self.goal_featurizer.set_frame_of_reference()

        # Determine if features need to be resampled
        resample_features = False
        if need_hard_reset:
            resample_features = True
        if self.resample_features == 'soft':
            resample_features = True
        elif self.resample_features.startswith('soft'):
            freq = int(self.resample_features[4:])
            resample_features = self._reset_counter % freq == 0

        # Resample features if needed
        if resample_features:
            self._features = self.sample_features()  # Sample new features
            self._features_s = self._feature_strings[
                ','.join(map(str, self._features))
            ]
            self._feature_mask *= 0  # Reset the feature mask
            for f in self._features:
                self._feature_mask[self.task_map[f]] = 1.0  # Update the feature mask based on the sampled features

        # Sample a new goal
        self.goal = self.sample_goals_random()[0]
        if self.goal_sampling in {'q', 'q_value'}:  # Sample goal using Q-values if specified
            if self.model:
                self.goal = self.sample_goal_using_q()
        elif self.goal_sampling in {'r', 'reachability', 'r2', 'reachability2'}:  # Sample goal using reachability if specified
            if self.model:
                self.goal = self.sample_goal_using_r()
        elif self.goal_sampling not in {'random', 'uniform'}:
            raise ValueError(
                f'Unknown goal sampling method "{self.goal_sampling}"'
            )

        # Calculate the initial distance to the goal
        def distance_to_goal():
            gs = self.proj(self.goal_featurizer(), self._features)
            d = self.goal - gs
            for i, f in enumerate(self._features):
                if f in self.goal_space['twist_feats']:
                    # Wrap around projected pi/-pi for distance
                    d[i] = (
                        np.remainder(
                            (self.goal[i] - gs[i]) + self.proj_pi,
                            2 * self.proj_pi,
                        )
                        - self.proj_pi
                    )
            return np.linalg.norm(d, ord=2)

        self._d_initial = distance_to_goal()  # Store the initial distance to the goal

        # Update flags and counters
        self._do_hard_reset = False
        self._reset_counter += 1
        self._step = 0

        # Return the current observation after the reset
        return self.get_observation()

    def step(self, action):
        """
        Execute one timestep within the environment.

        Args:
            action (np.ndarray): An action provided by the agent.

        Returns:
            np.ndarray: Observation after taking the action.
            float: Reward received after taking the action.
            bool: A flag indicating whether the episode has ended.
            dict: Additional information about the step.
        """
        
        # Define function to calculate distance to the goal
        def distance_to_goal():
            gs = self.proj(self.goal_featurizer(), self._features)
            d = self.goal - gs
            for i, f in enumerate(self._features):
                if f in self.goal_space['twist_feats']:
                    # Wrap around projected pi/-pi for distance
                    d[i] = (
                        np.remainder(
                            (self.goal[i] - gs[i]) + self.proj_pi,
                            2 * self.proj_pi,
                        )
                        - self.proj_pi
                    )
            return np.linalg.norm(d, ord=2)

        # Get the previous distance to the goal
        d_prev = distance_to_goal()

        # Execute the action in the environment
        next_obs, reward, done, info = super().step(action)

        # Get the new distance to the goal
        d_new = distance_to_goal()

        # Populate the info dictionary with additional information
        info['potential'] = d_prev - d_new  # Potential is the change in distance to the goal
        info['distance'] = d_new  # Distance to the goal after the step
        info['reached_goal'] = info['distance'] < self.precision  # Whether the goal has been reached

        # Calculate the reward based on the reward type
        if self.reward == 'potential':
            reward = info['potential']
        elif self.reward == 'potential2':
            reward = d_prev - self.gamma * d_new
        elif self.reward == 'potential3':
            reward = 1.0 if info['reached_goal'] else 0.0
            reward += d_prev - self.gamma * d_new
        elif self.reward == 'potential4':
            reward = (d_prev - d_new) / self._d_initial
        elif self.reward == 'distance':
            reward = -info['distance']
        elif self.reward == 'sparse':
            reward = 1.0 if info['reached_goal'] else 0.0
        else:
            raise ValueError(f'Unknown reward: {self.reward}')
        reward -= self.ctrl_cost * np.square(action).sum()  # Subtract control cost from the reward

        # Update the episode status
        info['EpisodeContinues'] = True
        if info['reached_goal'] == True and not self.full_episodes:
            done = True
        info['time'] = self._step
        self._step += 1  # Increment the step counter
        if self._step >= self.max_steps:  # If maximum steps reached, end the episode
            done = True
        elif (
            not info['reached_goal'] and self.np_random.random() < self.reset_p
        ):
            info['RandomReset'] = True  # Randomly reset the episode
            done = True

        # Check if the agent fell over, if falling over is not allowed
        if not self.allow_fallover and self.fell_over():
            reward = self.fallover_penalty  # Assign penalty for falling over
            done = True
            self._do_hard_reset = True
            info['reached_goal'] = False
            info['fell_over'] = True

        # If the episode has ended and a hard reset is needed, remove the 'EpisodeContinues' flag
        if done and (
            self._do_hard_reset
            or (self._reset_counter % self.hard_reset_interval == 0)
        ):
            del info['EpisodeContinues']

        # If it's the final step of the task, set the 'LastStepOfTask' flag
        if done:
            info['LastStepOfTask'] = True

        # If the episode is over but it should continue due to implicit soft resets
        if done and 'EpisodeContinues' in info and self.implicit_soft_resets:
            need_hard_reset = self._do_hard_reset or (
                self.hard_reset_interval > 0
                and self._reset_counter % self.hard_reset_interval == 0
            )
            if not need_hard_reset:
                # Perform a soft reset and let the episode continue
                next_obs = self.reset()
                done = False
                del info['EpisodeContinues']
                info['SoftReset'] = True

        # Record the features used in this step
        info['features'] = self._features_s

        # Return observation, reward, done flag, and info dictionary
        return next_obs, reward, done, info


    @staticmethod
    def feature_controllable(robot: str, features: str, dim: int) -> bool:
        """
        Check if a given feature is controllable (i.e., its range is non-zero).

        Args:
            robot (str): The type of robot.
            features (str): The feature space.
            dim (int): The dimension of the feature.

        Returns:
            bool: Whether the feature is controllable or not.
        """

        if not features in g_goal_spaces:
            raise ValueError(f'Unsupported feature space: {features}')
        if not robot in g_goal_spaces[features]:
            raise ValueError(f'Unsupported robot: {robot}')

        gs = g_goal_spaces[features][robot]

        if dim < 0 or dim >= len(gs['min']):
            raise ValueError(f'Feature dimension {dim} out of range')

        return gs['min'][dim] != gs['max'][dim]

    @staticmethod
    def abstraction_matrix(
        robot: str, features: str, sdim: int
    ) -> Tuple[np.array, np.array]:
        """
        Generate the abstraction matrix.

        Args:
            robot (str): The type of robot.
            features (str): The feature space.
            sdim (int): The dimension of the state space.

        Returns:
            Tuple[np.array, np.array]: The abstraction matrix and offset vector.
        """

        if not features in g_goal_spaces:
            raise ValueError(f'Unsupported feature space: {features}')
        if not robot in g_goal_spaces[features]:
            raise ValueError(f'Unsupported robot: {robot}')

        gs = g_goal_spaces[features][robot]

        gmin = np.array(gs['min'])
        gmax = np.array(gs['max'])

        if gmin.size == 0:
            gmin = -np.ones(sdim)
            gmax = np.ones(sdim)

        if len(gmin) < sdim:
            gmin = np.concatenate([gmin, np.zeros(sdim - len(gmin))])
            gmax = np.concatenate([gmax, np.zeros(sdim - len(gmax))])

        psi = 2 * (np.eye(len(gmin)) * 1 / (gmax - gmin + 1e-7))
        offset = -2 * (gmin / (gmax - gmin + 1e-7)) - 1

        return psi, offset

    def delta_features(self, robot: str, features: str) -> List[int]:
        """
        Return the delta features for the specified robot and feature space.

        Args:
            robot (str): The type of robot.
            features (str): The feature space.

        Returns:
            List[int]: The delta features.
        """
        
        return g_goal_spaces[features][robot]['delta_feats']

    def feature_name(self, robot: str, features: str, f: int) -> str:
        """
        Get the name of a feature for the specified robot and feature space.

        Args:
            robot (str): The type of robot.
            features (str): The feature space.
            f (int): The feature index.

        Returns:
            str: The feature name.
        """
        
        return g_goal_spaces[features][robot]['str'][f]
