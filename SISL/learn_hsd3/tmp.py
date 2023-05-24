class CompatSyncVectorEnv(SyncVectorEnv):
    """
    A subclass of SyncVectorEnv that allows for environments to reset when they are done,
    without support for copying observations. The class maintains a list of environments,
    and steps them all synchronously.

    Args:
        env_fns (list): A list of functions that create environments.
        observation_space (Space, optional): The observation space expected by the environments.
            Defaults to None.
        action_space (Space, optional): The action space expected by the environments.
            Defaults to None.

    Attributes:
        _observations (list): A private attribute that holds the latest observations from each environment.
    """

    def __init__(self, env_fns, observation_space=None, action_space=None):
        """
        Initialize the environments, calling the superclass's constructor with the copy parameter set to False.
        Initialize the _observations attribute as an empty list.
        """
        super().__init__(env_fns, observation_space, action_space, copy=False)
        self._observations = []

    def reset_if_done(self):
        """
        Reset environments that are done, and populate the _observations list with the latest
        observation from each environment.

        Returns:
            list: The latest observations from each environment.
        """
        observations = []  # Initialize an empty list to hold the observations

        # Loop over each environment
        for i, env in enumerate(self.envs):
            # If the environment is done, reset it and get the initial observation
            if self._dones[i]:
                observations.append(env.reset())
            else:
                # If the environment is not done, get the last observation
                observations.append(self._observations[i])

        # Concatenate all observations into a single array
        concatenate(
            observations, self.observations, self.single_observation_space
        )

        # Return the latest observations
        return self.observations

    def reset_wait(self):
        """
        Reset all environments and populate the _observations list with the initial observation
        from each environment.

        Returns:
            list: The initial observations from each environment.
        """
        self._dones[:] = False  # Mark all environments as not done
        observations = []  # Initialize an empty list to hold the observations

        # Loop over each environment
        for env in self.envs:
            # Reset the environment and get the initial observation
            observation = env.reset()
            observations.append(observation)

        self._observations = observations  # Update the _observations list

        # Concatenate all observations into a single array
        concatenate(
            observations, self.observations, self.single_observation_space
        )

        # Return the latest observations
        return self.observations

    def step_wait(self):
        """
        Step each environment with the corresponding action from _actions, and update the _observations,
        _rewards, and _dones lists with the results of the step.

        Returns:
            tuple: The latest observations, rewards, done flags, and info dictionaries from each environment.
        """
        observations, infos = [], []  # Initialize empty lists to hold the observations and info dictionaries

        # Loop over each environment and action
        for i, (env, action) in enumerate(zip(self.envs, self._actions)):
            # Step the environment and get the results
            observation, self._rewards[i], self._dones[i], info = env.step(
                action
            )
            observations.append(observation)  # Append the observation to the list
            infos.append(info)  # Append the info dictionary to the list

        self._observations = observations  # Update
        # Update the _observations list

        # Concatenate all observations into a single array
        concatenate(
            observations, self.observations, self.single_observation_space
        )

        # Return the latest observations, rewards, done flags, and info dictionaries
        return self.observations, self._rewards, self._dones, infos


class VecPyTorch(gym.Wrapper):
    '''
    This class is a PyTorch-compatible wrapper for a vectorized Gym environment.

    It ensures the inputs and outputs are PyTorch tensors, and exposes 
    action and observation spaces for a single environment within 
    a vector of environments.
    '''

    def __init__(self, venv: gym.Env, device: str):
        '''
        Constructor requires a vectorized environment and a device string.
        The device is usually 'cpu' or 'cuda'.
        '''

        # Check if venv is an instance of the supported classes
        if not isinstance(venv, CompatSyncVectorEnv) and not isinstance(
            venv, AsyncVectorEnv
        ):
            # If it isn't, throw a ValueError
            raise ValueError(
                f'This wrapper works with CompatSyncVectorEnv and AsynVectorEnv only; got {type(venv)}'
            )

        # Call superclass's constructor
        super().__init__(venv)

        # Store device for future reference
        self._device = th.device(device)

        # Initialize a free-form context dictionary
        self._ctx: Dict[str, Any] = {}

        # Set action and observation spaces to single environment's
        self.action_space = venv.single_action_space
        self.observation_space = venv.single_observation_space

    @property
    def device(self):
        '''
        Property for accessing the device
        '''
        return self._device

    @property
    def ctx(self):
        '''
        Property for accessing the context dictionary
        '''
        return self._ctx

    @property
    def num_envs(self) -> int:
        '''
        Property for accessing the number of environments
        '''
        return self.env.num_envs

    def _from_np(
        self, x: np.ndarray, dtype: th.dtype = th.float32
    ) -> th.Tensor:
        '''
        Helper method to convert a numpy array to a PyTorch tensor.
        Default dtype is float32.
        '''
        return th.from_numpy(x).to(dtype=dtype, device=self.device, copy=True)

    def seed(self, seeds):
        '''
        Seed the environments. If a single integer is provided, 
        it's interpreted as a seed for every environment.
        '''
        if seeds is None:
            seeds = [None] * self.num_envs
        if isinstance(seeds, int):
            seeds = [seeds] * self.num_envs
        assert len(seeds) == self.num_envs
        return self.env.seed(seeds)

    def reset(self):
        '''
        Reset the environments and return observations.
        '''
        obs = self.env.reset()
        if isinstance(obs, dict):
            obs = {k: self._from_np(v) for k, v in obs.items()}
        else:
            obs = self._from_np(obs)
        return obs

    def reset_if_done(self):
        '''
        Reset environments if they're done.
        '''
        if isinstance(self.env, CompatSyncVectorEnv):
            obs = self.env.reset_if_done()
        else:
            self.env._assert_is_running()
            if self.env._state != AsyncState.DEFAULT:
                raise AlreadyPendingCallError(
                    'Calling `reset_if_done` while waiting '
                    'for a pending call to `{0}` to complete'.format(
                        self.env._state.value
                    ),
                    self.env._state.value,
                )

            for pipe in self.env.parent_pipes:
                pipe.send(('reset_if_done', None))
            self.env._state = AsyncState.WAITING_RESET
            obs = self.env.reset_wait()

        # Convert observations to PyTorch tensors
        if isinstance(obs, dict):
            obs = {k: self._from_np(v) for k, v in obs.items()}
        else:
            obs = self._from_np(obs)
        return obs

    def step(self, actions):
        '''
        Take a step in the environments using the provided actions.
        '''

        if isinstance(actions, th.LongTensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        elif isinstance(actions, th.Tensor):
            actions = actions.cpu().numpy()

        # Step the environments
        obs, reward, done, info = self.env.step(actions)
        
        # Convert observations, rewards, and dones to PyTorch tensors
        if isinstance(obs, dict):
            obs = {k: self._from_np(v) for k, v in obs.items()}
        else:
            obs = self._from_np(obs)
        reward = self._from_np(reward).unsqueeze(dim=1)
        done = self._from_np(done, dtype=th.bool).unsqueeze(dim=1)
        return obs, reward, done, info

    def render_single(self, index: int = 0, **kwargs):
        '''
        Render a single environment.
        '''
        if isinstance(self.env, CompatSyncVectorEnv):
            return th.from_numpy(self.env.envs[index].render(**kwargs).copy())
        else:
            self.env.parent_pipes[index].send(('render', kwargs))
            out, success = self.env.parent_pipes[index].recv()
            return th.from_numpy(out)

    def render_all(self, **kwargs):
        '''
        Render all the environments.
        '''
        if isinstance(self.env, CompatSyncVectorEnv):
            return [
                th.from_numpy(e.render(**kwargs).copy()) for e in self.env.envs
            ]
        else:
            for pipe in self.env.parent_pipes:
                pipe.send(('render', kwargs))
            outs = []
            for pipe in self.env.parent_pipes:
                out, success = pipe.recv()
                outs.append(th.from_numpy(out))
            return outs

    def call(self, fn: str, *args, **kwargs):
        '''
        Call a method on all environments.
        '''
        if isinstance(self.env, CompatSyncVectorEnv):
            return [getattr(e, fn)(*args, **kwargs) for e in self.env.envs]
        else:
            for pipe in self.env.parent_pipes:
                pipe.send(('call', {'fn': fn, 'args': args, 'kwargs': kwargs}))
            outs = []
            for pipe in self.env.parent_pipes:
                out, success = pipe.recv()
                outs.append(out)
            return outs

          def async_worker_shared_memory(
    index, env_fn, pipe, parent_pipe, shared_memory, error_queue
):
    '''
    Function for a worker in an asynchronous environment setup. 
    It communicates with the main process via pipes and shares 
    observations via shared memory.

    Args:
        index (int): The index of the worker.
        env_fn (callable): Function that returns a Gym environment.
        pipe (multiprocessing.Pipe): Pipe for communication to the main process.
        parent_pipe (multiprocessing.Pipe): Pipe for communication from the main process.
        shared_memory (multiprocessing.shared_memory.SharedMemory): Shared memory object for storing observations.
        error_queue (multiprocessing.Queue): Queue for passing exceptions to the main process.
    '''
    # Check that shared memory is provided
    assert shared_memory is not None

    # Create the environment
    env = env_fn()
    observation_space = env.observation_space

    # Close the parent pipe as it's not used in the worker
    parent_pipe.close()

    # Initialize the done flag
    is_done = True

    try:
        # Start listening for commands from the main process
        while True:
            command, data = pipe.recv()

            # Reset the environment
            if command == 'reset' or command == 'reset_if_done':
                if is_done or command == 'reset':
                    observation = env.reset()
                    write_to_shared_memory(
                        index, observation, shared_memory, observation_space
                    )
                is_done = False
                pipe.send((None, True))

            # Step in the environment
            elif command == 'step':
                observation, reward, done, info = env.step(data)
                is_done = done
                write_to_shared_memory(
                    index, observation, shared_memory, observation_space
                )
                pipe.send(((None, reward, done, info), True))

            # Seed the environment
            elif command == 'seed':
                env.seed(data)
                pipe.send((None, True))

            # Close the environment
            elif command == 'close':
                pipe.send((None, True))
                break

            # Render the environment
            elif command == 'render':
                rendered = env.render(**data)
                pipe.send((rendered, True))

            # Check the observation space
            elif command == '_check_observation_space':
                pipe.send((data == observation_space, True))

            # Call a method on the environment
            elif command == 'call':
                ret = getattr(env, data['fn'])(*data['args'], **data['kwargs'])
                pipe.send((ret, True))

            # Raise an error for unknown commands
            else:
                raise RuntimeError(
                    'Received unknown command `{0}`. Must '
                    'be one of `reset`, `step`, `seed`, `close`, '
                    '`render`, `_check_observation_space`, `reset_if_done`, `call`.'.format(
                        command
                    )
                )
    # Handle exceptions and pass them to the main process
    except (KeyboardInterrupt, Exception):
        traceback.print_exc()
        error_queue.put((index,) + sys.exc_info()[:2])
        pipe.send((None, False))
    finally:
        env.close()

class RewardAccWrapper(gym.Wrapper):
    '''
    A Gym environment wrapper that accumulates rewards on a per-episode basis.
    '''

    def __init__(self, env):
        '''
        Initialize the wrapper.

        Args:
            env (gym.Env): The Gym environment to wrap.
        '''
        super().__init__(env)
        self._acc = 0.0  # Initialize the reward accumulator

    def reset(self, **kwargs):
        '''
        Reset the environment and the reward accumulator.

        Returns:
            The initial observation of the environment.
        '''
        self._acc = 0.0  # Reset the reward accumulator
        return self.env.reset(**kwargs)  # Reset the environment

    def step(self, action):
        '''
        Take a step in the environment using the given action.

        Args:
            action: The action to take in the environment.

        Returns:
            A tuple containing the new observation, the reward, the done flag, and an info dictionary.
        '''
        # Take a step in the environment
        observation, reward, done, info = self.env.step(action)

        self._acc += reward  # Accumulate the reward
        info['reward_acc'] = self._acc  # Store the accumulated reward in the info dictionary

        return observation, reward, done, info


class FrameCounter(gym.ObservationWrapper):
    '''
    A Gym observation wrapper that adds a frame counter to the observation.
    The resulting observation space will be a dictionary with an additional ['time'] entry.
    '''

    def __init__(self, env):
        '''
        Initialize the wrapper.

        Args:
            env (gym.Env): The Gym environment to wrap.
        '''
        super().__init__(env)

        # Initialize the frame counter
        self._time = np.array([0], dtype=np.int32)

        maxint = np.iinfo(np.int32).max  # Get the maximum integer value for the frame counter

        # Define the observation space based on the original environment's observation space
        if isinstance(env.observation_space, gym.spaces.Dict):
            self._wrap_in_dict = False
            self.observation_space = gym.spaces.Dict(
                dict(
                    time=gym.spaces.Box(
                        low=0, high=maxint, shape=(1,), dtype=np.int32
                    ),
                    **env.observation_space.spaces,
                )
            )
        else:
            self._wrap_in_dict = True
            self.observation_space = gym.spaces.Dict(
                dict(
                    time=gym.spaces.Box(
                        low=0, high=maxint, shape=(1,), dtype=np.int32
                    ),
                    observation=env.observation_space,
                )
            )

    def reset(self, **kwargs):
        '''
        Reset the environment and the frame counter.

        Returns:
            The initial observation of the environment.
        '''
        observation = self.env.reset(**kwargs)  # Reset the environment
        self._time[0] = 0  # Reset the frame counter
        return self.observation(observation)

    def step(self, action):
        '''
        Take a step in the environment using the given action.

        Args:
            action: The action to take in the environment.

        Returns:
            A tuple containing the new observation, the reward, the done flag, and an info dictionary.
        '''
        self._time[0] += 1  # Increment the frame counter
        return super().step(action)

    def observation(self, observation):
        '''
        Add the frame counter to the observation.

        Args:
            observation: The original observation from the environment.

        Returns:
            The modified observation which includes the frame counter
        '''
        if self._wrap_in_dict:  # If the original observation was not a dictionary
            observation = {'observation': observation}  # Wrap it in a dictionary
        observation['time'] = self._time  # Add the frame counter to the observation
        return observation


class DictObs(gym.ObservationWrapper):
    '''
    A Gym observation wrapper that ensures the observation is always a dictionary.
    '''

    def __init__(self, env):
        '''
        Initialize the wrapper.

        Args:
            env (gym.Env): The Gym environment to wrap.
        '''
        super().__init__(env)

        # Define the observation space as a dictionary containing the original observation space
        self.observation_space = gym.spaces.Dict(
            observation=env.observation_space
        )

    def observation(self, observation):
        '''
        Ensure the observation is a dictionary.

        Args:
            observation: The original observation from the environment.

        Returns:
            The modified observation which is always a dictionary.
        '''
        return {'observation': observation}

class BiskFeatures(gym.ObservationWrapper):
    '''
    A Gym observation wrapper for extracting features from a BiskSingleRobotEnv environment.
    '''

    def __init__(self, env, features: str):
        '''
        Initialize the wrapper.

        Args:
            env (gym.Env): The Gym environment to wrap.
            features (str): The features to extract from the environment.
        '''
        from bisk import BiskSingleRobotEnv

        super().__init__(env)

        # Assert the environment is of the correct type
        assert isinstance(
            env.unwrapped, BiskSingleRobotEnv
        ), 'BiskFeatures requires a BiskSingleRobotEnv environment'

        # Assert the observation space is a dictionary
        assert isinstance(env.observation_space, gym.spaces.Dict)

        # Parse the feature string
        if ':' in features:
            dest, features = features.split(':')
        else:
            dest, features = features, features

        # Create a featurizer using the environment's make_featurizer method
        self.featurizer = env.unwrapped.make_featurizer(features)

        # Save the destination key for the feature data
        self.dest = dest

        # Update the observation space to include the new features
        d = {self.dest: self.featurizer.observation_space}
        for k, v in env.observation_space.spaces.items():
            d[k] = v
        self.observation_space = gym.spaces.Dict(d)

    def observation(self, observation):
        '''
        Add the extracted features to the observation.

        Args:
            observation: The original observation from the environment.

        Returns:
            The modified observation which includes the extracted features.
        '''
        observation[self.dest] = self.featurizer()
        return observation


def make_vec_envs(
    env_name: str,
    n: int,
    device: str = 'cpu',
    seed: Optional[int] = None,
    fork: Optional[bool] = None,
    wrappers: Optional[List[Callable[[gym.Env], gym.Env]]] = None,
    **env_args,
) -> VecPyTorch:
    '''
    Create multiple environments and wrap them in a VecPyTorch for batched operations.

    Args:
        env_name (str): The name of the Gym environment to create.
        n (int): The number of environments to create.
        device (str, optional): The device where the environments will be run.
        seed (int, optional): The seed for the random number generator.
        fork (bool, optional): Whether to use multiprocessing.
        wrappers (list of callable, optional): A list of wrappers to apply to each environment.
        **env_args: Additional arguments to pass when creating the environments.

    Returns:
        A VecPyTorch wrapping the created environments.
    '''
    def make_env(seed, fork, i):
        def thunk():
            env = gym.make(env_name, **env_args)
            if fork and seed is not None:
                random.seed(seed + i)
            if seed is not None and hasattr(env, 'seed'):
                env.seed(seed + i)
            if wrappers:
                for w in wrappers:
                    env = w(env)
            env = RewardAccWrapper(env)
            return env

        return thunk

    fork = n > 1 if fork is None else fork
    if platform.system() == 'Darwin':
        log.info('Disabling forking on macOS due to poor support')
        fork = False
    # The rest of `make_vec_envs` function creates and returns 
    # a vectorized environment using either `AsyncVectorEnv` or `CompatSyncVectorEnv` 
    # based on the `fork` flag. `VecPyTorch` is then applied to these environments 
    # to allow them to be used with PyTorch, and the seed for the action space 
    # is optionally set. Finally, the vectorized environments are returned.
    envs = [make_env(seed, fork, i) for i in range(n)]
    if fork:
        envs = AsyncVectorEnv(
            envs,
            shared_memory=True,
        worker=async_worker_shared_memory,
        copy=False,
    )
else:
    envs = CompatSyncVectorEnv(envs)
envs = VecPyTorch(envs, device)
if seed is not None:
    envs.action_space.seed(seed)

return envs

def make_wrappers(cfg: DictConfig) -> List:
    """
    Creates a list of environment wrappers based on the given configuration.
    
    Args:
    cfg (DictConfig): A configuration object that specifies the wrappers to be used.

    Returns:
    List: A list of environment wrappers.
    """

    # Define the available wrappers that take no arguments
    # DictObs: Wraps the observations of the environment in a dictionary
    # FlattenObservation: Flattens the observations of the environment
    # FrameCounter: Adds a frame counter to the observations
    wrappers = []
    wrapper_map = {
        'dict_obs': lambda env: DictObs(env),
        'flatten_obs': lambda env: FlattenObservation(env),
        'frame_counter': lambda env: FrameCounter(env),
        'time': lambda env: FrameCounter(env),
    }

    # Define the available wrappers that take one argument
    # TimeLimit: Limits the number of steps in an episode
    # BiskFeatures: Adds additional features to the Bisk environment
    wrapper_map_arg1 = {
        'time_limit': lambda arg: lambda env: TimeLimit(
            env, max_episode_steps=int(arg)
        ),
        'bisk_features': lambda arg: lambda env: BiskFeatures(env, arg),
    }

    # Create the list of wrappers based on the configuration
    for w in cfg.wrappers:
        if isinstance(w, DictConfig): # If the wrapper has arguments
            if len(w) > 1:
                raise ValueError(f'Malformed wrapper item: {w}')
            for k, arg in w.items():
                if not k in wrapper_map_arg1:
                    raise ValueError(f'No such wrapper: {k}')
                wrappers.append(wrapper_map_arg1[k](arg))
                break
        else: # If the wrapper does not have arguments
            if not w in wrapper_map:
                raise ValueError(f'No such wrapper: {w}')
            wrappers.append(wrapper_map[w])

    return wrappers



