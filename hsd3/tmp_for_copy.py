def train_loop_mfdim_actor(setup: TrainingSetup):
    # ... same initialization code as before ...
    
    transitions = []  # Store transitions here instead of putting them into queues

    while setup.n_samples < max_steps:
        # ... same code as before, up to the point where you interact with the environment ...

        t_obs = (
            th_flatten(envs.observation_space, obs)
            if cfg.agent.name != 'sacmt'
            else obs
        )
        action, extra = agent.action(envs, t_obs)
        assert (
            extra is None
        ), "Distributed training doesn't work with extra info from action"
        next_obs, reward, done, info = envs.step(action)
        t_next_obs = (
            th_flatten(envs.observation_space, next_obs)
            if cfg.agent.name != 'sacmt'
            else next_obs
        )
        
        # Instead of putting transitions into queues, add them to the transitions list
        transitions.append((t_obs, action, extra, (t_next_obs, reward, done, info)))

        # ... same code as before ...

    return transitions  # Return the list of transitions at the end

    # ... rest of the code ...
