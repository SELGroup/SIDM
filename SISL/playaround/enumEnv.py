import gym
import hucc

def test_envs():
    envs = [
        "BiskHurdles-v1",
        "BiskLimbo-v1",
        "BiskHurdlesLimbo-v1",
        "BiskStairs-v1",
        "BiskGaps-v1",
    ]

    cfg = {
        "env": {
            "name": "Dummy",
            "train_procs": 1,
            "train_args": {},
        },
        "device": "cpu",
        "seed": 0,
    }

    for env_name in envs:
        cfg["env"]["name"] = env_name
        try:
            vec_envs = hucc.make_vec_envs(
                cfg["env"]["name"],
                cfg["env"]["train_procs"],
                device=cfg["device"],
                seed=cfg["seed"],
                **cfg["env"]["train_args"],
            )
            print(f"Successfully created {env_name}")
        except gym.error.UnregisteredEnv:
            print(f"Environment {env_name} is not registered.")
        except Exception as e:
            print(f"Error occurred while creating {env_name}: {str(e)}")


if __name__ == "__main__":
    test_envs()
