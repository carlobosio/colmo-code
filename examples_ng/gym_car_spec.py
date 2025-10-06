"""Finds a heuristic policy for the cheetah task. 
A running planar biped based on (Wawrzyński,2009). The reward r is linearly proportional to the forward velocity v up to a maximum of 10m/s.
On every iteration, improve heuristic_v1 over the heuristic_vX methods from previous iterations.
Make only small changes. Try to make the code short. 
"""

import numpy as np
import funsearch

@funsearch.run
def solve(num_runs) -> float:
    """Returns the reward for a heuristic.
    """
    import nevergrad as ng
    from loky import ProcessPoolExecutor
    from functools import partial

    num_params = 1

    if num_params == 0:
        # Just evaluate the heuristic directly
        dummy_params = np.array([])
        cost = objective_function(dummy_params, num_runs=num_runs)
        return -cost, dummy_params

    objective_partial = partial(objective_function, 
                                num_runs=num_runs, # number of runs to average the reward over
                                )

    ng_optimizer = ng.optimizers.OnePlusOne(
              parametrization=ng.p.Array(shape=(num_params,)),
              budget=100, # number of function evaluations
              num_workers=12
              )
    with ProcessPoolExecutor(max_workers=ng_optimizer.num_workers) as executor:
        solution = ng_optimizer.minimize(objective_partial, executor=executor, batch_mode=False)
    
    optimized_params = solution.value
    score = -solution.loss

    return score, optimized_params
  

def objective_function(params: np.ndarray,
                      num_runs: int,
                      ) -> float:


    import gymnasium as gym

    # Create environment
    env = gym.make("MountainCarContinuous-v0", goal_velocity=0.1)

    done = False
    total_reward = 0.0

    # from dm_control import suite
    # env = suite.load(domain_name="cheetah", task_name="run") 
    # obs_spec = env.observation_spec()
    # action_spec = env.action_spec()
    
    min_reward = 1e5
    rewards = []
    for i in range(num_runs):
        obs, info = env.reset(seed=i, ) #options={"low": -0.7, "high": -0.5}

        run_reward = 0.0
    
        for _ in range(1000):

            action = heuristic(obs)
            action = np.clip(action, env.action_space.low, env.action_space.high)

            obs, step_reward, terminated, truncated, info = env.step(action)
            # print(step_reward)
            
            if not step_reward:
                step_reward = 0.
            run_reward += step_reward


        rewards.append(run_reward)
    return -np.mean(rewards)


@funsearch.evolve
def heuristic(obs: np.ndarray) -> np.ndarray:
    """Returns an array with a single action.
    obs size is 2.
    """
    assert obs.shape == (2,)
    action = np.random.random(size=1)
    return action


# if __name__ == "__main__":
#     import pdb; pdb.set_trace()
#     mean = objective_function(None, 1)
#     print(mean)

