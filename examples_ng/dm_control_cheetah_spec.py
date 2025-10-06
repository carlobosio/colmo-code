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
    import warnings
    from glfw import GLFWError
    warnings.filterwarnings("ignore", category=GLFWError)

    from dm_control import suite
    env = suite.load(domain_name="cheetah", task_name="run") 
    obs_spec = env.observation_spec()
    action_spec = env.action_spec()
    
    min_reward = 1e5
    rewards = []
    for i in range(num_runs):
        time_step = env.reset()
        # initialize_env(env, x_pos=initial_xpos[0], y_pos=initial_xpos[1]) # random x and y position
        run_reward = 0.0
        # Observation: position (2), to_target (2), velocity (2)
        obs = concatenate_obs(time_step, obs_spec)
        for _ in range(1000):
            action = heuristic(obs)
            action = np.clip(action, action_spec.minimum[0], action_spec.maximum[0])
            time_step = env.step(action)
            obs = concatenate_obs(time_step, obs_spec)

            step_reward = time_step.reward # +1 within a range of the object, 0 otherwise
            if not step_reward:
                step_reward = 0.
            run_reward += step_reward


        rewards.append(run_reward)
    return -np.mean(rewards)

def concatenate_obs(time_step, obs_spec):
    return np.concatenate([time_step.observation[k].ravel() for k in obs_spec])


@funsearch.evolve
def heuristic(obs: np.ndarray) -> np.ndarray:
    """Returns actions between -1 and 1.
    obs size is 17: position (8), velocity (9). return shape is (6,).
    """
    action = np.random.random(size=6)
    return action