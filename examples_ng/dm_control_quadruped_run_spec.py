"""
Finds a heuristic policy for the quadruped task. 
On every iteration, improve heuristic_v1 over the heuristic_vX methods from previous iterations.
The reward is proportinal to the distance run.
Make only small changes. Try to make the code short. 
The heuristic receives a 78-dimensional obs array and outputs a 12-dimensional action.
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
  
# @funsearch.run
# def solve_non_param(num_runs) -> float:
#     # Just evaluate the heuristic directly
#     dummy_params = np.array([])
#     cost = objective_function(dummy_params, num_runs=num_runs)
#     return -cost, dummy_params

def objective_function(params: np.ndarray,
                      num_runs: int,
                      ) -> float:
    import warnings
    from glfw import GLFWError
    warnings.filterwarnings("ignore", category=GLFWError)


    from dm_control import suite
    env = suite.load(domain_name="quadruped", task_name="run") 
    obs_spec = env.observation_spec()
    action_spec = env.action_spec()

    min_reward = 1e5

    rewards = []
    for i in range(num_runs):
        time_step = env.reset()
        initialize_to_zero(env)
        run_reward = 0.0
        obs = concatenate_obs(time_step, obs_spec)
        for _ in range(1000):
            action = heuristic(obs)
            action = np.clip(action, action_spec.minimum, action_spec.maximum)
            time_step = env.step(action)
            obs = concatenate_obs(time_step, obs_spec)

            # step_reward = time_step.reward # +1 within a range of the object, 0 otherwise
            step_reward = max(0.0, min(1.0, env.physics.torso_velocity()[0]/10.0))
            # if not step_reward:
            #     step_reward = 0.
            run_reward += step_reward


        rewards.append(run_reward)
    return -np.mean(rewards)

def concatenate_obs(time_step, obs_spec):
    return np.concatenate([time_step.observation[k].ravel() for k in obs_spec])

def initialize_to_zero(env):
  env.physics.named.data.qpos['root'][0] = 0.0
  env.physics.named.data.qpos['root'][1] = 0.0
  env.physics.named.data.qpos['root'][2] = 0.1
  env.physics.named.data.qpos['root'][3] = 1.0
  env.physics.named.data.qpos['root'][4] = 0.0
  env.physics.named.data.qpos['root'][5] = 0.0
  env.physics.named.data.qpos['root'][6] = 0.0


@funsearch.evolve
def heuristic(obs: np.ndarray) -> np.ndarray:
    """Returns a 12 dimentional actions array.
    - obs[0:44]: egocentric_state (44,)
    - obs[44:47]: torso_velocity (3,)
    - obs[47]: torso_upright ()
    - obs[48:54]: imu (6,)
    - obs[54:78]: force_torque (24,)
    - actions[0:12]: actions (12,)
    """
    actions = np.random.random(size=12)
    return actions