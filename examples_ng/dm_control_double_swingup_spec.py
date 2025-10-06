"""Find a heuristic policy for the double pendulum swingup task.

On every iteration, improve heuristic_v1 over the heuristic_vX methods from previous iterations.
Make only small changes. Try to make the code short. Only add the missing function body, do not return additional code.
"""

import numpy as np
import funsearch
  
@funsearch.run
def solve(num_runs) -> float:
    """Returns the reward for a heuristic.
      Total number of function evaluations is num_runs * budget (100).
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
    env = suite.load(domain_name="acrobot", task_name="swingup")
    obs_spec = env.observation_spec()
    action_spec = env.action_spec()
    sum_reward = 0.0
    for _ in range(num_runs):
        time_step = env.reset()
        initialize_to_zero(env)
        run_reward = 0.0
        obs = concatenate_obs(time_step, obs_spec)
        for _ in range(1000):
            # cos_theta = time_step.observation['orientation'][0]
            # sin_theta = -time_step.observation['orientation'][1]
            # theta = np.arctan2(sin_theta, cos_theta)
            action = heuristic(obs)
            action = np.clip(action, -1, 1)
            time_step = env.step(action)
            run_reward += time_step.reward # the max per step reward is 1.0
            obs = concatenate_obs(time_step, obs_spec)
        sum_reward += run_reward
    return -sum_reward / num_runs  # returns the negative average reward

def concatenate_obs(time_step, obs_spec):
    return np.concatenate([time_step.observation[k].ravel() for k in obs_spec])

def initialize_to_zero(env):
    env.physics.named.data.qpos['shoulder'][0] = np.pi
    env.physics.named.data.qpos['elbow'][0] = np.pi
    env.physics.named.data.qvel['shoulder'][0] = 0.0
    env.physics.named.data.qvel['elbow'][0] = 0.0
    env.physics.named.data.qacc['shoulder'][0] = 0.0
    env.physics.named.data.qacc['elbow'][0] = 0.0
    env.physics.named.data.qacc_smooth['shoulder'][0] = 0.0
    env.physics.named.data.qacc_smooth['elbow'][0] = 0.0
    env.physics.named.data.qacc_warmstart['shoulder'][0] = 0.0
    env.physics.named.data.qacc_warmstart['elbow'][0] = 0.0
    env.physics.named.data.actuator_moment['elbow'] = np.zeros((2,))
    env.physics.named.data.qfrc_bias['shoulder'][0] = 0.0
    env.physics.named.data.qfrc_bias['elbow'][0] = 0.0

@funsearch.evolve
def heuristic(obs: np.ndarray) -> float:
    """Returns an action between -1 and 1.
    obs size is 6.
    """
    x1 = np.arctan2(-obs[1], obs[0])
    x2 = np.arctan2(-obs[2], obs[3])
    # x3 = obs[4]
    # x4 = obs[5]
    action = 0.0
    return action