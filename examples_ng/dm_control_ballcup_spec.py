"""Finds a heuristic policy for the ball in cup task.

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
    env = suite.load(domain_name="ball_in_cup", task_name="catch") 
    obs_spec = env.observation_spec()
    action_spec = env.action_spec()

    min_reward = 1e5
    initial_xpos = np.array([-0.5, 0.0, 0.5])
    for i in range(num_runs):
        time_step = env.reset()
        initialize_env(env, initial_xpos[i%3])
        run_reward = 0.0
        obs = concatenate_obs(time_step, obs_spec)
        obs[3] -= 0.3
        for _ in range(1000):
            action = heuristic(obs)
            action = np.clip(action, action_spec.minimum[0], action_spec.maximum[0])
            time_step = env.step(action)
            obs = concatenate_obs(time_step, obs_spec)
            obs[3] -= 0.3
            step_reward = time_step.reward # +1 if ball in cup, 0 otherwise
            step_reward += custom_reward(obs)
            step_reward /= 2.0
            run_reward += step_reward

        if run_reward < min_reward:
            min_reward = run_reward
    return -min_reward

def concatenate_obs(time_step, obs_spec):
    return np.concatenate([time_step.observation[k].ravel() for k in obs_spec])

def initialize_env(env, x_pos):
    env.physics.named.data.qpos['ball_x'][0] = x_pos
    env.physics.named.data.qpos['ball_z'][0] = 0.0

def custom_reward(obs: np.ndarray) -> float:
    x_cup = obs[0]
    z_cup = obs[1]
    x_ball = obs[2]
    z_ball = obs[3]
    angle = np.arctan2(x_ball - x_cup, z_ball - z_cup)
    vx_ball = obs[6]
    vz_ball = obs[7]
    v_ball = np.sqrt(vx_ball**2 + vz_ball**2)
    reward = 1 - np.abs(angle)/np.pi
    if v_ball > 4.0:
        reward -= 0.1*v_ball
    return reward

@funsearch.evolve
def heuristic(obs: np.ndarray) -> np.ndarray:
    """Returns actions between -1 and 1.
    obs size is 8. return shape is (2,).
    """
    action = np.zeros((2,))

    return action