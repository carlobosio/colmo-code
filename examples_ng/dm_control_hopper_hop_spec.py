"""Finds a heuristic policy for the hopper task.  
The planar one-legged hopper is initialised in a random configuration. In the hop task it is rewarded for torso height and forward velocity.
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
  
"""


import warnings
from glfw import GLFWError
warnings.filterwarnings("ignore", category=GLFWError)


from dm_control import suite
env = suite.load(domain_name="hopper", task_name="hop") 
obs_spec = env.observation_spec()
action_spec = env.action_spec()


def heuristic(obs: np.ndarray) -> np.ndarray:
    action = np.random.random(size=2)
    # return action
    return action
# from dm_control.utils import rewards

for _ in range(1000):
    action = heuristic(obs)
    # action = np.clip(action, action_spec.minimum[0], action_spec.maximum[0])
    for _ in range(10):
        time_step = env.step(action)
    obs = concatenate_obs(time_step, obs_spec)
    step_reward = time_step.reward # +1 if ball in cup, 0 otherwise
    print(action, f'{step_reward if step_reward else 0.:.2e}', [f'{o:.2e}' for o in obs], f'{env.physics.finger_to_target_dist():.2e}')

    # if  env.physics.finger_to_target_dist() < 0.03:
    #     print(action, f'{step_reward if step_reward else 0.:.2e}', [f'{o:.2e}' for o in obs], env.physics.finger_to_target_dist())
    #     # print(action, f'{step_reward if step_reward else 0.:.2e}', [f'{o:.2e}' for o in obs], env.physics.finger_to_target_dist())
    # print(env.physics.named.model.geom_pos)
    # if step_reward and (step_reward > 1e-10):
    #     print(step_reward, obs, env.physics.mass_to_target_dist())

"""


def objective_function(params: np.ndarray,
                      num_runs: int,
                      ) -> float:
    import warnings
    from glfw import GLFWError
    warnings.filterwarnings("ignore", category=GLFWError)


    from dm_control import suite
    env = suite.load(domain_name="hopper", task_name="hop") 
    obs_spec = env.observation_spec()
    action_spec = env.action_spec()
    # initial_xpos = np.array([2.85, 2.49])

    min_reward = 1e5
    # initial_xpos = np.array([-2, 1])
    rewards = []
    for i in range(num_runs):
        time_step = env.reset()

        run_reward = 0.0
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
    obs size is 15: position (6), velocity (7), touch (2). return shape is (4,).
    """
    action = np.random.random(size=4)
    return action