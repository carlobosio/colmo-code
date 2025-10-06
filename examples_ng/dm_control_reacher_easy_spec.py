"""Finds a heuristic policy for the reacher task.  The simple two-link planar reacher with a randomised target location.
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
    env = suite.load(domain_name="reacher", task_name="easy") 
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

        if run_reward < min_reward:
            min_reward = run_reward
            rewards.append(min_reward)
    return -np.mean(rewards)

def concatenate_obs(time_step, obs_spec):
    return np.concatenate([time_step.observation[k].ravel() for k in obs_spec])

# def initialize_env(env, x_pos, y_pos):
#     env.physics.named.data.qpos['shoulder'][0] = x_pos
#     env.physics.named.data.qpos['wrist'][0] = y_pos

# def custom_reward(obs: np.ndarray) -> float:
#     x_cup = obs[0]
#     z_cup = obs[1]
#     x_ball = obs[2]
#     z_ball = obs[3]
#     angle = np.arctan2(x_ball - x_cup, z_ball - z_cup)
#     vx_ball = obs[6]
#     vz_ball = obs[7]
#     v_ball = np.sqrt(vx_ball**2 + vz_ball**2)
#     reward = 1 - np.abs(angle)/np.pi
#     if v_ball > 4.0:
#         reward -= 0.1*v_ball
#     return reward

@funsearch.evolve
def heuristic(obs: np.ndarray) -> np.ndarray:
    """Returns actions between -1 and 1.
    obs size is 6: position (2), to_target (2), velocity (2). return shape is (2,).
    """
    # action = np.zeros((2,)) + 1
    action = np.random.random(size=2)
    return action