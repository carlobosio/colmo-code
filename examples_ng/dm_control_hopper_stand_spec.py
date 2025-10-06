"""
You are Qwen, created by Alibaba Cloud. You are a helpful assistant. You are Python coding assistant specializing in reinforcement learning and control heuristics.  
You always produce concise, executable Python code that directly adheres to instructions.  

Your task is to design control policies for the planar one-legged hopper task, where success is measured by torso height stability and maintenance.  
The system receives a 15-dimensional observation array and must output a 4-dimensional action array:
- obs[0:6]: position: rootz, rooty, waist, hip, knee, ankle  
- obs[6:13]: velocity: rootx, rootz, rooty, waist, hip, knee, ankle  
- obs[13:15]: touch: touch_toe, touch_heel  
- action[0:4]: action: waist, hip, knee, ankle  

### Instructions
- Implement heuristic_v2 that is structurally different from earlier attempts, avoiding minor variations.  
- The heuristic should prioritize torso height, while ensuring stability and rhythmic locomotion.  
- Be creative (e.g., oscillators, phase-based rules, state-machine logic, proportional-derivative control), but remain concise.  
- The heuristic must be written in *pure Python*.
- It should always return an array of dimension 4.  
- Do not include comments, print statements, or explanations.  
- Ensure the code is self-contained and executable.  

###
"""

import numpy as np
import funsearch

@funsearch.run
def solve_non_param(num_runs) -> float:
    # Just evaluate the heuristic directly
    dummy_params = np.array([])
    cost = objective_function(dummy_params, num_runs=num_runs)
    return -cost, dummy_params

@funsearch.run_param
def solve_param(num_runs) -> float:
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
              budget=200, # number of function evaluations
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
    env = suite.load(domain_name="hopper", task_name="stand") 
    obs_spec = env.observation_spec()
    action_spec = env.action_spec()
    # initial_xpos = np.array([2.85, 2.49])

    min_reward = 1e5
    # initial_xpos = np.array([-2, 1])
    rewards = []
    for i in range(num_runs):
        time_step = env.reset()
        # initialize_env(env, x_pos=initial_xpos[0], y_pos=initial_xpos[1]) # random x and y position
        run_reward = 0.0
        # Observation: position (2), to_target (2), velocity (2)
        obs = concatenate_obs(time_step, obs_spec)
        for _ in range(250):
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
    # action = np.zeros((2,)) + 1
    action = np.random.random(size=4)
    return action