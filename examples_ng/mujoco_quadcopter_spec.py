"""Finds a heuristic policy for the quadcopter hover task.
Find a heuristic policy that stabilizes the quadcopter in place at the origin. It should not rotate or move.
Analytic functions are better than hard-coded if statements.
On every iteration, improve heuristic_v1 over the heuristic_vX methods.
Make only small changes. Keep the code short.
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
    # ng_optimizer = ng.optimizers.ParametrizedBO(
    #                     initialization="random",
    #                     init_budget=None,
    #                     middle_point=False,
    #                     utility_kind='ucb',
    #                     utility_kappa=2.576,
    #                     utility_xi=0.0,
    #                     gp_parameters=None
    #                 )(parametrization=ng.p.Array(shape=(num_params,)),
    #                   budget=100) # number of function evaluations
                        
    with ProcessPoolExecutor(max_workers=12) as executor:
        solution = ng_optimizer.minimize(objective_partial, executor=executor, batch_mode=True)
    print("solution", solution)
    optimized_params = solution.value
    loss = objective_partial(optimized_params)
    # print("best_sol", best_sol)
    print("loss", loss)
    score = -loss

    return score, optimized_params
  


def objective_function(params: np.ndarray,
                      num_runs: int,
                      ) -> float:
    import warnings
    from glfw import GLFWError
    warnings.filterwarnings("ignore", category=GLFWError)


    import mujoco
    # Load the XML model
    model = mujoco.MjModel.from_xml_path("/home/cbosio/fun-design/examples_ng/quadcopter.xml")
    # Create a data object for simulation state
    data = mujoco.MjData(model)
    np.random.seed(42)

    def reset_random(data):
        mujoco.mj_resetData(model, data)  # reset to nominal
        
        # Randomize position
        data.qpos[0:3] = np.random.uniform(low=[-0.3, -0.3, -0.5],
                                        high=[ 0.3,  0.3, 0.5])
        
        # Randomize orientation (small random rotation)
        axis = np.random.randn(3)
        axis /= np.linalg.norm(axis)
        angle = np.random.uniform(-0.2, 0.2)  # radians
        qw = np.cos(angle / 2)
        qx, qy, qz = np.sin(angle / 2) * axis
        data.qpos[3:7] = [qw, qx, qy, qz]

        # Randomize velocities
        data.qvel[:] = np.random.uniform(-0.1, 0.1, size=model.nv)

    rewards = []
    for i in range(num_runs):
        # Reset data to initial state
        reset_random(data)
        run_reward = 0.0
        # Simulation loop
        for step in range(10000):
            # Read state
            qpos = data.qpos.copy()  # positions
            qvel = data.qvel.copy()  # velocities
            obs = np.concatenate([qpos, qvel])

            # Set control inputs (torques/thrusts)
            # data.ctrl[:] = [0.0, 0.0, 0.0, 0.0]  # Example: 4 rotor thrusts
            data.ctrl[:] = heuristic(obs)
            # Step physics
            mujoco.mj_step(model, data)
            action = data.actuator_force[:]
            # penalize position and yaw
            step_reward = 0.1 - 1/1000*(2.0*np.linalg.norm(qpos[0:3])**2
                                        + 5.0*np.linalg.norm(qpos[3])**2 
                                        + 0.1*(action[0] - 9.81)**2
                                        + 1.0*np.linalg.norm(action[1:])**2)
            run_reward += step_reward

        rewards.append(run_reward)
    return -np.mean(rewards)

# def concatenate_obs(time_step, obs_spec):
#     return np.concatenate([time_step.observation[k].ravel() for k in obs_spec])


@funsearch.evolve
def heuristic(obs: np.ndarray) -> np.ndarray:
    """obs size is 13: position (3), orientation (4), velocity (3), angular velocity (3). 
    return shape is (4,): total thrust and torques around x, y, z axes.
    """
    # action = np.zeros((2,)) + 1
    action = np.random.random(size=4)
    return action