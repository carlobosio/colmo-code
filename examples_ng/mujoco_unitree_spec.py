"""Finds a heuristic policy for the unitree quadruped to walk.  
The quadruped is initialised in a random configuration. It should walk moving forward.
Analytic functions are better than hard-coded numbers.
On every iteration, improve heuristic_v1 over the heuristic_vX methods.
Make only small changes. 
"""

import numpy as np
import funsearch


# @funsearch.run
# def solve_non_param(num_runs) -> float:
#     # Just evaluate the heuristic directly
#     dummy_params = np.array([])
#     cost = objective_function(dummy_params, num_runs=num_runs)
#     return -cost, dummy_params

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


    import mujoco
    # Load the XML model
    # model = mujoco.MjModel.from_xml_path("example_ng/scene_a1.xml")
    model = mujoco.MjModel.from_xml_path("/home/cbosio/fun-design/examples_ng/scene_a1.xml")

    # Create a data object for simulation state
    data = mujoco.MjData(model)
    np.random.seed(42)

    # this is a stable position
    qpos_stable = np.array([ 0.0747819, -0.00250327, 0.297014, 0.982411, 0.0169028, 0.173042, -0.0681166, -0.261824, 0.106889, -1.67226, 0.37394, 0.0530679, -1.53289, -0.021584, -0.012171, -0.967322, 0.0102457, -0.0160053, -0.954448 ])
    qvel_stable = np.array([ -2.68742e-05, 6.60169e-06, -1.10607e-05, 5.9525e-06, -4.31382e-05, -2.40179e-05, -0.000124619, 0.000159939, 1.81136e-05, 0.000122983, 4.13959e-06, 0.000107984, -0.00010803, -0.000127288, -1.35778e-05, 8.55673e-05, -1.32014e-05, -6.36596e-05 ])
    ctrl_stable = np.array([ -0.256928, 0.1311, -1.64639, 0.353276, 0.07874, -1.49503, 0, 0, 0, 0, 0, 0 ])

    # kneeling down
    qpos_kneeling = np.array([0.21786, -0.0404961, 0.239809, 0.940305, 0.0142138, 0.339918, -0.00899535, -0.253175, 0.205447, -2.69373, 0.384122, 0.0537244, -2.32895, -0.0282176, 0.0103672, -0.955282, -0.00177678, 0.00500136, -0.916689])
    qvel_kneeling = np.array([ 1.49099e-05, -2.22474e-05, -4.68756e-06, 4.42746e-05, 4.3666e-05, 0.000133908, -7.51805e-05, 0.000119192, 1.41947e-07, -2.23065e-05, -0.000110449, 5.45502e-05, 1.08657e-05, 0.000174452, 5.26502e-05, -5.56769e-06, -0.000327803, -0.00018357])
    ctrl_kneeling = np.array([-0.256928, 0.18346, -2.697, 0.353276, 0.07874, -2.23402, 0, 0, 0, 0, 0, 0 ])


    # another starting state
    qpos_another = np.array([ 0.0013086, 0.00654509, 0.243641, 0.909149, 0.0245889, -0.0493595, -0.412804, -0.0512622, 0.546474, -1.74656, 0.409966, 0.447184, -1.14358, -0.485439, 0.563781, -1.56537, 0.0316025, -0.00804151, -1.30438 ])
    qvel_another = np.array([ -1.10627e-05, 0.000194929, -8.93969e-05, -0.000289785, -0.000566213, 0.000196864, 9.60049e-05, 3.94665e-05, 0.00019782, 0.000303767, 0.000504482, 0.000112599, -0.000563961, 0.000239257, 4.89132e-05, 0.000156098, -0.000548316, 4.9872e-05 ])
    ctrl_another = np.array([ -0.056203, 0.54998, -1.74433, 0.40145, 0.47144, -1.12108, -0.433566, 0.62852, -1.46832, 0, 0.07874, -1.20121 ])

    ctrl_ranges = {
      "abduction": (-0.402851, 0.402851),
      "hip": (-0.5472, 1.18879),
      "knee": (-1.79653, -0.916298),
    }

    actuators = [
        ("FR_hip", "abduction"),
        ("FR_thigh", "hip"),
        ("FR_calf", "knee"),
        ("FL_hip", "abduction"),
        ("FL_thigh", "hip"),
        ("FL_calf", "knee"),
        ("RR_hip", "abduction"),
        ("RR_thigh", "hip"),
        ("RR_calf", "knee"),
        ("RL_hip", "abduction"),
        ("RL_thigh", "hip"),
        ("RL_calf", "knee"),
    ]

    ctrl_list = [ctrl_ranges[cls] for _, cls in actuators]
    mins, maxs = zip(*ctrl_list)
    # def compute_reward(lst_x, lst_y, alpha=1.0, beta=0.5,  y_ref=0.0):
    #     """
    #     Compute reward based on maximum x distance and penalty for wandering along y.

    #     Parameters:
    #     - lst_x: list of x positions (floats)
    #     - lst_y: list of y positions (floats)
    #     - alpha: penalty strength for wandering along y
    #     - y_ref: reference y position (usually starting y)

    #     Returns:
    #     - reward (float)
    #     """
    #     if len(lst_x) < 2 or len(lst_y) < 2:
    #         return 0.0

    #     # # maximum distance along x axis (current max x minus initial x)
    #     # max_x_dist = max(lst_x) - lst_x[0]

    #     # # penalty for moving back
    #     # # sum of absolute differences between consecutive x positions
    #     # x_movements = sum( np.maximum((lst_x[i] - lst_x[i-1]), 0) for i in range(1, len(lst_x)))
    #     max_x_dist = - np.sum(np.diff(lst_x))

    #     # penalty for y wandering as sum of absolute distances from y_ref
    #     y_wandering = sum(abs(y - y_ref) for y in lst_y)

    #     # combined reward
    #     reward = max_x_dist + alpha * y_wandering #- beta * x_movements

    #     return reward

    def reset_random(data, seed):
        np.random.seed(seed)
        mujoco.mj_resetData(model, data)  # reset to nominal
        
        # TODO: 
        # Randomize position
        # Randomize orientation (small random rotation)
        data.qpos = qpos_stable
        # Randomize velocities
        data.qvel = qvel_stable
        # Randomize control
        data.ctrl = ctrl_stable

    rewards = []
    for seed in range(num_runs):
        # Reset data to initial state
        reset_random(data, seed)
        run_reward = 0.0
        # Simulation loop
        for step in range(10000):
            # Read state
            qpos = data.qpos.copy()  # positions
            qvel = data.qvel.copy()  # velocities
            obs = np.concatenate([qpos, qvel])

            # Set control inputs (torques/thrusts)
            # data.ctrl[:] = [0.0, 0.0, 0.0, 0.0]  # Example: 4 rotor thrusts
            controls = heuristic(obs)
            
            data.ctrl[:] = np.clip(controls, mins, maxs)
            # Step physics
            mujoco.mj_step(model, data)
            # penalize position and yaw
            
            # TODO: improve the reward
            step_reward = 0.1*min(1.0, max(0.0, qvel[0]/10.0)) # max reward was 21k bring the value down
            
            # step_reward = 0.1 - 0.5*np.linalg.norm(qpos[0:3]) - 2.0*np.linalg.norm(qpos[3])
            run_reward += step_reward

        rewards.append(run_reward)
    return -np.mean(rewards)


@funsearch.evolve
def heuristic(obs: np.ndarray) -> np.ndarray:
    """Returns an array of shape is (12,)
    obs size is 27, obs[0:19] position, orientation and joint angles, obs[19:27] body and joint velocities
    """
    # stable upright control
    return np.random.random(12)
