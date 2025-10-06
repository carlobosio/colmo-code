"""Find a heuristic policy for the pendulum swingup task.

On every iteration, improve heuristic_v1 over the heuristic_vX methods from previous iterations.
Make only small changes. Try to make the code short. Only add the missing function body, do not return additional code.
"""

import numpy as np
import funsearch
  
@funsearch.run
def solve(init_angle) -> float:
  """Returns the reward for a heuristic.
  """
  import nevergrad as ng
  from loky import ProcessPoolExecutor
  from functools import partial

  num_params = 1
  if num_params == 0:
    # Just evaluate the heuristic directly
    dummy_params = np.array([])
    cost = objective_function(dummy_params, init_angle=init_angle)
    return -cost, dummy_params
  
  
  objective_partial = partial(objective_function, 
                              init_angle=init_angle,
                              )

  ng_optimizer = ng.optimizers.OnePlusOne(
            parametrization=ng.p.Array(shape=(num_params,)),
            budget=100,
            num_workers=12
            )
  with ProcessPoolExecutor(max_workers=ng_optimizer.num_workers) as executor:
    solution = ng_optimizer.minimize(objective_partial, executor=executor, batch_mode=False)
  
  optimized_params = solution.value
  score = -solution.loss

  return score, optimized_params

def objective_function(params: np.ndarray,
                      init_angle: int,
                      ) -> float:
    initial_state = np.array([init_angle, 0.0], dtype=np.float32)
    horizon_length = 1000
    sampling_time = 0.01
    state = initial_state.copy()
    cost_tot = 0.0

    for _ in range(horizon_length):
      obs = state
      control_input = heuristic(obs)
      state = simulate(state, control_input, sampling_time)
      step_cost = np.linalg.norm(state)**2 + control_input**2 - 1.0# Q is identity, R is 1
      cost_tot += step_cost
    
    if np.isfinite(cost_tot):
      return float(cost_tot)
    else:
      # print(f"[run] output rmse is not finite: {rmse_value}")
      return 1e5

def simulate(state: np.ndarray, control_input: float, sampling_time: float) -> np.ndarray:
  """Simulates a step.
  """
  next_state = state.copy()
  next_state[0] += state[1] * sampling_time
  next_state[0] = ((next_state[0] + np.pi) % (2 * np.pi)) - np.pi
  next_state[1] += (np.sin(state[0]) - state[1] + control_input) * sampling_time

  return next_state

@funsearch.evolve
def heuristic(obs: np.ndarray) -> float:
  """Returns an action between -1 and 1.
  obs size is 2.
  """
  action = 0.0
  return action