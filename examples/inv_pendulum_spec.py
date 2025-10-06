"""Finds a control ***heuristic*** to stabilize a two dimensional nonlinear system.

On every iteration, improve heuristic_v1 over the heuristic_vX methods from previous iterations.
Make only small changes.
Try to make the code short and be creative with the method you use.
"""

import numpy as np

import funsearch


@funsearch.run
def evaluate(init_angle) -> float:
  """Returns the negative rmse score for a heuristic."""
  cost = solve(init_angle)
  # print(f"[run] output rmse: {rmse_value}")
  if np.isfinite(cost):
    return float(-cost)
  else:
    # print(f"[run] output rmse is not finite: {rmse_value}")
    return -100.0


def solve(init_angle) -> float:
  """Returns the RMSE value for a run of the inverted pendulum."""
  initial_state = np.array([init_angle, 0.0], dtype=np.float32)
  horizon_length = 100
  sampling_time = 0.1
  state = initial_state.copy()
  J = 0.0

  for _ in range(horizon_length):
    control_input = heuristic(state)
    state = simulate(state, control_input, sampling_time)
    J += np.linalg.norm(state)**2 + control_input**2 # Q is identity, R is 1
  
  return J / horizon_length

def simulate(state: np.ndarray, control_input: float, sampling_time: float) -> np.ndarray:
  """Simulates a step.
  """
  next_state = state.copy()
  next_state[0] += state[1] * sampling_time
  # make sure within -pi and pi
  next_state[0] = ((next_state[0] + np.pi) % (2 * np.pi)) - np.pi
  next_state[1] += (np.sin(state[0]) - state[1] + control_input) * sampling_time

  return next_state


@funsearch.evolve
def heuristic(state: np.ndarray) -> float:
  """Returns a control input. state is a 2D array contaning x and x_dot.
  The function is going to return a float input value.
  """
  return 0.0