
import re
text = '''```python
def heuristic_v2(obs: np.ndarray) -> float:
  """Improved version of `heuristic_v1`."""
  x1 = np.arctan2(-obs[1], obs[0]) * -0.551
  x2 = obs[2] * -3.862
  action = -x1 + -5.814 * x2 + 0.2 * x1 ** 2
  return action
```'''

import pdb; pdb.set_trace()
import re

def extract_first_python_code_block(text):
    match = re.search(r"```python(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else None

extract_first_python_code_block(text)
