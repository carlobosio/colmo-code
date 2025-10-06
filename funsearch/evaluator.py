# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Class for evaluating programs proposed by the Sampler."""
import ast
import re
from collections.abc import Sequence
import copy
from typing import Any, Tuple, Optional, List

from funsearch import code_manipulation
from funsearch import programs_database
from funsearch import sandbox
from funsearch.float_extractor import ProgramWrapper
import logging
import hashlib

"""
  Regex to find all methods named 'heuristic_vX'.
  With each match, start from the 'def heuristic_vX(' and continue until there's a new line with any of
  - a new 'def'
  - ` or ' or # without indentation
"""

class _FunctionLineVisitor(ast.NodeVisitor):
  """Visitor that finds the last line number of a function with a given name."""

  def __init__(self, target_function_name: str) -> None:
    self._target_function_name: str = target_function_name
    self._function_end_line: int | None = None

  def visit_FunctionDef(self, node: Any) -> None:  # pylint: disable=invalid-name
    """Collects the end line number of the target function."""
    if node.name == self._target_function_name:
      self._function_end_line = node.end_lineno
    self.generic_visit(node)

  @property
  def function_end_line(self) -> int:
    """Line number of the final line of function `target_function_name`."""
    assert self._function_end_line is not None  # Check internal correctness.
    return self._function_end_line


def _find_method_implementation(generated_code: str, METHOD_MATCHER, METHOD_NAME_MATCHER) -> Tuple[str, str]:
  """Find the last method specified in METHOD_MATCHER within generated code.

  Return the code and the name of the method.
  """
  matches = METHOD_MATCHER.findall(generated_code)  
  if not matches:
    return "", ""
  last_match = matches[-1]
  name = METHOD_NAME_MATCHER.search(last_match).group()
  return last_match, name


def _trim_function_body(generated_code: str, METHOD_MATCHER, METHOD_NAME_MATCHER, method_str) -> str:
  """Extracts the body of the generated function, trimming anything after it."""
  if not generated_code:
    return ''
  if not type(generated_code) is str:
    generated_code = str(generated_code)

  method_name = "fake_function_header"
  # Check is the response only a continuation for our prompt or full method implementation with header
  if method_str in generated_code:
    code, method_name = _find_method_implementation(generated_code, METHOD_MATCHER, METHOD_NAME_MATCHER)
  else:
    code = f'def {method_name}():\n{generated_code}'

  # Finally parse the code to make sure it's valid Python
  tree = None
  # We keep trying and deleting code from the end until the parser succeeds.
  while tree is None:
    try:
      tree = ast.parse(code)
    except SyntaxError as e:
      code = '\n'.join(code.splitlines()[:e.lineno - 1]) # trimming the code from the error line onwards
  if not code:
    # Nothing could be saved from `generated_code`
    return ''

  visitor = _FunctionLineVisitor(method_name)
  visitor.visit(tree)
  body_lines = code.splitlines()[1:visitor.function_end_line]
  return '\n'.join(body_lines) + '\n\n'


def create_string_hash(input_string, algorithm='sha256'):
    """
    Creates a cryptographic hash of a given string using the specified algorithm.
    # # Example usage:
    # my_string = "Hello, world!"
    # sha256_hash = create_string_hash(my_string, 'sha256')
    # print(f"SHA-256 hash of '{my_string}': {sha256_hash}")

    Args:
        input_string (str): The string to be hashed.
        algorithm (str): The hashing algorithm to use (e.g., 'md5', 'sha1', 'sha256').

    Returns:
        str: The hexadecimal representation of the hash.
    """
    # Encode the string to bytes, as hashlib works with bytes
    encoded_string = input_string.encode('utf-8')

    # Create a hash object based on the specified algorithm
    if algorithm == 'md5':
        hasher = hashlib.md5()
    elif algorithm == 'sha1':
        hasher = hashlib.sha1()
    elif algorithm == 'sha256':
        hasher = hashlib.sha256()
    elif algorithm == 'sha512':
        hasher = hashlib.sha512()
    else:
        raise ValueError(f"Unsupported hashing algorithm: {algorithm}")

    # Update the hash object with the encoded string
    hasher.update(encoded_string)

    # Get the hexadecimal representation of the hash
    hex_digest = hasher.hexdigest()
    return hex_digest


def _sample_to_program(
    generated_sample: str,
    version_generated: int | None,
    template: code_manipulation.Program,
    function_to_evolve: str,
    parametric_program: bool = False,
    optimization_budget: int = 100,
    METHOD_MATCHER = None, 
    METHOD_NAME_MATCHER = None,
    method_str = None
) -> tuple[code_manipulation.Function, str]:
  """Returns the compiled generated function (generated_code) and the full runnable program.
     It does not alter the self._template.
  """  

  body = _trim_function_body(generated_sample, METHOD_MATCHER, METHOD_NAME_MATCHER, method_str)
  if version_generated is not None:
    body = code_manipulation.rename_function_calls(
        body,
        f'{function_to_evolve}_v{version_generated}',
        function_to_evolve)

  program = copy.deepcopy(template)



  evolved_function = program.get_function(function_to_evolve) # this basically returns a pointer to the function element
  evolved_function.body = body # modifies directly the function in variable `program`

  # # [carlo] extract floats from the body
  float_extractor = ProgramWrapper(body)
  evolved_function_copy = copy.deepcopy(evolved_function) # keep this with numeric params
  evolved_function_copy.body = body
  if parametric_program:
    evolved_function.body = float_extractor.sub_params()
  else:
    evolved_function.body = body

  # print("######################### 0 before sim ######################### ")
  hash_of_program = create_string_hash(evolved_function.body, algorithm='sha256')
  # print("######################### 1 before sim ######################### ")
  # print(evolved_function.body) # we should create an hash of this.
  # print("######################### 2 before sim ######################### ")

  program_str = str(program)

  if parametric_program:
    # logging.info(f"[Parametric program: {parametric_program}]\t Number of floats to optimize: {float_extractor.num_floats}")

    program_str = program_str.replace("heuristic(obs", "heuristic(params, obs")
    program_str = program_str.replace("num_params = 1", f"num_params = {float_extractor.num_floats}")
    program_str = program_str.replace("budget=100", f"budget={optimization_budget}")
  else:
    program_str = program_str.replace("num_params = 1", f"num_params = 0")

  return evolved_function_copy, program_str, hash_of_program # returns original function and parametric program
  # return evolved_function, program_str # returns original function and parametric program



def _calls_ancestor(program: str, function_to_evolve: str) -> bool:
  """Returns whether the generated function is calling an earlier version."""
  for name in code_manipulation.get_functions_called(program):
    # In `program` passed into this function the most recently generated
    # function has already been renamed to `function_to_evolve` (wihout the
    # suffix). Therefore any function call starting with `function_to_evolve_v`
    # is a call to an ancestor function.
    if name.startswith(f'{function_to_evolve}_v'):
      return True
  return False


class Evaluator:
    """Class that analyses functions generated by LLMs."""

    def __init__(
        self,
        database: programs_database.ProgramsDatabase,
        sbox: sandbox.DummySandbox,
        template: code_manipulation.Program,
        function_to_evolve: str,
        function_to_run: str,
        inputs: Sequence[Any],
        timeout_seconds: int = 30,
        optimize_floats: bool = False,
        optimization_budget: int = 300,
        significant_digits: int = 3,
        parametric_program: bool = False,
        spec_filename = None,
        ):
        """Initialize evaluator.
        
        Args:
            database: The database to store programs
            sbox: The sandbox to run programs
            template: The template program to evolve
            function_to_evolve: The function to evolve
            function_to_run: The function to run
            inputs: The input values to test the program
            timeout_seconds: The timeout for running programs
            optimize_floats: Whether to perform float optimization on programs
            optimization_budget: Number of evaluations for float optimization
            significant_digits: Number of significant digits for float values (to print in program)
        """

        logging.info(f"Initializing Evaluator with optimize_floats: {optimize_floats}")
        logging.info(f"Initializing Evaluator with optimization_budget: {optimization_budget}")
        logging.info(f"Initializing Evaluator with significant_digits: {significant_digits}")
        logging.info(f"Initializing Evaluator with parametric_program: {parametric_program}")

        self._database = database
        self._template = template
        self._tmp_template = copy.deepcopy(template)
        self._function_to_evolve = function_to_evolve
        self._function_to_run = function_to_run
        self._inputs = inputs
        self._timeout_seconds = timeout_seconds
        self._sandbox = sbox
        assert not optimize_floats, "Float optimization is deprecated"
        self._optimize_floats = optimize_floats
        self._optimization_budget = optimization_budget
        self._significant_digits = significant_digits
        self._parametric_program = parametric_program
        self._runs_ok_per_batch = 0
        self._evaluation_counter = 0
        # self._hash_set = set()

        print(f"spec_filename: {spec_filename}")
        if "swingup" in spec_filename:
          # use this for pendulum swingup
          self._METHOD_MATCHER = re.compile(r"def heuristic_v\d\(.*?\) -> float:(?:\s*(?:[ \t]*(?!def|#|`|').*(?:\n|$)))+")
          self._METHOD_NAME_MATCHER = re.compile(r"heuristic_v\d+")
          self._method_str = "def heuristic_v"
        
        else:
          # use this for ball in cup
          self._METHOD_MATCHER = re.compile(r"def heuristic_v\d\(.*?\) -> np.ndarray:(?:\s*(?:[ \t]*(?!def|#|`|').*(?:\n|$)))+")
          self._METHOD_NAME_MATCHER = re.compile(r"heuristic_v\d+")
          self._method_str = "def heuristic_v"

    def _evaluate_program(self, program: str, current_input: Any) -> Tuple[Optional[float], bool]:
        """Evaluate a single program on one input."""
        try:
            # [carlo] this now could also output the params
            # [matteo] TODO: we need to save the program with the parameters
            out, runs_ok = self._sandbox.run(program, self._function_to_run, current_input, self._timeout_seconds)
            if isinstance(out, (int, float)):
                return out, None, runs_ok  # Return tuple of (score, success)
            elif isinstance(out, Tuple):
                return out[0], out[1], runs_ok
            return None, None, False
        except Exception as e:
            print(f"Evaluation failed: {str(e)}")
            return None, None, False

    def evaluate_program_total_score(self, program_str: str) -> float:
        """Evaluate program on all inputs and return total score.
           I don't like this very much, we are just summing up the scores...
        Args:
            program_str: The program string to evaluate
            
        Returns:
            Total score across all inputs, or -1e6 if any evaluation fails
        """
        total_score = 0.0
        for current_input in self._inputs:
            score, _, runs_ok = self._evaluate_program(program_str, current_input)
            if runs_ok and score is not None:
                total_score += score
            else:
                return -1e6  # Penalize failed runs
        return total_score

    def _evaluate_with_full_program(self, function_body: str) -> float:
        """Evaluate a function body by inserting it into the full program.
        
        Args:
            function_body: The body of the function to evaluate
            
        Returns:
            float: Total score from evaluating the complete program
        """
        # Update the function body in the template
        self._tmp_template.get_function(self._function_to_evolve).body = function_body
        # Get the full program string
        full_program = str(self._tmp_template)
        # Evaluate using the full program
        return self.evaluate_program_total_score(full_program)

    def analyse(
        self,
        sample: str,
        island_id: int | None,
        version_generated: int | None,
        num_llm_inferences: int = None, 
        ) -> bool:
        """Compiles the sample (response directly from LLM output) 
        into a full program and executes it on test inputs."""
        
        RED = "\033[91m"
        GREEN = "\033[92m"
        RESET = "\033[0m"
        self._evaluation_counter +=1
        
        if not("return" in sample):
          print(sample)
          logging.info(f"{RED}Missing pieces{RESET}")
          return      

        # first we map the sample into a full program (basically paste it into the template, which is a parsed spec file)
        new_function, program, hash_of_program = _sample_to_program(
            generated_sample=sample, 
            version_generated=version_generated, 
            template=self._template, 
            function_to_evolve=self._function_to_evolve, 
            parametric_program=self._parametric_program,
            optimization_budget=self._optimization_budget, 
            METHOD_MATCHER = self._METHOD_MATCHER, 
            METHOD_NAME_MATCHER = self._METHOD_NAME_MATCHER,
            method_str = self._method_str)
        
        print(end='')
        
        if len(new_function.body) == 0:
          print(sample)
          logging.info(f"{RED}Empty body{RESET}")
          # return
        
        # evaluate the program normally to get baseline scores
        scores_per_test = {}
        input_scores = []  # Changed from input_output_pairs
        
        for current_input in self._inputs:

            test_output, params, runs_ok = self._evaluate_program(program, current_input) 

            if runs_ok:
              formatted_params = [f"{p:.2f}" for p in params]
              logging.info(f"{GREEN}[runs_ok: {runs_ok}] Params: {formatted_params} \t Test outputs: {test_output:.2f}{RESET}")

            else:
              logging.info(f"{RED}[runs_ok: {runs_ok}] Params {RESET}")
            

            if (runs_ok and not _calls_ancestor(program, self._function_to_evolve)
                and test_output is not None):
                if not isinstance(test_output, (int, float)):
                    raise ValueError('@function.run did not return an int/float score.')
                
                scores_per_test[current_input] = test_output
                input_scores.append((current_input, test_output))  # Store input and achieved score

        if runs_ok:
          self._runs_ok_per_batch += 1
          # logging.info(f"{scores_per_test}")
        else:
          print(sample)
        
        if scores_per_test:
            if self._parametric_program:
                # store the program with the correct parameters inside :) 
                float_extractor = ProgramWrapper(new_function.body)
                new_function.body = float_extractor.sub_floats(params)

            stop_experiment = self._database.register_program(new_function, hash_of_program, island_id, scores_per_test, num_llm_inferences = num_llm_inferences)

            return stop_experiment 

        return False


if __name__ == "__main__":
      import pathlib
      import time
      import argparse
      from funsearch import config, core, sandbox, code_manipulation
      from funsearch.__main__ import parse_input
      import logging 
      from funsearch import code_manipulation
      from funsearch import programs_database
      from funsearch import sandbox
      from funsearch.float_extractor import ProgramWrapper
      import logging

      # test the evaluator with ng inside spec file
      parser = argparse.ArgumentParser(description='Test the evaluator')
      parser.add_argument('--spec_file', type=str,  default='examples_ng/dm_control_finger_easy_spec.py', help='Path to the specification file')
      parser.add_argument('--inputs')
      args = parser.parse_args()
      file_name = args.spec_file
      # inputs = args.inputs
      # inputs = parse_input(inputs)

      specification = open(file_name, "r").read()
      function_to_evolve, function_to_run = core._extract_function_names(specification)
      template = code_manipulation.text_to_program(specification)

      timestamp = str(int(time.time()))
      folder_name = "evaluator_test_" + timestamp
      log_path = pathlib.Path("./data/") / folder_name
      if not log_path.exists():
        log_path.mkdir(parents=True)
        logging.info(f"Writing logs to {log_path}")

      def get_all_subclasses(cls):
          all_subclasses = []
          for subclass in cls.__subclasses__():
            all_subclasses.append(subclass)
            all_subclasses.extend(get_all_subclasses(subclass))
          return all_subclasses

      SANDBOX_TYPES = get_all_subclasses(sandbox.DummySandbox) + [sandbox.DummySandbox]
      SANDBOX_NAMES = [c.__name__ for c in SANDBOX_TYPES]
      sandbox_type = "DummySandbox"
      sbox = next(c for c in SANDBOX_TYPES if c.__name__ == sandbox_type)(base_path=log_path)

      conf = config.Config()
      database = programs_database.ProgramsDatabase(conf.programs_database,  template,  function_to_evolve, identifier=timestamp, log_path=log_path)
      from funsearch.evaluator import Evaluator 
      inputs = [1]# if "dm_control_swingup_spec.py" in file_name.split("/") else [0.5]
      optimize_floats = False
      optimization_budget = 100
      parametric_program = True
      evaluator = Evaluator(database, 
                            sbox, 
                            template, 
                            function_to_evolve, 
                            function_to_run, 
                            inputs, 
                            optimize_floats=optimize_floats, 
                            optimization_budget=optimization_budget, 
                            parametric_program=parametric_program, 
                            spec_filename=file_name)


      if "dm_control_swingup_spec.py" in file_name.split("/"):
        sample = """def heuristic_v1(obs: np.ndarray) -> float:
    x1 = np.arctan2(-obs[1], obs[0])                                                                                                                                                                       
    x2 = obs[2]                                                                                                                                                                                            

    # Adjust action based on x1 and x2
    if x1 < -np.pi / 5.312 or x1 > np.pi / 3.971:
        action = 4.505 * np.sign(x2)
    else:
        action = 5*np.sin(x1) + 5.99 * x2

    return action"""
      elif "inv_pendulum_spec.py" in file_name.split("/"):
        sample = """def heuristic_v1(obs: np.ndarray) -> float:
    x1 = obs[0]
    x2 = obs[1]
    action = 0.0*x1 + 3.5*x2
    return action"""
      elif "dm_control_ballcup_spec.py" in file_name.split("/"):
        sample = """def heuristic_v1(obs: np.ndarray) -> np.ndarray:
    action = np.zeros((2,))
    # Simple heuristic: move towards the center if not already there
    action[0] += np.sign(0.5 - obs[3])
    action[1] += np.sign(0.5 - obs[4])

    return action"""
      elif "dm_control_quadruped_run_spec.py" in file_name.split("/"):
        sample = """def heuristic_v1(obs: np.ndarray) -> np.ndarray:
    egocentric_state = obs[:44]
    torso_velocity = obs[44:47]
    torso_upright = obs[47]
    imu = obs[48:54]
    force_torque = obs[54:78]

    # Example improvement: Use torso velocity and upright state to generate actions
    velocity_norm = np.linalg.norm(torso_velocity)
    upright_state = np.clip(torso_upright, 0.5, 1.0)

    # Simple linear combination of these features to generate actions
    actions = velocity_norm * upright_state * np.array([0.1] * 12)
    return actions"""
      elif "mujoco_quadcopter_spec.py" in file_name.split("/"):
        sample = """def heuristic_v1(obs: np.ndarray) -> np.ndarray:
    position = obs[:3]
    orientation = obs[3:7]
    velocity = obs[7:10]
    angular_velocity = obs[10:]
    return np.array([0.0, 0.0, 0.0, 0.0])"""
      else:
        sample = """def heuristic_v1(obs: np.ndarray) -> float:
    x1 = np.arctan2(-obs[1], obs[0])
    x2 = np.arctan2(-obs[2], obs[3])
    x3 = obs[4]
    x4 = obs[5]
    action = 0
    if x3 > 0:
      action += x3
    else:
      action -= x3
    if x4 > 0:
      action += x4
    else:
      action -= x4
    return action"""
      start_time = time.time()
      evaluator.analyse(sample, island_id=0, version_generated=0)
      end_time = time.time()
      print(f"Time taken: {end_time - start_time} seconds")

