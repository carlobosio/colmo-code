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

"""A programs database that implements the evolutionary algorithm."""
import pathlib
import pickle
from collections.abc import Mapping, Sequence
import copy
import dataclasses
import time
from typing import Any, Iterable, Tuple

from absl import logging
import numpy as np
import scipy

from funsearch import code_manipulation
from funsearch import config as config_lib
import wandb
import re
import multiprocessing as mp
import threading


# ANSI color codes
RESET   = "\033[0m"
BLACK   = "\033[30m"
RED     = "\033[31m"
GREEN   = "\033[32m"
YELLOW  = "\033[33m"
BLUE    = "\033[34m"
MAGENTA = "\033[35m"
CYAN    = "\033[36m"
WHITE   = "\033[37m"

# Bright versions
BRIGHT_BLACK   = "\033[90m"
BRIGHT_RED     = "\033[91m"
BRIGHT_GREEN   = "\033[92m"
BRIGHT_YELLOW  = "\033[93m"
BRIGHT_BLUE    = "\033[94m"
BRIGHT_MAGENTA = "\033[95m"
BRIGHT_CYAN    = "\033[96m"
BRIGHT_WHITE   = "\033[97m"

def print_color(msg, color):
    print(f"{color}{msg}{RESET}")

Signature = tuple[float, ...]
Signature = tuple[str, ...] # hash

ScoresPerTest = Mapping[Any, float]
STRIP_REGEX = re.compile(r'"""Improved version of `heuristic_v\d+`\."""')

def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
  """Returns the tempered softmax of 1D finite `logits`."""
  if not np.all(np.isfinite(logits)):
    non_finites = set(logits[~np.isfinite(logits)])
    print("logits: ", logits)
    raise ValueError(f'`logits` contains non-finite value(s): {non_finites}')
  if not np.issubdtype(logits.dtype, np.floating):
    logits = np.array(logits, dtype=np.float32)

  result = scipy.special.softmax(logits / temperature, axis=-1)
  # Ensure that probabilities sum to 1 to prevent error in `np.random.choice`.
  index = np.argmax(result)
  result[index] = 1 - np.sum(result[0:index]) - np.sum(result[index+1:])
  return result


def _reduce_score(scores_per_test: ScoresPerTest) -> float:
  """Reduces per-test scores into a single score."""
  return scores_per_test[list(scores_per_test.keys())[-1]]


def _get_signature(scores_per_test: ScoresPerTest) -> Signature:
  """Represents test scores as a canonical signature."""
  return tuple(scores_per_test[k] for k in sorted(scores_per_test.keys()))


@dataclasses.dataclass(frozen=True)
class Prompt:
  """A prompt produced by the ProgramsDatabase, to be sent to Samplers.

  Attributes:
    code: The prompt, ending with the header of the function to be completed.
    version_generated: The function to be completed is `_v{version_generated}`.
    island_id: Identifier of the island that produced the implementations
       included in the prompt. Used to direct the newly generated implementation
       into the same island.
  """
  code: str
  version_generated: int
  island_id: int


class ProgramsDatabase:
  """A collection of programs, organized as islands."""

  def __init__(
      self,
      config: config_lib.ProgramsDatabaseConfig,
      template: code_manipulation.Program,
      function_to_evolve: str,
      identifier: str = "",
      log_path=None,
      wandb_run = None
  ) -> None:
    # import pdb; pdb.set_trace()

    self._config: config_lib.ProgramsDatabaseConfig = config
    self._template: code_manipulation.Program = template
    self._function_to_evolve: str = function_to_evolve
    self.log_path = log_path
    if wandb_run:
      self._wandb_run = wandb_run

    # Initialize empty islands.
    self._islands: list[Island] = []
    self._locks: list[mp.Lock] = []

    # import pdb; pdb.set_trace()
    for _ in range(config.num_islands):
      self._islands.append(
          Island(template, function_to_evolve, config.functions_per_prompt,
                 config.cluster_sampling_temperature_init,
                 config.cluster_sampling_temperature_period))
      # Create ONE global lock
      self._locks.append(mp.Lock())
    
    self._best_score_per_island: list[float] = (
        [-float('inf')] * config.num_islands)
    self._best_program_per_island: list[code_manipulation.Function | None] = (
        [None] * config.num_islands)
    self._best_program_per_island_hash: list[str | None] = (
        [None] * config.num_islands)
    self._best_scores_per_test_per_island: list[ScoresPerTest | None] = (
        [None] * config.num_islands)

    self._last_reset_time: float = time.time()
    self._program_counter = 0
    self._backups_done = 0
    self.identifier = identifier

    self._best_scores = []
    self._best_scores_iter = []

  def get_best_programs_per_island(self) -> Iterable[Tuple[code_manipulation.Function | None]]:
    return sorted(zip(self._best_program_per_island, self._best_score_per_island), key=lambda t: t[1], reverse=True)

  def save(self, file):
    """Save database to a file"""
    data = {}
    keys = ["_islands", "_best_score_per_island", "_best_program_per_island", "_best_scores_per_test_per_island", "_best_scores", "_best_scores_iter"]
    for key in keys:
      data[key] = getattr(self, key)
    pickle.dump(data, file)

  def load(self, file):
    """Load previously saved database"""
    data = pickle.load(file)
    print(f"Loading: {data.keys()}")
    print(f"Loading N keys: {len(data.keys())}")

    for key in data.keys():
      setattr(self, key, data[key])

  def backup(self, custom_naming = None):
    if custom_naming:
      filename = f"program_db_{self._function_to_evolve}_{self.identifier}_{self._backups_done}_ninference_{custom_naming}.pickle"

    else:
      filename = f"program_db_{self._function_to_evolve}_{self.identifier}_{self._backups_done}.pickle"
    # p = pathlib.Path(self._config.backup_folder)
    p = pathlib.Path(self.log_path)
    if not p.exists():
      p.mkdir(parents=True, exist_ok=True)
    filepath = p / filename
    logging.info(f"Saving backup to {filepath}.")

    with open(filepath, mode="wb") as f:
      self.save(f)
    self._backups_done += 1

  def get_prompt(self) -> Prompt:
    """Returns a prompt containing implementations from one chosen island."""
    island_id = np.random.randint(len(self._islands))
    logging.info(f"Prompts from island: {island_id}")
    
    code, version_generated = self._islands[island_id].get_prompt()
    return Prompt(code, version_generated, island_id)

  def wandb_log(self, steps):
      if self._wandb_run:
          score_lst = []
          score_per_test_lst = []

          columns = ["Island", "Best_program"]
          table = wandb.Table(columns=columns)

          for island_id, score in enumerate(self._best_score_per_island):
              program = self._best_program_per_island[island_id]
              # hash_of_program = self._best_program_per_island_hash[island_id]
              # scores_per_test = self._best_scores_per_test_per_island[island_id]

              self._wandb_run.log({
                  # f"best_score": score_max,
                  f"island{island_id:02d}_best_score": score,
                  # f"island{island_id:02d}_program": program.body,
                  # f"island{island_id:02d}_best_score_test": scores_per_test
              },  step = steps)

              table.add_data(island_id, program.body)              
              score_lst.append(score)
              # score_per_test_lst.append(scores_per_test)
          
          self._wandb_run.log({"programs": table}, step = steps)
          self._wandb_run.log({f"best_score": np.max(score_lst)},  step = steps) #, "best_score_test": np.max(score_per_test_lst)})

    # if self._wandb_run:

    #   self._best_score_per_island[island_id]:
    #     self._best_program_per_island[island_id] = program
    #     self._best_program_per_island_hash[island_id] = hash_of_program
    #     self._best_scores_per_test_per_island[island_id] = scores_per_test
    #     self._best_score_per_island[island_id] = score

    #     self._wandb_run.log({f"island{island_id:02d}_best_score": score, 
    #                           f"island{island_id:02d}_program": program, 
    #                           f"island{island_id:02d}_best_score_test": scores_per_test}) #, step=num_llm_inferences)

  def _register_program_in_island(
      self,
      program: code_manipulation.Function,
      hash_of_program: str,
      island_id: int,
      scores_per_test: ScoresPerTest,
      num_llm_inferences = None,
  ) -> float:
    """Registers `program` in the specified island."""
    with self._locks[island_id]:
      score = _reduce_score(scores_per_test)
      if score < self._config.min_score and self._islands[island_id]._num_programs > 0:
        print_color('Score is below the min score, skipping', BRIGHT_YELLOW)
        return -float('inf')
      self._islands[island_id].register_program(program, hash_of_program, scores_per_test)
      if score > self._best_score_per_island[island_id]:
        self._best_program_per_island[island_id] = program
        self._best_program_per_island_hash[island_id] = hash_of_program
        self._best_scores_per_test_per_island[island_id] = scores_per_test
        self._best_score_per_island[island_id] = score



        logging.info('Best score of island %d increased to %s', island_id, score)
        # Save the best program to a file.
        with open(self.log_path / f'best_program_{island_id}.py', 'w') as f:
          f.write(str(program)) 
      return score

  def register_program(
      self,
      program: code_manipulation.Function,
      hash_of_program: str,
      island_id: int | None,
      scores_per_test: ScoresPerTest,
      num_llm_inferences = None,
  ) -> bool:
    
    """Registers `program` in the database."""
    # In an asynchronous implementation we should consider the possibility of
    # registering a program on an island that had been reset after the prompt
    # was generated. Leaving that out here for simplicity.
    if island_id is None:
      # This is a program added at the beginning, so adding it to all islands.
      for island_id in range(len(self._islands)):
        score = self._register_program_in_island(program, hash_of_program, island_id, scores_per_test, num_llm_inferences)
    else:
      score = self._register_program_in_island(program, hash_of_program, island_id, scores_per_test, num_llm_inferences)

    # if num_llm_inferences % 100 == 0:
    #   self.backup(custom_naming=f'{num_llm_inferences:05d}')

    # Check whether it is time to reset an island.
    if (time.time() - self._last_reset_time > self._config.reset_period):
      self._last_reset_time = time.time()
      print("Resetting islands")
      self.reset_islands()
    
    # print(f"Number of programs in the program database: {self._program_counter}")
    # Backup every N iterations
    # if self._program_counter > 0:
    self._program_counter += 1
    # if self._program_counter % self._config.backup_period == 0:
    #   self.backup()
    
    if len(self._best_scores) == 0:
      self._best_scores.append(score)
      self._best_scores_iter.append(self._program_counter)
    elif score > self._best_scores[-1]:
      self._best_scores.append(score)
      self._best_scores_iter.append(self._program_counter)

    stop_experiment = score > self._config.score_threshold
    if stop_experiment:
      self.backup()
    return stop_experiment

  def reset_islands(self) -> None:
    """Resets the weaker half of islands."""
    # We sort best scores after adding minor noise to break ties.
    indices_sorted_by_score: np.ndarray = np.argsort(
        self._best_score_per_island +
        np.random.randn(len(self._best_score_per_island)) * 1e-6)
    num_islands_to_reset = self._config.num_islands // 2
    reset_islands_ids = indices_sorted_by_score[:num_islands_to_reset]
    keep_islands_ids = indices_sorted_by_score[num_islands_to_reset:]
    for island_id in reset_islands_ids:
      self._islands[island_id] = Island(
          self._template,
          self._function_to_evolve,
          self._config.functions_per_prompt,
          self._config.cluster_sampling_temperature_init,
          self._config.cluster_sampling_temperature_period)
      self._best_score_per_island[island_id] = -float('inf')
      founder_island_id = np.random.choice(keep_islands_ids)
      founder = self._best_program_per_island[founder_island_id]
      founder_scores = self._best_scores_per_test_per_island[founder_island_id]
      founder_hash = self._best_program_per_island_hash[founder_island_id]
      self._register_program_in_island(founder, founder_hash, island_id, founder_scores)


class Island:
  """A sub-population of the programs database."""

  def __init__(
      self,
      template: code_manipulation.Program,
      function_to_evolve: str,
      functions_per_prompt: int,
      cluster_sampling_temperature_init: float,
      cluster_sampling_temperature_period: int,
  ) -> None:
    self._template: code_manipulation.Program = template
    self._function_to_evolve: str = function_to_evolve
    self._functions_per_prompt: int = functions_per_prompt
    # import pdb; pdb.set_trace()
    self._cluster_sampling_temperature_init = cluster_sampling_temperature_init
    self._cluster_sampling_temperature_period = (
        cluster_sampling_temperature_period)

    self._clusters: dict[Signature, Cluster] = {}
    self._num_programs: int = 0
    self._best_scores = []
    self._best_scores_iter = []

  def register_program(
      self,
      program: code_manipulation.Function,
      hash_of_program: str, 
      scores_per_test: ScoresPerTest,
  ) -> None:
    
    """Stores a program on this island, in its appropriate cluster."""
    # signature = _get_signature(scores_per_test)
    signature = hash_of_program
    score = _reduce_score(scores_per_test)
    program.body = STRIP_REGEX.sub('', program.body) # removing the docstring
    if signature not in self._clusters:
      # import pdb; pdb.set_trace()
      # logging.info()
      print_color("New cluster new structure", BRIGHT_MAGENTA)
      self._clusters[signature] = Cluster(score, program)
    else:
      # import pdb; pdb.set_trace()
      # logging.info("")
      if score > self._clusters[signature].score:
        print_color("Old cluster old structure", BRIGHT_BLUE)
        self._clusters[signature]._score = score
        self._clusters[signature]._programs[0] = program
        # self._clusters[signature].register_program(program)



    self._num_programs += 1

    if len(self._best_scores) == 0:
      self._best_scores.append(score)
      self._best_scores_iter.append(self._num_programs)
    elif score > self._best_scores[-1]:
      self._best_scores.append(score)
      self._best_scores_iter.append(self._num_programs)

  def get_prompt(self) -> tuple[str, int]:
    """Constructs a prompt containing functions from this island."""
    
    # import pdb; pdb.set_trace()
    signatures = list(self._clusters.keys())
    cluster_scores = np.array(
        [self._clusters[signature].score for signature in signatures])

    # Convert scores to probabilities using softmax with temperature schedule.
    # period = self._cluster_sampling_temperature_period
    # temperature = self._cluster_sampling_temperature_init * (
    #     1 - (self._num_programs % period) / period)
    temperature = self._cluster_sampling_temperature_init # TODO: check how this value gets overwritten
    # temperature = 10.0
    # print("Softmax in get_prompt, cluster scores: ", cluster_scores)
    probabilities = _softmax(cluster_scores, temperature)
    # count non-zero probabilities
    non_zero_probabilities = np.sum(probabilities != 0)

    # At the beginning of an experiment when we have few clusters, place fewer
    # programs into the prompt.
    functions_per_prompt = min(len(self._clusters), self._functions_per_prompt, non_zero_probabilities)
    # import pdb; pdb.set_trace()
    idx = np.random.choice(
        len(signatures), size=functions_per_prompt, replace=False, p=probabilities)
    chosen_signatures = [signatures[i] for i in idx]
    implementations = []
    scores = []
    for signature in chosen_signatures:
      cluster = self._clusters[signature]
      implementations.append(cluster.sample_program())
      scores.append(cluster.score)

    indices = np.argsort(scores)
    sorted_implementations = [implementations[i] for i in indices]
    version_generated = len(sorted_implementations) + 1
    return self._generate_prompt(sorted_implementations), version_generated

  def _generate_prompt(
      self,
      implementations: Sequence[code_manipulation.Function]) -> str:
    """Creates a prompt containing a sequence of function `implementations`."""
    # import pdb; pdb.set_trace()
    
    implementations = copy.deepcopy(implementations)  # We will mutate these.

    # Format the names and docstrings of functions to be included in the prompt.
    versioned_functions: list[code_manipulation.Function] = []
    for i, implementation in enumerate(implementations):
      new_function_name = f'{self._function_to_evolve}_v{i}'
      implementation.name = new_function_name
      # Update the docstring for all subsequent functions after `_v0`.
      if i >= 1:
        implementation.docstring = (
            f'Improved version of `{self._function_to_evolve}_v{i - 1}`.')
      # If the function is recursive, replace calls to itself with its new name.
      implementation = code_manipulation.rename_function_calls(
          str(implementation), self._function_to_evolve, new_function_name)
      versioned_functions.append(
          code_manipulation.text_to_function(implementation))

    # Create the header of the function to be generated by the LLM.
    next_version = len(implementations)
    new_function_name = f'{self._function_to_evolve}_v{next_version}'
    header = dataclasses.replace(
        implementations[-1],
        name=new_function_name,
        body='',
        docstring=('Improved version of '
                   f'`{self._function_to_evolve}_v{next_version - 1}`.'),
    )
    versioned_functions.append(header)

    # Replace functions in the template with the list constructed here.
    prompt = dataclasses.replace(self._template, functions=versioned_functions)
    return str(prompt)


class Cluster:
  """A cluster of programs on the same island and with the same Signature."""

  def __init__(self, score: float, implementation: code_manipulation.Function):
    self._score = score
    self._programs: list[code_manipulation.Function] = [implementation]
    self._lengths: list[int] = [len(str(implementation))]

  @property
  def score(self) -> float:
    """Reduced score of the signature that this cluster represents."""
    return self._score

  def register_program(self, program: code_manipulation.Function) -> None:
    """Adds `program` to the cluster."""
    self._programs.append(program)
    self._lengths.append(len(str(program)))

  def sample_program(self) -> code_manipulation.Function:
    """Samples a program, giving higher probability to shorther programs."""
    normalized_lengths = (np.array(self._lengths) - min(self._lengths)) / (
        max(self._lengths) + 1e-6)
    # print("Softmax in sample_program")
    probabilities = _softmax(-normalized_lengths, temperature=1.0)
    return np.random.choice(self._programs, p=probabilities)

if __name__ == "__main__":
  # test the program db backup
  from funsearch import config
  conf = config.Config(num_evaluators=1)
  database = ProgramsDatabase(conf.programs_database, None, "", identifier="")
  file_name = conf.programs_database.backup_folder + "/program_db_heuristic_1746573950_0.pickle"
  with open(file_name, "rb") as f:
    database.load(f)
  import pdb; pdb.set_trace()
  print(database._best_score_per_island)
  print(database._best_program_per_island)
  print(database._best_scores_per_test_per_island)