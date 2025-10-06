import json
import logging
import os
import pathlib
import pickle
import time

import click
from dotenv import load_dotenv

from funsearch import config, core, sandbox, programs_database, code_manipulation, evaluator

LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()
logging.basicConfig(level=LOGLEVEL)

# Define your program template here
PROGRAM_TEMPLATE = """def heuristic_v1(state: np.ndarray) -> float:
  # Parameters to be optimized
  k1 = 1.0  # gain for angle
  k2 = 2.5  # gain for angular velocity

  # State variables
  theta = state[0]     # angle
  theta_dot = state[1] # angular velocity

  # Linear control law
  return k1 * theta + k2 * theta_dot"""

def get_all_subclasses(cls):
    all_subclasses = []
    for subclass in cls.__subclasses__():
        all_subclasses.append(subclass)
        all_subclasses.extend(get_all_subclasses(subclass))
    return all_subclasses

SANDBOX_TYPES = get_all_subclasses(sandbox.DummySandbox) + [sandbox.DummySandbox]
SANDBOX_NAMES = [c.__name__ for c in SANDBOX_TYPES]

def parse_input(filename_or_data: str):
    if len(filename_or_data) == 0:
        raise Exception("No input data specified")
    p = pathlib.Path(filename_or_data)
    if p.exists():
        if p.name.endswith(".json"):
            return json.load(open(filename_or_data, "r"))
        if p.name.endswith(".pickle"):
            return pickle.load(open(filename_or_data, "rb"))
        raise Exception("Unknown file format or filename")
    if "," not in filename_or_data:
        data = [filename_or_data]
    else:
        data = filename_or_data.split(",")
    if data[0].isnumeric():
        f = int if data[0].isdecimal() else float
        data = [f(v) for v in data]
    return data

@click.command()
@click.argument("spec_file", type=click.File("r"))
@click.argument('inputs')
@click.option('--output_path', default="./data/", type=click.Path(file_okay=False), help='path for logs and data')
@click.option('--sandbox_type', default="ContainerSandbox", type=click.Choice(SANDBOX_NAMES), help='Sandbox type')
@click.option('--optimization_budget', default=600, type=click.INT, help='Number of evaluations for float optimization')
@click.option('--significant_digits', default=4, type=click.INT, help='Number of significant digits for float values')
def optimize(spec_file, inputs, output_path, sandbox_type, optimization_budget, significant_digits):
    """Optimize floating point parameters in a program:

    SPEC_FILE: Python module that defines the evaluation metric (run function)
    INPUTS: Input data for evaluation (file or comma-separated values)
    """
    load_dotenv()

    # Setup logging
    timestamp = str(int(time.time()))
    log_path = pathlib.Path(output_path) / timestamp
    if not log_path.exists():
        log_path.mkdir(parents=True)
        logging.info(f"Writing logs to {log_path}")

    # Parse specification
    specification = spec_file.read()
    
    # Extract function names and create template
    function_to_evolve, function_to_run = core._extract_function_names(specification)
    template = code_manipulation.text_to_program(specification) # template contains a parsed version of the spec file

    # Setup database and sandbox
    conf = config.Config(num_evaluators=1)
    database = programs_database.ProgramsDatabase(
        conf.programs_database, template, function_to_evolve, 
        identifier=timestamp, log_path=log_path
    )

    inputs = parse_input(inputs)
    sandbox_class = next(c for c in SANDBOX_TYPES if c.__name__ == sandbox_type)
    # Create evaluator with float optimization enabled
    evaluator_instance = evaluator.Evaluator(
        database=database,
        sbox=sandbox_class(base_path=log_path),
        template=template,
        function_to_evolve=function_to_evolve,
        function_to_run=function_to_run,
        inputs=inputs,
        optimize_floats=True,
        optimization_budget=optimization_budget,
        significant_digits=significant_digits
    )

    # Run optimization
    try:
        print("trying analyse")
        evaluator_instance.analyse(
            sample=PROGRAM_TEMPLATE,
            island_id=None,
            version_generated=None
        )
        
        # Get and display results
        best_programs = database.get_best_programs_per_island()
        if best_programs:
            best_program, best_score = best_programs[0]
            print("\nOptimization Results:")
            print(f"Best score achieved: {best_score}")
            print("\nOptimized program:")
            print(best_program)
        else:
            print("No valid programs found")
            
    except Exception as e:
        print("Throwing exception")
        print(f"Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == '__main__':
    optimize() 