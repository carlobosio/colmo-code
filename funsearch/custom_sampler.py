import torch
import logging
from pathlib import Path
# import torch.multiprocessing as mp
# from torch.nn.parallel import DistributedDataParallel as DDP
# import os
from transformers import BitsAndBytesConfig
import numpy as np
# from funsearch import custom_llm  # Import our StarCoder2
from funsearch import custom_llm_pipeline as custom_llm
from concurrent.futures import ThreadPoolExecutor, as_completed

from collections.abc import Collection, Sequence, Callable
from funsearch import evaluator
from funsearch import programs_database
import time

# ANSI color codes
YELLOW = "\033[93m"
CYAN = "\033[1;96m"   # Bold + Cyan
RESET = "\033[0m"

class CustomSampler:
    """Node that samples program continuations and sends them for analysis."""

    def __init__(
        self,
        rank: int,
        seed: int,
        model_name: str = None,
        database: programs_database.ProgramsDatabase = None,
        evaluator_factory: Callable[[], evaluator.Evaluator] = None,
        num_evaluators: int = 1,
        samples_per_prompt = None,
        log_path = None,
        quantization_config = None
    ) -> None:
        self._database = database
        self._evaluator_factory = evaluator_factory
        self._num_evaluators = num_evaluators
        self._evaluators = None
        self._samples_per_prompt = samples_per_prompt
        self._rank = rank
        self.seed = seed
        self.model_name = model_name
        self.device = f"cuda:{self._rank}"
        self.log_path = Path(str(log_path) + f"/gpu_{self._rank}")
        print(f"Log path for gpu {self._rank}: {self.log_path}")
        self.quantization_config = quantization_config
        self._llm = None
        self._num_llm_inferences = 0
        self._num_eval_function_call = 0


    def initialize_llm(self):
        """Initialize the LLM in the subprocess."""
        logging.info(f"################")
        logging.info(f"Initializing LLM")

        # torch.cuda.set_device(self._rank)
        if not self.log_path.exists():
            self.log_path.mkdir(parents=True)
            logging.info(f"Initializing LLM to write in {self.log_path}")

        self._evaluators = [
            self._evaluator_factory() 
            for _ in range(self._num_evaluators)
        ]

        self._llm = custom_llm.CustomLLM(
            samples_per_prompt=self._samples_per_prompt,
            seed=self.seed,
            model_name=self.model_name,
            log_path=self.log_path
        )

    def cleanup(self):
        """Clean up any resources allocated by the sampler."""
        logging.info(f"Cleaning up GPU {self._rank}.")
        torch.distributed.destroy_process_group()

    def sample(self):
        start_time = time.time()
        # logging.info(f"[{start_time}] Sampling on GPU {self._rank}.")
        # logging.info(f"{YELLOW}[{start_time}]{RESET} Sampling on {CYAN}GPU {self._rank}{RESET}.")
        if self._evaluators is None:
            raise RuntimeError("Sampler not initialized. Call initialize_llm() first")

        # import pdb; pdb.set_trace()
        prompt = self._database.get_prompt()

        if not prompt:
            logging.info(f"No prompt from database for gpu {self._rank}.")
            return False
        
        samples = self._llm.draw_samples(prompt.code)
        self._num_llm_inferences += 1

        end_time = time.time()
        logging.info(f"{YELLOW}[{end_time}]{RESET} Sampling on {CYAN}GPU {self._rank}{RESET} took {CYAN}{end_time - start_time:.2f} seconds{RESET}.")

        return {'samples': samples, 'prompt': prompt}

    def evaluate_samples(self, dict_prompt_sample):
        self._num_eval_function_call += 1
        self._database.wandb_log(steps = self._num_eval_function_call) # log best programs on wandb TODO: better if then else

        prompt = dict_prompt_sample['prompt']
        samples = dict_prompt_sample['samples']


        start_time = time.time()

        if self._evaluators is None:
            raise RuntimeError("Sampler not initialized. Call initialize_llm() first")

        def parallel_evaluate(samples, evaluators):
            chunk_size = len(evaluators)
            for i in range(0, len(samples), chunk_size):
                batch = samples[i:i+chunk_size]
                with ThreadPoolExecutor(max_workers=chunk_size) as executor:
                    futures = [
                        executor.submit(
                            evaluator.analyse, sample, prompt.island_id, prompt.version_generated
                        )
                        for evaluator, sample in zip(evaluators, batch)
                    ]
                    for future in as_completed(futures):
                        if future.result():
                            for remaining_future in futures:
                                remaining_future.cancel()
                            return True
            return False

        stop_experiment = parallel_evaluate(samples, self._evaluators)
        if stop_experiment:
            return True

        end_time = time.time()
        logging.info(f"{YELLOW}[{end_time}]{RESET} Eval only on {CYAN}GPU {self._rank}{RESET} took {CYAN}{end_time - start_time:.2f} seconds{RESET}.")

        return False

    def sample_test(self, prompt="def fibonacci(n):"):

        if not prompt:
            logging.info(f"No prompt from database for gpu {self._rank}.")
            return
        
        samples = self._llm.draw_samples(prompt)

        return samples


# Usage
if __name__ == "__main__":
    
    import pickle
    
    # database = programs_database.ProgramsDatabase()
    # evaluators = [evaluator.Evaluator() for _ in range(4)]
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    sampler = CustomSampler(rank=1, quantization_config=quantization_config)
    # samples = sampler.sample_test()
    # for i, sample in enumerate(samples, 1):
    #     print(f"Sample {i}: {sample}")

    for attr_name, attr_value in vars(sampler).items():
        try:
            pickle.dumps(attr_value)
        except Exception as e:
            print(f"Attribute {attr_name} is not picklable: {e}")
