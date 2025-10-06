import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import transformers
import logging
import time
import os

HOSTNAME = os.uname().nodename

# ANSI color codes
YELLOW = "\033[93m"
CYAN = "\033[1;96m"   # Bold + Cyan
RESET = "\033[0m"

class CustomLLM(torch.nn.Module):
    def __init__(self, samples_per_prompt: int, 
                 device=None, 
                  model_name="bigcode/starcoder2-3b" if HOSTNAME == "matteogu" else "bigcode/starcoder2-15b-instruct-v0.1", 
                 # model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
                 # Qwen2.5-Coder-7B-Instruct does not have a padding token.
                 # Asking to pad but the tokenizer does not have a padding token. 
                 # Please select a token to use as `pad_token` `(tokenizer.pad_token = tokenizer.eos_token e.g.)` or add a new pad token via `tokenizer.add_special_tokens({'pad_token': '[PAD]'})`.
                 quantization_config=None, 
                 log_path=None):
        super().__init__()
        self._samples_per_prompt = samples_per_prompt
        self.prompt_count = 0
        self.log_path = log_path
        self.device = device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.device_map = {"": self.device}
        if quantization_config is not None:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                              trust_remote_code=True, 
                                                              quantization_config=quantization_config,
                                                              low_cpu_mem_usage=True,
                                                              device_map=self.device_map)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                              trust_remote_code=True,
                                                              low_cpu_mem_usage=True,
                                                              device_map=self.device_map)
        print("Compiling model...")
        start_time = time.time()
        # https://huggingface.co/docs/transformers/en/perf_torch_compile
        try:
            print("Compiling model with reduce-overhead...")
            self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=True) # "max-autotune", "reduce-overhead"
        except Exception as e:
            print("Compiling model with max-autotune...")
            self.model = torch.compile(self.model, mode="max-autotune", fullgraph=False) # "max-autotune", "reduce-overhead"
        end_time = time.time()
        print(f"Model compiled in {end_time - start_time:.2f} seconds")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if not("Qwen" in model_name): 
            self.tokenizer.pad_token = "[PAD]" 
            self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids("[PAD]")
        else:
            print(model_name)

        # self.tokenizer.pad_token = self.tokenizer.eos_token
        self.system_prompt = ("You are an exceptionally intelligent PYTHON coding assistant " 
                            "that consistently delivers accurate and reliable responses to user instructions. The response should be PYTHON code with this format"
                            "```python"
                            "def heuristic_vX(obs: np.ndarray):"
                            "#FILL"
                            "return action"
                            "```"
                            "### Instruction\n")
        
        self.response_prompt = "\n### Response\n"
        
    def forward(self, input_ids):
        return self.model(input_ids)
    
    def draw_samples(self, prompt: str, max_length=800):
        # import pdb; pdb.set_trace()
        logging.info(f"Drawing samples with prompt: {prompt.split('@funsearch.run')[1]}")

        # print("Model is on device:", self.model.device)
        prompt = self.system_prompt + prompt + self.response_prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt', padding=True).to(self.model.device)
        
        
        samples = []
        start_time = time.time()
        # for _ in range(self._samples_per_prompt):
        logging.info(f'Generating sample {self._samples_per_prompt}...')
        
        with torch.no_grad():
            # Batch generate the outputs
            outputs = self.model.generate(
                input_ids, 
                max_length=max_length, 
                num_return_sequences=self._samples_per_prompt, 
                no_repeat_ngram_size=None, 
                do_sample=True, 
                top_k=40, 
                top_p=0.95, 
                repetition_penalty=1.1,
                temperature=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True
            )

        # Batch decode the outputs
        responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # import pdb; pdb.set_trace()

        # mask = [True if r.startswith(prompt) else False for r in responses]


        # filtered = [r for r in responses if (r.startswith(prompt) and len(r[len(prompt):].strip()) > 0)]

        samples = []
        for response in responses:
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
                print(response)
            if len(response) == 0:
                # if it is not outputing values we skip the eval
                # we still log
                pass 
            else:
                samples.append(response)
            self._log(prompt, response, self.prompt_count)
            self.prompt_count += 1
        end_time = time.time()
        logging.info(f'[Not empty {len(samples)/len(responses):.2f}] {CYAN}[GPU {self.model.device}] Samples drawn in {end_time - start_time} seconds...{RESET}')

        return samples

    def _log(self, prompt: str, response: str, index: int):
        """
        Log the prompt and response to a file.
        TODO: this is not efficient. The log should be written in a separate process.
        It is fast for now. It is not a bottleneck.
        """
        start_time = time.time()
        if self.log_path is not None:
            with open(self.log_path / f"prompt_{index}.log", "a") as f: # saves the prompt in file
                f.write(prompt)
            with open(self.log_path / f"response_{index}.log", "a") as f:
                f.write(str(response))
        end_time = time.time()
        # logging.info(f'Log written in {end_time - start_time} seconds...')

# Example usage
if __name__ == "__main__":

    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    llm = CustomLLM(samples_per_prompt=10, quantization_config=quantization_config) #  device="cuda:1",
    # prompt = 
    
    # prompt = ("### Instruction\n def sum_first_n(n):\n"
    #           "# fill here\n"
    #           "return out\n"
    #           "### Response")
    prompt = "def fibonacci(n):\n"
    start_time = time.time()
    samples = llm.draw_samples(prompt=prompt)
    end_time = time.time()
    print(f"Time taken to draw samples: {end_time - start_time:.2f} seconds")
    # for i, sample in enumerate(samples, 1):
    #     print(f"Sample {i}: {sample}")

    print('done')