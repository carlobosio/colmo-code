import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ANSI color codes
RESET = "\033[0m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
MAGENTA = "\033[35m"

def print_color(msg, color):
    print(f"{color}{msg}{RESET}")


# Step 1: Quantization config
start = time.perf_counter()
print_color("🔧 Setting quantization config...", CYAN)

# model_name = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
# quant_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.float16,
# )

model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
quant_config = None
print_color(f"⏱️ Quantization config time: {time.perf_counter()-start:.2f}s", YELLOW)

# Step 2: Load tokenizer
start = time.perf_counter()
print_color("📦 Loading tokenizer...", CYAN)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
print_color(f"⏱️ Tokenizer load time: {time.perf_counter()-start:.2f}s", YELLOW)

# Step 3: Device and memory info
start = time.perf_counter()
print_color("🧠 Inferring device map across GPUs...", CYAN)
device_count = torch.cuda.device_count()
max_mem = {i: "20GiB" for i in range(device_count)}
print_color(f"🖥️ Found {device_count} CUDA devices: {max_mem}", MAGENTA)
print_color(f"⏱️ Device info time: {time.perf_counter()-start:.2f}s", YELLOW)

# Step 4: Load model
start = time.perf_counter()
print_color("�� Loading model weights with device map and quantization...", CYAN)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    quantization_config=quant_config,
    device_map="auto",
    max_memory=max_mem,
    low_cpu_mem_usage=True,
)
print_color("✅ Model loaded successfully.", GREEN)
print_color(f"⏱️ Model load time: {time.perf_counter()-start:.2f}s", YELLOW)

# Step 5: Print device map
device_map = getattr(model, 'hf_device_map', None)
if device_map:
    print_color("✅ Device map created:", GREEN)
    for k, v in device_map.items():
        print_color(f"  - {k} → {v}", GREEN)
else:
    print_color("ℹ️ Device map not available as an attribute.", MAGENTA)

# Step 6: Set to eval mode for inference
print_color("🛠️ Setting model to inference mode...", CYAN)
model.eval()

max_new_tokens = 200 # it is painful

# Step 7: Test single inference
start = time.perf_counter()
print_color(f"🚀 Running test inference... max_new_tokens={max_new_tokens}", CYAN)
input_text = "Write a Python function to sort a list of integers."
inputs = tokenizer(input_text, return_tensors="pt").to("cuda:0")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
print_color(f"⏱️ Inference time: {time.perf_counter()-start:.2f}s", YELLOW)
print_color("\n📄 Output:", CYAN)
#print(tokenizer.decode(outputs, skip_special_tokens=True))
# Example: decoding batch generation outputs correctly
#with torch.no_grad():
#    outputs = model.generate(**inputs, max_new_tokens=32)

for i, output_ids in enumerate(outputs):
    text = tokenizer.decode(output_ids, skip_special_tokens=True)
    print_color(f"Output {i+1}:\n{text}\n", CYAN)

# Step 8: Batch inference loop (powers of 2)
print_color("\n🔁 Profiling inference with increasing batch size (powers of 2):", CYAN)
max_batch = 4096
test_text = "Sort a list of integers in Python."
sequence = [test_text] * max_batch  # Prepare enough samples

for batch_exp in range(0, int(max_batch).bit_length()):
    batch_size = 2 ** batch_exp
    batch_texts = sequence[:batch_size]
    
    inputs = tokenizer(batch_texts, return_tensors="pt", padding=True).to("cuda:0")
    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

    with open(f"model_output_{batch_size:<3}.txt", "w", encoding="utf-8") as f:
        for i, output_ids in enumerate(outputs):
            text = tokenizer.decode(output_ids, skip_special_tokens=True)
            f.write(f"Output {i+1}:\n{text}\n\n")
    print_color(f"Model output saved to model_output_{batch_size:<3}.txt", GREEN)
    
    elapsed = time.perf_counter() - start
    print_color(f"🔢 Batch size {len(batch_texts)} {batch_size:<3}: {elapsed:.3f}s", MAGENTA)