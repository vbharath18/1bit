import os
# Set tokenizer parallelism before importing tokenizer-related modules
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.backends.cpu

# Enable memory alignment
os.environ['PYTORCH_CPU_ALLOC_CONF'] = 'max_split_size_mb:64'

# Enable optimizations
torch.backends.cpu.optimize = True
torch.set_num_threads(os.cpu_count())

model_id = "microsoft/bitnet-b1.58-2B-4T"

# Load tokenizer and model with optimizations
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Set up padding token (using EOS token as pad token is a common practice)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    pad_token_id=tokenizer.pad_token_id
)

# Move model to CPU explicitly and optimize
model = model.cpu()
model.eval()  # Set to inference mode

# Batch size for more efficient processing
BATCH_SIZE = 1  # Adjust based on your CPU memory

# Apply the chat template
messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "Tell me about the latest advancements in AI."},
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

import time

# Use efficient batching
with torch.inference_mode():  # More efficient than no_grad for inference
    # Measure tokenization time
    tokenization_start = time.time()
    chat_input = tokenizer(
        prompt, 
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048  # Adjust based on your needs
    )
    tokenization_time = time.time() - tokenization_start
    
    # Measure generation time
    generation_start = time.time()
    chat_outputs = model.generate(
        **chat_input,
        max_new_tokens=500,
        num_return_sequences=1,
        do_sample=True,  # Enable sampling
        temperature=0.6,  # Control randomness (lower = more focused)
        top_p=0.9,       # Nucleus sampling
        use_cache=True,  # Enable KV-cache
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    generation_time = time.time() - generation_start
    
    # Measure decoding time
    decoding_start = time.time()
    response = tokenizer.decode(
        chat_outputs[0][chat_input['input_ids'].shape[-1]:],
        skip_special_tokens=True
    )
    decoding_time = time.time() - decoding_start

# Calculate total time and tokens per second
total_time = tokenization_time + generation_time + decoding_time
num_input_tokens = chat_input['input_ids'].shape[1]
num_output_tokens = chat_outputs[0].shape[0] - num_input_tokens
tokens_per_second = num_output_tokens / generation_time

print("\nAssistant Response:", response)
print("\nPerformance Metrics:")
print(f"Tokenization time: {tokenization_time:.2f}s")
print(f"Generation time: {generation_time:.2f}s")
print(f"Decoding time: {decoding_time:.2f}s")
print(f"Total time: {total_time:.2f}s")
print(f"Input tokens: {num_input_tokens}")
print(f"Output tokens: {num_output_tokens}")
print(f"Generation speed: {tokens_per_second:.2f} tokens/second")
