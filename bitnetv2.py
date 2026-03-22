import os
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CPU_ALLOC_CONF"] = "max_split_size_mb:64"

import argparse

import torch
import torch.backends.cpu
from transformers import AutoModelForCausalLM, AutoTokenizer, BitNetConfig

MODEL_ID = "microsoft/bitnet-b1.58-2B-4T"


def setup_cpu_optimizations():
    """Configure CPU-specific performance settings."""
    torch.backends.cpu.optimize = True
    torch.set_num_threads(os.cpu_count())


def load_model(model_id):
    """Load tokenizer and model with BitNet-optimized configuration."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = BitNetConfig.from_pretrained(model_id)
    config.low_cpu_mem_usage = True
    config.pad_token_id = tokenizer.pad_token_id
    config.use_cache = True
    config.pretraining_tp = 1
    config.max_position_embeddings = 2048

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        torch_dtype=torch.bfloat16,
    )

    try:
        model = torch.compile(model)
    except Exception as e:
        print(f"torch.compile not available or failed: {e}")

    model = model.cpu()
    model.eval()

    return tokenizer, model


def generate_response(tokenizer, model, user_message, max_new_tokens=500):
    """Generate a response and return it with detailed performance metrics."""
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": user_message},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    with torch.inference_mode():
        tokenization_start = time.time()
        chat_input = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )
        tokenization_time = time.time() - tokenization_start

        generation_start = time.time()
        chat_outputs = model.generate(
            **chat_input,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            use_cache=True,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
            length_penalty=1.0,
            no_repeat_ngram_size=3,
        )
        generation_time = time.time() - generation_start

        decoding_start = time.time()
        response = tokenizer.decode(
            chat_outputs[0][chat_input["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        )
        decoding_time = time.time() - decoding_start

    num_input_tokens = chat_input["input_ids"].shape[1]
    num_output_tokens = chat_outputs[0].shape[0] - num_input_tokens
    tokens_per_second = num_output_tokens / generation_time if generation_time > 0 else 0

    return response, {
        "tokenization_time": tokenization_time,
        "generation_time": generation_time,
        "decoding_time": decoding_time,
        "total_time": tokenization_time + generation_time + decoding_time,
        "input_tokens": num_input_tokens,
        "output_tokens": num_output_tokens,
        "tokens_per_second": tokens_per_second,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark BitNet model inference.")
    parser.add_argument("--user", type=str, default="Tell me about the latest advancements in AI.",
                        help="User message to send to the assistant.")
    parser.add_argument("--max-tokens", type=int, default=500, help="Maximum number of new tokens to generate.")
    args = parser.parse_args()

    setup_cpu_optimizations()

    try:
        tokenizer, model = load_model(MODEL_ID)
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        return

    try:
        response, metrics = generate_response(tokenizer, model, args.user, args.max_tokens)
    except Exception as e:
        print(f"Error during generation: {e}")
        return

    print("\nAssistant Response:", response)
    print("\nPerformance Metrics:")
    print(f"  Tokenization time: {metrics['tokenization_time']:.2f}s")
    print(f"  Generation time:   {metrics['generation_time']:.2f}s")
    print(f"  Decoding time:     {metrics['decoding_time']:.2f}s")
    print(f"  Total time:        {metrics['total_time']:.2f}s")
    print(f"  Input tokens:      {metrics['input_tokens']}")
    print(f"  Output tokens:     {metrics['output_tokens']}")
    print(f"  Generation speed:  {metrics['tokens_per_second']:.2f} tokens/second")


if __name__ == "__main__":
    main()
