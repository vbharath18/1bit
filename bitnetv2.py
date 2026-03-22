import os
import time
import argparse
import json

# Supported BitNet-compatible models
SUPPORTED_MODELS = {
    # Microsoft official
    "bitnet-2b": "microsoft/bitnet-b1.58-2B-4T",
    # Falcon3 1.58-bit quantized
    "falcon3-1b": "tiiuae/Falcon3-1B-Instruct-1.58bit",
    "falcon3-3b": "tiiuae/Falcon3-3B-Instruct-1.58bit",
    "falcon3-7b": "tiiuae/Falcon3-7B-Instruct-1.58bit",
    "falcon3-10b": "tiiuae/Falcon3-10B-Instruct-1.58bit",
    # Llama3 1.58-bit quantized
    "llama3-8b": "HF1BitLLM/Llama3-8B-1.58-100B-tokens",
    # Community
    "bitnet-3b": "1bitLLM/bitnet_b1_58-3B",
}

DEFAULT_MODEL = "bitnet-2b"


def setup_cpu_optimizations(num_threads=None):
    """Configure CPU-specific performance settings."""
    import torch
    import torch.backends.cpu

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_CPU_ALLOC_CONF"] = "max_split_size_mb:64"

    torch.backends.cpu.optimize = True
    torch.set_num_threads(num_threads or os.cpu_count())


def load_model(model_id):
    """Load tokenizer and model with BitNet-optimized configuration."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitNetConfig

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
    import torch

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


def generate_via_server(server_url, user_message, model_name, max_new_tokens=500):
    """Generate a response using a bitnet.cpp OpenAI-compatible server.

    Requires a running bitnet.cpp inference server (run_inference_server.py).
    Uses urllib so no extra dependencies are needed beyond the standard library.
    """
    import urllib.request

    url = f"{server_url}/v1/chat/completions"
    payload = json.dumps({
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": user_message},
        ],
        "max_tokens": max_new_tokens,
        "temperature": 0.7,
        "top_p": 0.95,
    }).encode()

    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})

    start_time = time.time()
    resp = urllib.request.urlopen(req)
    total_time = time.time() - start_time

    data = json.loads(resp.read().decode())
    choice = data["choices"][0]
    usage = data.get("usage", {})

    output_tokens = usage.get("completion_tokens", 0)
    return choice["message"]["content"], {
        "total_time": total_time,
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": output_tokens,
        "tokens_per_second": output_tokens / total_time if total_time > 0 else 0,
    }


def print_metrics(metrics, server_mode=False):
    """Print performance metrics."""
    print("\nPerformance Metrics:")
    if not server_mode:
        print(f"  Tokenization time: {metrics['tokenization_time']:.2f}s")
        print(f"  Generation time:   {metrics['generation_time']:.2f}s")
        print(f"  Decoding time:     {metrics['decoding_time']:.2f}s")
    print(f"  Total time:        {metrics['total_time']:.2f}s")
    print(f"  Input tokens:      {metrics['input_tokens']}")
    print(f"  Output tokens:     {metrics['output_tokens']}")
    print(f"  Generation speed:  {metrics['tokens_per_second']:.2f} tokens/second")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark BitNet-compatible model inference (HuggingFace or bitnet.cpp server).",
    )
    parser.add_argument("--user", type=str, default="Tell me about the latest advancements in AI.",
                        help="User message to send to the assistant.")
    parser.add_argument("--max-tokens", type=int, default=500, help="Maximum number of new tokens to generate.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Model to use. Shorthand: {', '.join(SUPPORTED_MODELS.keys())}. "
                             "Or pass a full HuggingFace model ID.")
    parser.add_argument("--list-models", action="store_true", help="List all supported model shorthands and exit.")
    parser.add_argument("--threads", type=int, default=None, help="Number of CPU threads (default: all cores).")

    # bitnet.cpp server mode
    parser.add_argument("--server", type=str, default=None, metavar="URL",
                        help="Use a bitnet.cpp OpenAI-compatible server instead of local HuggingFace inference. "
                             "Example: http://127.0.0.1:8080")
    args = parser.parse_args()

    if args.list_models:
        print("Supported models:")
        for shorthand, model_id in SUPPORTED_MODELS.items():
            print(f"  {shorthand:16s} -> {model_id}")
        return

    model_id = SUPPORTED_MODELS.get(args.model, args.model)

    # Server mode: use bitnet.cpp OpenAI-compatible API
    if args.server:
        print(f"Using bitnet.cpp server at {args.server} (model: {model_id})")
        try:
            response, metrics = generate_via_server(args.server, args.user, model_id, args.max_tokens)
        except Exception as e:
            print(f"Error communicating with server: {e}")
            return
        print("\nAssistant Response:", response)
        print_metrics(metrics, server_mode=True)
        return

    # Local HuggingFace inference mode
    print(f"Loading model: {model_id}")
    setup_cpu_optimizations(args.threads)

    try:
        tokenizer, model = load_model(model_id)
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        return

    try:
        response, metrics = generate_response(tokenizer, model, args.user, args.max_tokens)
    except Exception as e:
        print(f"Error during generation: {e}")
        return

    print("\nAssistant Response:", response)
    print_metrics(metrics)


if __name__ == "__main__":
    main()
