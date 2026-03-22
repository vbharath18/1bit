import os
import time
import warnings

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message="You don't have a GPU available to load the model*")

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


def load_model(model_id):
    """Load tokenizer and model from Hugging Face."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cpu",
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    return tokenizer, model


def generate_response(tokenizer, model, user_message, max_new_tokens=500):
    """Generate a response and return it with performance metrics."""
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": user_message},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    chat_input = tokenizer(prompt, return_tensors="pt").to(model.device)

    num_input_tokens = chat_input["input_ids"].shape[-1]

    with torch.inference_mode():
        start_time = time.time()
        chat_outputs = model.generate(**chat_input, max_new_tokens=max_new_tokens)
        inference_time = time.time() - start_time

    num_output_tokens = chat_outputs[0].shape[0] - num_input_tokens
    response = tokenizer.decode(
        chat_outputs[0][num_input_tokens:], skip_special_tokens=True
    )

    return response.strip(), {
        "inference_time": inference_time,
        "input_tokens": num_input_tokens,
        "output_tokens": num_output_tokens,
        "tokens_per_second": num_output_tokens / inference_time if inference_time > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Chat with BitNet-compatible models.")
    parser.add_argument("--user", type=str, default="How are you?", help="User message to send to the assistant.")
    parser.add_argument("--max-tokens", type=int, default=500, help="Maximum number of new tokens to generate.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Model to use. Shorthand: {', '.join(SUPPORTED_MODELS.keys())}. "
                             "Or pass a full HuggingFace model ID.")
    parser.add_argument("--list-models", action="store_true", help="List all supported model shorthands and exit.")
    args = parser.parse_args()

    if args.list_models:
        print("Supported models:")
        for shorthand, model_id in SUPPORTED_MODELS.items():
            print(f"  {shorthand:16s} -> {model_id}")
        return

    model_id = SUPPORTED_MODELS.get(args.model, args.model)
    print(f"Loading model: {model_id}")

    try:
        tokenizer, model = load_model(model_id)
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        return

    try:
        response, metrics = generate_response(tokenizer, model, args.user, args.max_tokens)
        print("\nAssistant Response:\n", response)
        print(f"\nInference time: {metrics['inference_time']:.2f}s")
        print(f"Input tokens: {metrics['input_tokens']}")
        print(f"Output tokens: {metrics['output_tokens']}")
        print(f"Generation speed: {metrics['tokens_per_second']:.2f} tokens/second")
    except Exception as e:
        print(f"Error during generation: {e}")


if __name__ == "__main__":
    main()
