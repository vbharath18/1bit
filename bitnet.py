import os
import warnings
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

# Suppress Hugging Face and PyTorch CPU/GPU warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message="You don't have a GPU available to load the model*")

def main():
    parser = argparse.ArgumentParser(description="Chat with BitNet model.")
    parser.add_argument('--user', type=str, default="How are you?", help="User message to send to the assistant.")
    args = parser.parse_args()

    model_id = "microsoft/bitnet-b1.58-2B-4T"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        # Use 8-bit quantization for CPU with bitsandbytes if available
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="cpu",
            low_cpu_mem_usage=True,
            # torch_dtype=torch.float32  # Use float32 for CPU for numerical stability
            torch_dtype=torch.bfloat16
            # load_in_8bit=True  # Uncomment if bitsandbytes and model support it
        )
        # Note: For true 1-bit quantization, a custom model or library is required.
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        return

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": args.user},
    ]
    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        chat_input = tokenizer(prompt, return_tensors="pt").to(model.device)
        chat_outputs = model.generate(**chat_input, max_new_tokens=30)  # Reduce tokens for lower memory
        response = tokenizer.decode(chat_outputs[0][chat_input['input_ids'].shape[-1]:], skip_special_tokens=True)
        print("\nAssistant Response:\n", response.strip())
    except Exception as e:
        print(f"Error during generation: {e}")

if __name__ == "__main__":
    main()
