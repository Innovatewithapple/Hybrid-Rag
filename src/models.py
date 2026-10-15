import torch
from sentence_transformers import SentenceTransformer,CrossEncoder
from transformers import AutoTokenizer,AutoModelForCausalLM
from functools import lru_cache
import requests
import time

PUBLIC_URL = "https://absentee-mulled-stadium.ngrok-free.dev"

def get_device():
    if torch.cuda.is_available():
        return "cuda:0"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

@lru_cache(maxsize=1)
def load_models():
    device = get_device()
    print(f"Available Device: {device}")

    encoder_tokenizer = AutoTokenizer.from_pretrained('nomic-ai/nomic-embed-text-v1.5',trust_remote_code=True)
    encoder_model = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5',trust_remote_code=True,device=device)

    reranker_model = CrossEncoder('BAAI/bge-reranker-large',device=device)

    return encoder_tokenizer,encoder_model,reranker_model


def Call_LLM(messages):
    formatted_messages = [
        {
            "role": {
                "system": "system",
                "human": "user",
                "ai": "assistant",
            }[message.type],
            "content": message.content,
        }
        for message in messages
    ]

    payload = {
        "messages": formatted_messages,
        "temperature": 0.2,
        "max_tokens": 500,
        "cache_prompt": True,
    }

    # Debug: inspect the actual prompt being sent
    print("\n--- LLM REQUEST DEBUG ---")
    for message in formatted_messages:
        print(f"\nRole: {message['role']}")
        print(f"Content:\n{message['content'][:1000]}")
    print("--- END DEBUG ---\n")
    start = time.perf_counter()

    response = requests.post(
        f"{PUBLIC_URL}/v1/chat/completions",
        json=payload,
        timeout=180,
    )
    print(f"LLM request time: {time.perf_counter() - start:.2f}s")

    response.raise_for_status()
    result = response.json()["choices"][0]["message"]["content"]

    # print("\n--- LLM RESPONSE DEBUG ---")
    # print(result[:1000])
    # print("--- END RESPONSE DEBUG ---\n")

    return result


