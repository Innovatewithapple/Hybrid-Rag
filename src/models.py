import torch
from sentence_transformers import SentenceTransformer,CrossEncoder
from transformers import AutoTokenizer,AutoModelForCausalLM
from functools import lru_cache
import requests
import time
import os

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

    encoder_model = SentenceTransformer(
        "nomic-ai/nomic-embed-text-v1.5",
        device=device,
        trust_remote_code=True
    )

    reranker_model = CrossEncoder(
        "BAAI/bge-reranker-large",
        device=device,
    )
    if str(device).startswith("cuda"):
        reranker_model.model.half()

    return None, encoder_model, reranker_model



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
    }

    provider = os.getenv("LLM_PROVIDER", "ngrok")

    if provider == "gemini":
        url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
        headers = {"Authorization": f"Bearer {os.environ['GEMINI_API_KEY']}"}
        payload["model"] = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    else:
        url = f"{PUBLIC_URL}/v1/chat/completions"
        headers = {}
        payload["cache_prompt"] = True

    start = time.perf_counter()

    response = requests.post(
        url,
        json=payload,
        headers=headers,
        timeout=180,
    )
    print(f"LLM request time: {time.perf_counter() - start:.2f}s")

    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


