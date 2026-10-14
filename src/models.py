import torch
from sentence_transformers import SentenceTransformer,CrossEncoder
from transformers import AutoTokenizer,AutoModelForCausalLM
from functools import lru_cache
import requests

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
    formatted_messages = [{"role":message.type,"content":message.content} for message in messages]
    response = requests.post(
    f"{PUBLIC_URL}/v1/chat/completions",
    json={
        "messages": formatted_messages,
        "temperature": 0.2,
        "max_tokens": 500
    },
    timeout=180
    )

    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

