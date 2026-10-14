from langchain_core.embeddings import Embeddings
from src.models import load_models


class NomicEmbeddings(Embeddings):
    def __init__(self,encoder_model):
        self.encoder_model = encoder_model

    def embed_documents(self, texts:list[str]) -> list[list[float]]:
        embeddings = self.encoder_model.encode([f'search_document: {text}' for text in texts],normalize_embeddings=True,convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, text:str) -> list[float]:
        embedding = self.encoder_model.encode(f'search_query: {text}',normalize_embeddings=True,convert_to_numpy=True)
        return embedding.tolist()

    