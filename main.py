from clean_pdf import Load_PDF, Save_Chunks, Load_Chunks
from src.models import load_models, Call_LLM
from clean_pdf import PDF_Chunking
from Prompt.legel_prompt import prompt

from src.reranking import Reranking
from src.embeddings import NomicEmbeddings

from pathlib import Path
import faiss

from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

encoder_tokenizer, encoder_model, reranker = load_models()
embeddings = NomicEmbeddings(encoder_model=encoder_model)

path='/Users/himanshuvyas/Downloads/HybridRag/regulation eu 2016679 of the european parliament and-DOC_1.pdf'
CHUNKS_PATH = Path("data/semantic_chunks.json")

# Load saved chunks first; skip model initialization if available.
chunks = Load_Chunks(CHUNKS_PATH)

if chunks is None:
    docs = Load_PDF(path)

    chunks = PDF_Chunking(docs, encoder_model)
    Save_Chunks(chunks, CHUNKS_PATH)

bm25_retriever = BM25Retriever.from_documents(documents=chunks)
bm25_retriever.k = 5

vector_stores = FAISS.from_documents(documents=chunks,embedding=embeddings)
faiss_retriever = vector_stores.as_retriever(search_type='similarity',search_kwargs={'k':5})
ensamble_retrieval = EnsembleRetriever(retrievers=[bm25_retriever,faiss_retriever],weights=[0.5,0.5])
reranker = RunnableLambda(Reranking(model=reranker,top_k=5))
parser = StrOutputParser()


rag_chain = (
    {
        "context":ensamble_retrieval,
        "question":RunnablePassthrough()
    }
    | reranker
    | prompt
    | RunnableLambda(lambda prompt_value: Call_LLM(prompt_value.to_messages()))
)



answer = rag_chain.invoke('What are the main objectives of the GDPR?')
print(f"Answer: {answer}")