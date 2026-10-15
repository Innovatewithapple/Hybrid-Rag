from clean_pdf import Load_PDF, Save_Chunks, Load_Chunks
from src.models import load_models, Call_LLM
from clean_pdf import PDF_Chunking
from src.document_summary import Generate_Document_Summary,Incoming_quesiton_Embeddings,Overview_quesiton_Embeddings,overview_questions
from Prompt.legel_prompt import prompt

from src.reranking import Reranking
from src.embeddings import NomicEmbeddings

from pathlib import Path
import faiss
import numpy as np

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
document_summary = Generate_Document_Summary(chunks)

print("\nDOCUMENT OVERVIEW\n")
print(document_summary)

overveiw = Overview_quesiton_Embeddings(embeddings=embeddings)
overveiw = np.array(overveiw)

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


def Routing(final_score):
    if final_score >= 0.80:
        print("Overview!")
        return "Overveiw"
    else:
        print("Retrieving!")
        return "Retrieving"

def Final_Answer(Quesiton):
    incoming = Incoming_quesiton_Embeddings(embeddings=embeddings,question=Quesiton)
    incoming = np.array(incoming)
    scores = overveiw @ incoming
    best_index = scores.argmax()
    final_score = scores[best_index]

    print("Best match:", overview_questions[best_index])
    print("Similarity:", final_score)
    if Routing(final_score) == "Overview":
        return document_summary
    else:
        rag_chain = (
        {
            "context":ensamble_retrieval,
            "question":RunnablePassthrough()
        }
        | reranker
        | prompt
        | RunnableLambda(lambda prompt_value: Call_LLM(prompt_value.to_messages()))
        )

        return rag_chain.invoke(Quesiton)


Answer = Final_Answer("what is article 10 about?")
print(f'\n\n Answer: {Answer}')