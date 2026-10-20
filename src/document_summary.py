from ast import Call
from email import message
import json
from pathlib import Path
from langchain_core.messages import HumanMessage,SystemMessage
from src.models import Call_LLM

def Generate_Document_Summary(cache_path,chunks,batch_size=5):
    cache_file = Path(cache_path)
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    # Reuse an existing summary
    if cache_file.exists():
        with cache_file.open("r", encoding="utf-8") as file:
            cached_data = json.load(file)

        print("Loaded cached document summary")
        return cached_data["summary"]

    if not chunks:
        raise ValueError("Cannot summarize an empty document.")

    if batch_size < 2:
        raise ValueError("batch_size must be at least 2.")

    system_prompt = """
    You are an expert document summarization assistant.

    Summarize only the information present in the supplied text.
    Preserve the document's purpose, scope, major topics,
    important obligations, rights, exceptions, and relevant
    article or section references.

    Do not invent facts or introduce unsupported information.
    """

    # Stage 1: Summarize batches of original chunks
    partial_summaries = []

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]

        batch_text = "\n\n".join(
            doc.page_content for doc in batch
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""
Summarize this part of the document. Preserve its key
information and important legal details.

DOCUMENT TEXT:
{batch_text}
"""),
        ]

        summary = Call_LLM(messages)

        if not isinstance(summary, str) or not summary.strip():
            raise ValueError(
                f"Empty or invalid summary for chunks "
                f"{start + 1}-{min(start + batch_size, len(chunks))}"
            )

        partial_summaries.append(summary.strip())

        print(
            f"Summarized chunks "
            f"{start + 1}-{min(start + batch_size, len(chunks))}"
        )

    # Stage 2: Reduce summaries until a manageable number remain
    while len(partial_summaries) > batch_size:
        reduced_summaries = []

        for start in range(0, len(partial_summaries), batch_size):
            batch = partial_summaries[start:start + batch_size]

            # Join summary strings, not individual characters
            batch_text = "\n\n".join(batch)

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"""
Combine these summaries into one concise summary.
Preserve distinct topics and important article references.
Do not add unsupported information.

SUMMARIES:
{batch_text}
"""),
            ]

            summary = Call_LLM(messages)

            if not isinstance(summary, str) or not summary.strip():
                raise ValueError(
                    f"Empty or invalid summary during reduction "
                    f"at batch starting at {start}"
                )

            reduced_summaries.append(summary.strip())

        # Replace the old summaries with the reduced summaries
        partial_summaries = reduced_summaries

    # Stage 3: Create the final document overview
    combined_summaries = "\n\n".join(partial_summaries)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"""
Create an overall overview of the entire document using
the supplied summaries.

Include:
1. What the document is and its purpose.
2. Its main objectives.
3. The major topics and rights it covers.
4. Important responsibilities and obligations.
5. A concise list of its key themes.

Preserve relevant article references. Distinguish the
document's overall purpose from individual provisions.
Do not invent information.

DOCUMENT SUMMARIES:
{combined_summaries}
"""),
    ]

    print("Number of summaries for final synthesis:", len(partial_summaries))

    final_summary = Call_LLM(messages)

    # if not isinstance(final_summary, str) or not final_summary.strip():
    #     raise ValueError("The LLM returned an empty final document summary.")

    # # Save only after successful completion
    # with cache_file.open("w", encoding="utf-8") as file:
    #     json.dump(
    #         {"summary": final_summary.strip()},
    #         file,
    #         ensure_ascii=False,
    #         indent=2,
    #     )

    # print(f"Saved document summary to {cache_path}")

    return final_summary.strip()




#=============================
overview_questions = [
    "What is this document about?",
    "What is this PDF about",
    "Can you give me an overview of this document?",
    "Summarize the entire GDPR regulation.",
    "What is the main purpose of the GDPR?",
    "What are the main objectives of this regulation?",
    "What does the GDPR aim to achieve?",
    "Explain the GDPR in simple terms.",
    "What are the major topics covered in this document?",
    "Give me a high-level summary of the GDPR.",
    "What are the key principles discussed in this regulation?",
    "What should I understand about the GDPR as a beginner?",
    "What is the overall scope of this regulation?",
    "Explain the general structure and purpose of this document.",
    "What are the most important things to know about the GDPR?",
    "Give me a general introduction to this regulation."
]

def Overview_quesiton_Embeddings(embeddings):
    return embeddings.embed_documents(overview_questions)

def Incoming_quesiton_Embeddings(embeddings,question):
    return embeddings.embed_query(question)

