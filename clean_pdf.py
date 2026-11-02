from langchain_community.document_loaders import PyMuPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from src.embeddings import NomicEmbeddings
import json
from pathlib import Path
import uuid
from nltk.tokenize import sent_tokenize
from langchain_core.documents import Document
import fitz
import re

def Load_PDF(path):
    documents = []
    # 1. Open the PDF using raw PyMuPDF
    doc = fitz.open(path)
    
    print(f"Total pages read by PyMuPDF: {len(doc)}")
    
    for page_num, page in enumerate(doc):
        # 2. Extract using standard layout processing blocks
        text = page.get_text("text")
        
        # 3. Wrap it inside the exact LangChain Document structure your pipeline expects
        langchain_doc = Document(
            page_content=text,
            metadata={"source": path, "page": page_num}
        )
        documents.append(langchain_doc)
        
    return documents

def PDF_Chunking(docs,encoder_model):
    embeddings = NomicEmbeddings(encoder_model=encoder_model)
    text_splitter = SemanticChunker(embeddings)
    print("Total documents:", len(docs))
    semantic_chunks = text_splitter.create_documents(texts=[text.page_content for text in docs],metadatas=[text.metadata for text in docs])
    
    return semantic_chunks


ART_TITLED = re.compile(r"\bArticle\s+(\d+)\s+([A-Z][^.]{2,160}?)\s+1\.")
ART_NUMBER = re.compile(r"\bArticle\s+(\d+)\s+[A-Z]")
MIN_ARTICLE_HEADINGS = 3   # fewer distinct headings than this = no usable Article structure
ANNEX_HEAD = re.compile(r"\bANNEX\s+([IVXLC]+)\b")


def _assign_labels(result, any_start):
    num, label = 0, ""
    in_annex = False                                              # NEW
    labels = []
    for item in result:
        text = item["child_text"]

        annex = ANNEX_HEAD.search(text)                           # NEW
        if annex and num > 0:                                     # NEW
            in_annex, label = True, f"Annex {annex.group(1)}"     # NEW
        if in_annex:                                              # NEW
            labels.append(label)                                  # NEW
            continue                                              # NEW

        found = [(m.start(), int(m.group(1)), f"Article {m.group(1)} {m.group(2).strip()}", True)
                 for m in ART_TITLED.finditer(text)]
        titled = {n for _, n, _, _ in found}
        found += [(m.start(), int(m.group(1)), f"Article {m.group(1)}", False)
                  for m in ART_NUMBER.finditer(text) if int(m.group(1)) not in titled]
        found.sort()

        def accept(n, is_titled):
            return num <= n <= num + 3 or (any_start and num == 0 and is_titled)

        i = 0
        while i < len(found) and found[i][0] <= 2:
            if accept(found[i][1], found[i][3]):
                num, label = found[i][1], found[i][2]
            i += 1
        labels.append(label)
        for _, n, lab, is_titled in found[i:]:
            if accept(n, is_titled):
                num, label = n, lab
    return labels


def _label_articles(result):
    """Sets metadata['article'] on every chunk. Documents without Article headings get ''."""
    labels = None
    for any_start in (False, True):                     # pass 1 = exactly the current behaviour
        labels = _assign_labels(result, any_start)
        if len({l for l in labels if l}) >= MIN_ARTICLE_HEADINGS:
            break
    else:                                               # neither pass found a real structure
        labels = [""] * len(result)
    for item, lab in zip(result, labels):
        item["metadata"]["article"] = lab

def parent_child_chunk(docs, parent_size=1000, child_size=300, overlap=1):
    result = []

    # NEW: heading tracker, defined before the page loop so it carries across pages
    ART_TITLED = re.compile(r"\bArticle\s+(\d+)\s+([A-Z][^.]{2,160}?)\s+1\.")
    ART_NUMBER = re.compile(r"\bArticle\s+(\d+)\s+[A-Z]")
    state = {"num": 0, "label": ""}

    def track(sent):
        m = ART_TITLED.search(sent)
        if m and state["num"] <= int(m.group(1)) <= state["num"] + 3:
            state.update(num=int(m.group(1)), label=f"Article {m.group(1)} {m.group(2).strip()}")
            return
        m = ART_NUMBER.search(sent)
        if m and state["num"] < int(m.group(1)) <= state["num"] + 3:
            state.update(num=int(m.group(1)), label=f"Article {m.group(1)}")

    # Process each page/document from your PDF pipeline
    for doc in docs:
        text = doc.page_content
        # 1. Strip out the annoying recurrent CELEX/EU document headers
        # This removes lines like "02016R0679 — EN — 04.05.2016 — 000.002 — 9"
        text = text.replace("—", "-").replace("–", "-")

        # 2. Use a robust regex pattern targeting the unified layout format
        text = re.sub(r'02016R0679\s*-\s*EN\s*-\s*[\d\.]+\s*-\s*[\d\.]+\s*-\s*\d+', '', text)

        # 3. Clean up the trailing word fragments and newlines
        text = text.replace("compre­\xad\xad\xad\xad\xad\xad\xad\xad\xad\xadhensive", "comprehensive")
        text = text.replace("compre\xadhensive", "comprehensive")
        text = text.replace("compre\xad\nhensive", "comprehensive")
        text = text.replace("\n", " ")
        text = re.sub(r'\s+', ' ', text).strip()

        metadata = doc.metadata
        sentences = sent_tokenize(text)

        # NEW: remember which article each sentence belongs to
        art_of = {}
        for s in sentences:
            track(s)
            art_of.setdefault(s, state["label"])

        # 1. Build Parent Chunks (Sentence-based)
        parents = []
        current = []
        current_len = 0

        for sent in sentences:
            # If adding the sentence exceeds size, save current and shift window
            if current_len + len(sent) > parent_size and current:
                parents.append(" ".join(current))
                current = current[-overlap:] if overlap > 0 else []

            current.append(sent)
            current_len = sum(len(s) for s in current)

        if current:
            parents.append(" ".join(current))

        # 2. Build Child Chunks from each Parent
        for p_idx, parent_text in enumerate(parents):
            parent_id = str(uuid.uuid4())  # Unique ID to reference the parent block
            parent_article = art_of.get(sent_tokenize(parent_text)[0], "")   # NEW

            children = []
            current_child = []
            current_child_len = 0

            for sent in sent_tokenize(parent_text):
                if current_child_len + len(sent) > child_size and current_child:
                    children.append((" ".join(current_child), current_child[0]))   # CHANGED: tuple
                    current_child = current_child[-overlap:] if overlap > 0 else []

                current_child.append(sent)
                current_child_len = sum(len(s) for s in current_child)

            if current_child:
                children.append((" ".join(current_child), current_child[0]))       # CHANGED: tuple

            # 3. Flatten into the structured array mapping dictionary
            for c_idx, (child_text, first_sent) in enumerate(children):            # CHANGED: unpack tuple
                # Preserve your original metadata (like page number)
                child_metadata = metadata.copy()
                child_metadata.update({
                    'parent_id': parent_id,
                    'p_idx': p_idx,
                    'child_idx': f"{p_idx}_{c_idx}",
                    'article': art_of.get(first_sent, parent_article),             # NEW
                })

                result.append({
                    'child_text': child_text,
                    'parent_text': parent_text,
                    'metadata': child_metadata
                })
    _label_articles(result)
    return result



CHUNKS_PATH = Path("data/semantic_chunks.json")
def Save_Chunks(chunks, path=CHUNKS_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)

    data = [
        {
            "page_content": chunk.page_content,
            "metadata": chunk.metadata,
        }
        for chunk in chunks
    ]

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    print(f"Saved {len(chunks)} chunks to {path}")


def Load_Chunks(path=CHUNKS_PATH):
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    chunks = [
        Document(
            page_content=item["page_content"],
            metadata=item["metadata"],
        )
        for item in data
    ]

    print(f"Loaded {len(chunks)} cached chunks from {path}")
    return chunks
