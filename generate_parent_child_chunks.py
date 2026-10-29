
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from clean_pdf import Load_PDF, parent_child_chunk


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

PDF_PATH = PROJECT_ROOT / (
    "regulation eu 2016679 of the european parliament and-DOC_1.pdf"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "Real_Metrics"
    / "gdpr_parent_child_chunks_with_ids.json"
)


def stable_hash(text: str) -> str:
    """Generate a deterministic ID component from text."""
    normalized = " ".join(text.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def main():
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    print("Loading GDPR PDF...")
    documents = Load_PDF(str(PDF_PATH))

    print("\nGenerating parent-child chunks...")
    raw_chunks = parent_child_chunk(docs=documents)

    if not raw_chunks:
        raise ValueError("parent_child_chunk() returned no chunks.")

    print(f"Raw child chunks: {len(raw_chunks)}")
    print(f"Returned keys: {list(raw_chunks[0].keys())}")

    output_chunks = []
    seen_child_ids = set()
    child_counts = defaultdict(int)
    parent_ids = set()

    for index, item in enumerate(raw_chunks):
        # Match the actual output structure from parent_child_chunk().
        child_text = item.get("child_text")
        parent_text = item.get("parent_text")
        metadata = dict(item.get("metadata") or {})

        if not isinstance(child_text, str) or not child_text.strip():
            raise ValueError(f"Chunk {index} has missing or empty child_text.")

        if not isinstance(parent_text, str) or not parent_text.strip():
            raise ValueError(f"Chunk {index} has missing or empty parent_text.")

        # Same parent text produces the same parent ID.
        parent_id = f"parent_{stable_hash(parent_text)}"

        # Assign a child index within each parent.
        child_index = child_counts[parent_id]
        child_counts[parent_id] += 1

        child_id = f"{parent_id}_child_{child_index:04d}"

        if child_id in seen_child_ids:
            raise ValueError(f"Duplicate child ID generated: {child_id}")

        seen_child_ids.add(child_id)
        parent_ids.add(parent_id)

        metadata.update({
            "parent_id": parent_id,
            "child_id": child_id,
        })

        output_chunks.append({
            "chunk_id": child_id,
            "parent_id": parent_id,
            "child_id": child_id,
            "page_content": child_text,
            "parent_text": parent_text,
            "child_text":child_text,
            "metadata": metadata,
        })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            output_chunks,
            file,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    # Validate the saved file.
    with OUTPUT_PATH.open("r", encoding="utf-8") as file:
        saved_chunks = json.load(file)

    assert len(saved_chunks) == len(raw_chunks)
    assert len({chunk["child_id"] for chunk in saved_chunks}) == len(saved_chunks)

    print("\nParent-child chunk generation complete.")
    print(f"Source documents/pages: {len(documents)}")
    print(f"Child chunks: {len(saved_chunks)}")
    print(f"Unique parents: {len(parent_ids)}")
    print(f"Unique child IDs: {len(seen_child_ids)}")
    print(f"Output: {OUTPUT_PATH}")

    print("\nFirst 3 generated chunks:")
    for chunk in saved_chunks[:3]:
        print(f"\nChild ID: {chunk['child_id']}")
        print(f"Parent ID: {chunk['parent_id']}")
        print(f"Text: {chunk['page_content'][:250]}")


if __name__ == "__main__":
    main()
