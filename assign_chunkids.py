import json
from pathlib import Path

CHUNKS_PATH = Path("data/semantic_chunks.json")
OUTPUT_PATH = Path("data/semantic_chunks_with_ids.json")

def assing_chunk_ids():
    with CHUNKS_PATH.open("r",encoding="utf-8") as file:
        chunks = json.load(file)
    if not isinstance(chunks,list):
        raise ValueError("Chunks are not list!!!")
        
    for index,chunk in enumerate(chunks):
        if not isinstance(chunk,dict) or "page_content" not in chunk:
            raise ValueError("Chunk are empty!!")


        chunk['chunk_id'] = f"gdpr_chunk_{index:04d}"

    with OUTPUT_PATH.open("w",encoding="utf-8") as file:
        json.dump(chunks,file,ensure_ascii=False,indent=2)

    print(f"Total chunks: {len(chunks)}")
    print(f"Saved to: {OUTPUT_PATH}")

    print("\nFirst chunk:")
    print(json.dumps(chunks[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    assing_chunk_ids()
