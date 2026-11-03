"""RAG step 2: compute one Gemini embedding per passage. Resumable: run it again after an interruption.
Run from the project root: python scripts/13_embed_chunks.py [--batch 20] [--pause 4] [--limit N]

Needs GEMINI_API_KEY (in the environment or in the .env file at the project root).
The free tier limits requests and tokens per minute: when Gemini answers "429", the client waits and tries again,
and what is already computed is saved after every batch.

Reads   data/processed/chunks.json
Writes  models/chunk_embeddings.npz
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from src.rag.gemini import GeminiClient, GeminiError, load_api_key  # noqa: E402
from src.rag.store import load_chunks, load_embeddings, save_embeddings  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=20, help="passages per request (max 100)")
    parser.add_argument("--pause", type=float, default=4.0, help="seconds between requests")
    parser.add_argument("--limit", type=int, default=0, help="only embed this many passages (to test)")
    args = parser.parse_args()

    key = load_api_key(config)
    if not key:
        sys.exit("No API key. Create the file .env at the project root containing:\nGEMINI_API_KEY=your_key_here")
    chunks = load_chunks(config)
    if args.limit:
        chunks = chunks[:args.limit]
    done = {}
    existing = load_embeddings(config.EMBEDDINGS_NPZ)
    if existing:
        done = dict(zip(existing[0], existing[1]))
    todo = [c for c in chunks if c["chunk_id"] not in done]
    print(f"{len(chunks)} passages, {len(done)} already embedded, {len(todo)} to do")

    client = GeminiClient(key, config)
    started = time.time()
    try:
        for i in range(0, len(todo), args.batch):
            batch = todo[i:i + args.batch]
            vectors = client.embed([c["text"] for c in batch])
            for chunk, vector in zip(batch, vectors):
                done[chunk["chunk_id"]] = vector
            ids = list(done)
            save_embeddings(config.EMBEDDINGS_NPZ, ids, np.stack([done[j] for j in ids]))
            print(f"  {len(done)}/{len(chunks)} embedded ({time.time() - started:.0f} s)")
            time.sleep(args.pause)
    except GeminiError as error:
        sys.exit(f"\nStopped: {error}\nWhat was computed is saved. Run the script again later to continue.")
    print(f"saved {config.EMBEDDINGS_NPZ}")


if __name__ == "__main__":
    main()
