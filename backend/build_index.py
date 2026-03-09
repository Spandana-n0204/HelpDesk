import os
import logging

from document_loader import load_all_chunks
from vector_store     import VectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DOCS_DIRS = [
    os.path.join(BASE_DIR, "..", "documents"),
    os.path.join(BASE_DIR, "..", "extracted_data"),
    os.path.join(BASE_DIR, "data"),
]


def rebuild(docs_dirs: list = None) -> VectorStore:
    dirs   = docs_dirs or DOCS_DIRS
    chunks = load_all_chunks(dirs)
    if not chunks:
        raise RuntimeError(f"No chunks found in: {dirs}")
    store  = VectorStore()
    store.build(chunks)
    logger.info(f"Index built with {len(chunks)} chunks.")
    return store


if __name__ == "__main__":
    rebuild()
    print("Done. vector_store.pkl written.")