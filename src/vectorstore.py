"""
vectorstore.py
Embeds clinical note chunks and stores/retrieves them using ChromaDB.
"""

import chromadb
from sentence_transformers import SentenceTransformer
from chunker import load_and_chunk_all

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_PERSIST_DIR = "./chroma_store"
COLLECTION_NAME = "clinical_notes"


class ClinicalVectorStore:

    def __init__(self, persist_dir: str = CHROMA_PERSIST_DIR):
        self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(COLLECTION_NAME)

    def reset(self):
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(COLLECTION_NAME)

    def index_chunks(self, chunks):
        texts = [c.text for c in chunks]
        ids = [c.chunk_id for c in chunks]
        metadatas = [{"note_id": c.note_id, "section": c.section} for c in chunks]
        embeddings = self.embedder.encode(texts).tolist()
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        print(f"Indexed {len(texts)} chunks.")

    def query(self, question: str, top_k: int = 3, note_id: str = None):
        """Returns top_k most relevant chunks, optionally filtered to one note."""
        query_embedding = self.embedder.encode([question]).tolist()
        where_filter = {"note_id": note_id} if note_id else None
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            where=where_filter,
        )
        return results


if __name__ == "__main__":
    store = ClinicalVectorStore()
    chunks = load_and_chunk_all("../data/notes")
    store.index_chunks(chunks)
    test_question = "What medication was given for chest pain?"
    results = store.query(test_question)
    print(results)