from ingest.embeddings import embed_texts
from vectorstore.chromadb import query_collection

def retrieve_top_k(collection, query, k=3):
    q_emb = embed_texts([query])[0]
    result = query_collection(collection, q_emb, k=k)
    return result
