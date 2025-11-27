from ingest.embeddings import embed_batch
from evaluation.similarity import embedding_similarity

def grounding_score(answer, retrieved_docs):
    docs = "\n".join(retrieved_docs)
    emb = embed_batch([answer, docs])
    return embedding_similarity(emb[0], emb[1])
