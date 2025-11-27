from ingest.loader import load_pdf
from ingest.create_chunks import chunk_text
from ingest.embeddings import embed_batch
from vectorstore.chromadb import get_or_create_collection, is_collection_empty
from retriever.retriever import retrieve_top_k
from rag.rag_pipeline import run_rag
from evaluation.grounding import grounding_score
from evaluation.hallucination import classify_hallucination

# --- Load and prepare docs ---
doc = load_pdf("data/Indian_History.pdf")
chunks = chunk_text(doc)

# Load or create collection
collection, exists = get_or_create_collection("gemini_rag")

# Use pickle for embeddings
embeddings = embed_batch(chunks)

# If Chroma collection is empty, add the embeddings
if not exists or is_collection_empty(collection):
    print("[INFO] Adding embeddings to Chroma collection...")
    ids = [str(i) for i in range(len(chunks))]
    collection.add(ids=ids, documents=chunks, embeddings=embeddings)
else:
    print("[INFO] Skipping adding to Chroma; data already exists.")

# --- Query ---
query = "What differences does the book highlight between different social organisations or class/caste structures in ancient and medieval societies?"
results = retrieve_top_k(collection, query, k=3)

retrieved_docs = results["documents"][0]
answer = run_rag(query, retrieved_docs)

print("ANSWER:\n", answer)

# --- Evaluation ---
g_score = grounding_score(answer, retrieved_docs)
is_hallucination = classify_hallucination(g_score)

print("\nGrounding Score:", g_score)
print("Hallucination:", is_hallucination)
