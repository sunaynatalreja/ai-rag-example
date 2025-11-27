import google.generativeai as genai
import os
import pickle

PICKLE_PATH = "./vector_db/embeddings.pkl"

def embed_texts(text_list):
    response = genai.embed_content(
        model="text-embedding-004",
        content=text_list
    )
    return response['embedding']

def embed_batch(texts):
    # If pickle exists, load embeddings
    if os.path.exists(PICKLE_PATH):
        print("[INFO] Loading embeddings from pickle...")
        with open(PICKLE_PATH, "rb") as f:
            data = pickle.load(f)
            return data["embeddings"]

    print("[INFO] Pickle not found. Creating embeddings with Gemini...")
    embeddings = []
    for t in texts:
        emb = genai.embed_content(model="text-embedding-004", content=t)
        embeddings.append(emb["embedding"])

    # Save to pickle
    os.makedirs(os.path.dirname(PICKLE_PATH), exist_ok=True)
    with open(PICKLE_PATH, "wb") as f:
        pickle.dump({"embeddings": embeddings}, f)

    print("[INFO] Embeddings saved to pickle.")

    return embeddings

