import chromadb

#client = chromadb.Client()
client = chromadb.PersistentClient(path="chroma_db")

def get_or_create_collection(name):
    existing = [c.name for c in client.list_collections()]
    if name in existing:
        print(f"[INFO] Using existing Chroma collection: {name}")
        col = client.get_collection(name)
        return col, True
    else:
        print(f"[INFO] Creating new Chroma collection: {name}")
        col = client.create_collection(name)
        return col, False

def add_to_collection(collection, ids, documents, embeddings):
    collection.add(ids=ids, documents=documents, embeddings=embeddings)

def query_collection(collection, embedding, k=3):
    return collection.query(query_embeddings=[embedding], n_results=k)

def is_collection_empty(collection):
    return collection.count() == 0
