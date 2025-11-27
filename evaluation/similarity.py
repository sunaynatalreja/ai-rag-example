from sklearn.metrics.pairwise import cosine_similarity

def embedding_similarity(e1, e2):
    return cosine_similarity([e1], [e2])[0][0]
