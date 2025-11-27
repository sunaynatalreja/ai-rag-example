def classify_hallucination(grounding_score, threshold=0.55):
    return grounding_score < threshold
