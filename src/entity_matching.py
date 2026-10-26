from difflib import SequenceMatcher

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing import normalize_text


def _char_similarity(a, b):
    a, b = normalize_text(a), normalize_text(b)
    if not a or not b:
        return 0.0
    grams_a = {a[index : index + 3] for index in range(max(1, len(a) - 2))}
    grams_b = {b[index : index + 3] for index in range(max(1, len(b) - 2))}
    return len(grams_a & grams_b) / max(len(grams_a | grams_b), 1)


def _token_overlap(a, b):
    tokens_a, tokens_b = set(normalize_text(a).split()), set(normalize_text(b).split())
    return len(tokens_a & tokens_b) / max(len(tokens_a | tokens_b), 1)


def compare_offers(a, b, strong_threshold=0.8, review_threshold=0.6):
    normalized_a, normalized_b = normalize_text(a), normalize_text(b)
    if not normalized_a or not normalized_b:
        raise ValueError("Both product offers must contain text.")
    matrix = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 5), min_df=1, norm="l2"
    ).fit_transform([normalized_a, normalized_b])
    cosine = float(cosine_similarity(matrix[0], matrix[1])[0, 0])
    char_score = _char_similarity(normalized_a, normalized_b)
    overlap = _token_overlap(normalized_a, normalized_b)
    sequence = SequenceMatcher(None, normalized_a, normalized_b).ratio()
    score = float(0.45 * cosine + 0.30 * char_score + 0.15 * overlap + 0.10 * sequence)
    if score >= strong_threshold:
        decision = "Strong candidate"
    elif score >= review_threshold:
        decision = "Review"
    else:
        decision = "Low similarity"
    return {
        "decision": decision,
        "score": score,
        "tfidf_cosine": cosine,
        "char_ngram": char_score,
        "token_overlap": overlap,
        "normalized_text_similarity": sequence,
        "strong_threshold": strong_threshold,
        "review_threshold": review_threshold,
    }


def find_candidates(frame, product_index, limit=8, strong_threshold=0.8, review_threshold=0.6):
    if product_index not in frame.index:
        raise ValueError("Selected product is not present in the dataset.")
    source = frame.loc[product_index]
    candidates = frame.loc[frame.index != product_index]
    if "category_id" in frame.columns and "category_id" in source.index:
        candidates = candidates.loc[candidates["category_id"] == source["category_id"]]
    elif "category_label" in frame.columns:
        candidates = candidates.loc[candidates["category_label"] == source["category_label"]]
    if "merchant_id" in frame.columns:
        candidates = candidates.loc[candidates["merchant_id"] != source.get("merchant_id")]

    candidates = candidates.loc[candidates["normalized_title"].astype(str).str.strip().ne("")]
    if candidates.empty:
        return []

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=1,
        max_features=80_000,
        norm="l2",
    )
    texts = [source["normalized_title"], *candidates["normalized_title"].astype(str).tolist()]
    vectors = vectorizer.fit_transform(texts)
    scores = cosine_similarity(vectors[0], vectors[1:]).ravel()
    top_positions = np.argsort(scores)[::-1][: max(limit * 4, limit)]
    results = []
    for position in top_positions:
        candidate_index = candidates.index[position]
        row = candidates.loc[candidate_index]
        features = compare_offers(
            source["product_title"],
            row["product_title"],
            strong_threshold=strong_threshold,
            review_threshold=review_threshold,
        )
        result = {
            "index": candidate_index,
            "score": features["score"],
            "decision": features["decision"],
            "other_product": row["product_title"],
            "merchant_id": row.get("merchant_id"),
            "category_label": row.get("category_label"),
            "product_id": row.get("product_id"),
            "features": features,
        }
        if "cluster_id" in frame.columns:
            result["same_reference_group"] = bool(
                row.get("cluster_id") == source.get("cluster_id")
            )
        results.append(result)
    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:limit]
