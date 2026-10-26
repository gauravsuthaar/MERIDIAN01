import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score


def run_dbscan(
    frame,
    text_col="normalized_title",
    eps=0.35,
    min_samples=5,
    sample_size=2500,
    random_state=42,
):
    if eps <= 0 or min_samples < 2 or sample_size < 2:
        raise ValueError("Epsilon must be positive; minimum samples and sample size must be at least 2.")

    subset = frame.loc[frame[text_col].astype(str).str.strip().ne("")].copy()
    if len(subset) > sample_size:
        subset = subset.sample(n=sample_size, random_state=random_state)
    if len(subset) < 2:
        raise ValueError("At least two non-empty product descriptions are required.")

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=60_000,
        norm="l2",
        dtype=np.float32,
    )
    vectors = vectorizer.fit_transform(subset[text_col].astype(str))

    # Cosine distance operates directly on normalized sparse TF-IDF and avoids
    # applying Euclidean distance to a high-dimensional text feature space.
    labels = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    ).fit_predict(vectors)
    subset["cluster_id_dbscan"] = labels

    cluster_counts = {
        int(label): int(count)
        for label, count in zip(*np.unique(labels, return_counts=True))
        if label != -1
    }
    noise_count = int(np.count_nonzero(labels == -1))
    assigned = labels != -1
    silhouette = None
    assigned_labels = labels[assigned]
    if len(assigned_labels) > 2 and 1 < len(np.unique(assigned_labels)) < len(assigned_labels):
        silhouette = float(
            silhouette_score(vectors[assigned], assigned_labels, metric="cosine")
        )
    representatives = {}
    for cluster_id in cluster_counts:
        positions = np.flatnonzero(labels == cluster_id)
        centroid = vectors[positions].mean(axis=0)
        similarities = np.asarray(vectors[positions] @ centroid.T).reshape(-1)
        representative_position = positions[int(np.argmax(similarities))]
        representatives[cluster_id] = subset.iloc[int(representative_position)].to_dict()

    return {
        "labels": labels,
        "subset": subset,
        "cluster_counts": cluster_counts,
        "noise_count": noise_count,
        "assigned_count": int(np.count_nonzero(assigned)),
        "silhouette": silhouette,
        "sample_size": len(subset),
        "eps": eps,
        "min_samples": min_samples,
        "representatives": representatives,
        "representation": "Character TF-IDF (3–5 grams)",
        "metric": "Cosine distance",
    }


def create_product_embedding(frame, text_col="normalized_title", sample_size=1800, random_state=42):
    from sklearn.feature_extraction.text import TfidfVectorizer

    subset = frame.loc[frame[text_col].astype(str).str.strip().ne("")].copy()
    if len(subset) > sample_size:
        subset = subset.sample(n=sample_size, random_state=random_state)
    if len(subset) < 3:
        raise ValueError("At least three non-empty product descriptions are required.")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_features=40_000, strip_accents="unicode"
    )
    vectors = vectorizer.fit_transform(subset[text_col].astype(str))
    if vectors.shape[1] < 2:
        raise ValueError("The selected descriptions do not contain enough vocabulary to map.")
    coordinates = TruncatedSVD(n_components=2, random_state=random_state).fit_transform(vectors)
    subset["embedding_x"] = coordinates[:, 0]
    subset["embedding_y"] = coordinates[:, 1]
    return subset
