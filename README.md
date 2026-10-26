# MERIDIAN AI

MERIDIAN AI is a Streamlit product intelligence workspace for the official UCI Product Classification and Clustering dataset. It supports product classification, DBSCAN-based clustering and candidate matching across merchant offers.

**Problem statement:** Develop an intelligent, scalable and text-aware machine learning framework that automatically classifies and clusters product offers based on their short textual descriptions and subsequently performs entity matching to identify duplicate or equivalent products across different merchants.

## Official dataset and uploaded file

L. Akritidis. “Product Classification and Clustering,” UCI Machine Learning Repository, 2020.  
DOI: https://doi.org/10.24432/C5M91Z

The application prioritizes the uploaded CSV:

`/Users/gauravsuthar/Desktop/pricerunner_aggregate.csv`

The uploaded file has 35,311 rows and seven original columns: Product ID, Product Title, Merchant ID, Cluster ID, Cluster Label, Category ID and Category Label. The verified source has 10 categories, no missing values, and no exact duplicate rows. Repeated titles can represent offers from different merchants and are not removed.

To use another local copy, set `MERIDIAN_DATA_PATH` to its CSV or ZIP path. No generated or unrelated fallback data is used.

## Install and run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Methodology

- **Classification:** word-level TF-IDF (1–2 grams) and balanced Logistic Regression. Training is user-triggered, evaluation uses a reproducible stratified hold-out split, and displayed confidence is the fitted model's predicted class probability.
- **Clustering:** DBSCAN over sparse character-level TF-IDF (3–5 grams) with cosine distance. The studio exposes epsilon, minimum samples and sample size. Noise label `-1` is reported separately. A silhouette score is shown only when assumptions are satisfied and excludes noise.
- **Entity matching:** nearest text candidates from other merchants in the same category, scored with character TF-IDF cosine, character n-gram overlap, token Jaccard overlap and normalized-text sequence similarity. Scores are heuristic similarities, not calibrated probabilities.
- **Embedding explorer:** actual word-level TF-IDF features reduced to two dimensions using TruncatedSVD. This is a projection and does not preserve every original text distance.
- **Matching validation:** official Cluster ID values can be used as reference groups to check generated same-category candidate pairs. Pair metrics describe only the retrieved candidate set; they do not claim exhaustive recall over all possible offer pairs.
- **Preprocessing:** original CSV columns and values are retained. Normalized product title and title length are derived helper fields.

## Limitations and persistence

- DBSCAN is parameter-sensitive and uses a reproducible, configurable sample for interactive performance.
- Review decisions are session-local; no shared persistence database is configured.
- Candidate-level matching metrics are limited to retrieved candidates and use Cluster ID as a reference grouping.
- The trained model artifact is stored locally under `.meridian/artifacts/` and loaded only when its dataset fingerprint matches the selected source data.

## Project structure

```text
app.py
pages/             Native Streamlit multipage entry points
components/        Shared page views, chart helpers and visual styling
src/               Data loading, preprocessing and machine-learning workflows
.meridian/         Local model artifact generated after training
requirements.txt
README.md
```
