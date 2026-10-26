import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from components.charts import (
    category_distribution,
    cluster_distribution,
    cluster_noise_donut,
    confusion_matrix_chart,
    confusion_point_examples,
    matching_distribution,
    per_category_performance,
    precision_recall_curve,
    product_embedding_map,
    text_length_distribution,
)
from components.layout import empty_state, metric_card, page_header, section_title
from src.classification import predict_category, save_model_artifact, train_text_classifier
from src.clustering import create_product_embedding, run_dbscan
from src.entity_matching import compare_offers, find_candidates


@st.cache_data(show_spinner=False, max_entries=8)
def cached_dbscan(frame, eps, min_samples, sample_size, random_state=42):
    return run_dbscan(
        frame,
        eps=eps,
        min_samples=min_samples,
        sample_size=sample_size,
        random_state=random_state,
    )


@st.cache_data(show_spinner=False, max_entries=8)
def cached_embedding(frame, sample_size):
    return create_product_embedding(frame, sample_size=sample_size)


def _frame():
    return st.session_state["dataset_df"]


def _model():
    return st.session_state.get("model_result")


def _navigate(page):
    st.switch_page(f"pages/{page}.py")


def _train_model(test_size=0.2, sample_limit=12_000):
    with st.status("Training and evaluating the category model…", expanded=True) as status:
        st.write("Preparing a stratified split of the actual product titles and category labels.")
        result = train_text_classifier(
            _frame(),
            test_size=test_size,
            max_rows=sample_limit,
        )
        st.write(f"Fitted on {result['n_train']:,} records; evaluated on {result['n_test']:,} held-out records.")
        save_model_artifact(
            result,
            st.session_state["dataset_fingerprint"],
            st.session_state["artifact_path"],
        )
        st.session_state["model_result"] = result
        status.update(label="Classification model trained", state="complete", expanded=False)
    return result


def _cluster_state():
    return st.session_state.get("cluster_result")


def _run_cluster(eps, min_samples, sample_size):
    with st.status("Running cosine-distance DBSCAN…", expanded=True) as status:
        st.write("Building sparse character TF-IDF features and clustering the selected offer sample.")
        result = cached_dbscan(_frame(), eps, min_samples, sample_size)
        st.session_state["cluster_result"] = result
        st.session_state["cluster_parameters"] = {
            "eps": eps,
            "min_samples": min_samples,
            "sample_size": sample_size,
        }
        status.update(label="DBSCAN analysis complete", state="complete", expanded=False)
    return result


def _product_label(row):
    product_id = row.get("product_id", row.name)
    return f"{product_id}  ·  {str(row['product_title'])[:76]}"


def _filtered_products(frame, search, category, merchant, sort_column, ascending):
    filtered = frame
    if search.strip():
        filtered = filtered.loc[
            filtered["product_title"].str.contains(search.strip(), case=False, regex=False, na=False)
        ]
    if category != "All categories":
        filtered = filtered.loc[filtered["category_label"] == category]
    if merchant != "All merchants" and "merchant_id" in filtered:
        filtered = filtered.loc[filtered["merchant_id"].astype(str) == merchant]
    if sort_column in filtered.columns:
        filtered = filtered.sort_values(sort_column, ascending=ascending, kind="stable")
    return filtered


def render_overview():
    frame = _frame()
    page_header(
        "Understand your product catalog.",
        "Classify offers, discover product relationships and resolve duplicate listings through text-aware machine learning.",
    )

    action_cols = st.columns(4)
    with action_cols[0]:
        if st.button("Explore products", use_container_width=True):
            _navigate("product_explorer")
    with action_cols[1]:
        if st.button("Train classifier", type="primary", use_container_width=True):
            try:
                _train_model()
            except Exception as exc:
                st.error(f"Training did not complete: {exc}")
    with action_cols[2]:
        if st.button("Run DBSCAN", use_container_width=True):
            try:
                _run_cluster(0.35, 5, min(len(frame), 2500))
            except Exception as exc:
                st.error(f"Clustering did not complete: {exc}")
    with action_cols[3]:
        if st.button("Compare offers", use_container_width=True):
            _navigate("entity_resolution")

    model = _model()
    cluster = _cluster_state()
    cluster_count = len(cluster["cluster_counts"]) if cluster else "Not run"
    candidate_count = len(st.session_state.get("candidate_matches", []))
    metrics = st.columns(4)
    with metrics[0]:
        metric_card("Product offers", f"{len(frame):,}")
    with metrics[1]:
        metric_card("Categories", f"{frame['category_label'].nunique():,}")
    with metrics[2]:
        metric_card("DBSCAN clusters", cluster_count)
    with metrics[3]:
        metric_card("Retrieved match candidates", f"{candidate_count:,}" if candidate_count else "Not run")

    first, second = st.columns([1.1, 0.9], gap="large")
    with first:
        section_title("Category distribution", "Actual offer counts by the dataset's category labels. Select a category to explore its listings.")
        chart_event = st.plotly_chart(
            category_distribution(frame, st.session_state.get("overview_category")),
            use_container_width=True,
            on_select="rerun",
            selection_mode="points",
            key="overview_category_chart",
        )
        points = chart_event.selection.points
        if points:
            category = points[0].get("y")
            if category in frame["category_label"].values:
                st.session_state["overview_category"] = category
        selected_category = st.session_state.get("overview_category")
        if selected_category:
            st.caption(f"Selected category: {selected_category}")
            category_products = frame.loc[frame["category_label"] == selected_category]
            st.dataframe(
                category_products[["product_id", "product_title", "merchant_id"]].head(6),
                hide_index=True,
                use_container_width=True,
            )
    with second:
        section_title("Clustering analysis", "DBSCAN uses cosine distance over sparse character TF-IDF features.")
        if cluster:
            st.plotly_chart(
                cluster_noise_donut(cluster["assigned_count"], cluster["noise_count"]),
                use_container_width=True,
            )
            st.caption(
                f"{len(cluster['cluster_counts']):,} clusters · {cluster['noise_count']:,} noise points "
                f"from a reproducible sample of {cluster['sample_size']:,} offers."
            )
        else:
            empty_state(
                "Clustering has not been run",
                "Run DBSCAN to inspect actual cluster sizes and noise assignments.",
                "Open clustering studio",
                key="overview_open_clusters",
            )
            if st.session_state.get("overview_open_clusters"):
                _navigate("clustering_studio")

    lower_left, lower_right = st.columns([1.1, 0.9], gap="large")
    with lower_left:
        section_title("Representative listings", "A small sample from the uploaded catalog.")
        st.dataframe(
            frame[["product_id", "product_title", "category_label", "merchant_id"]]
            .sample(n=min(7, len(frame)), random_state=7)
            .reset_index(drop=True),
            hide_index=True,
            use_container_width=True,
        )
    with lower_right:
        section_title("Pipeline status", "Available data and model outputs.")
        status_rows = [
            ("Uploaded dataset", "Ready", f"{len(frame):,} records"),
            ("Text normalization", "Ready", "Product titles normalized"),
            ("Classification model", "Ready" if model else "Not trained", f"{model['accuracy']:.1%} hold-out accuracy" if model else "Train to evaluate"),
            ("DBSCAN", "Ready" if cluster else "Not run", f"{len(cluster['cluster_counts']):,} clusters" if cluster else "Run clustering to inspect"),
            ("Matching candidates", "Ready" if candidate_count else "Not run", f"{candidate_count:,} retrieved" if candidate_count else "Search a product to generate candidates"),
        ]
        for name, state, note in status_rows:
            label, detail = st.columns([1.15, 1.4])
            label.markdown(f"**{name}**")
            detail.caption(f"{state} · {note}")


def render_product_explorer():
    frame = _frame()
    page_header("Explore your catalog", "Search and inspect the actual product offers, with filters and exports.")
    controls = st.columns([2, 1, 1, 1])
    search = controls[0].text_input("Search descriptions", key="explorer_search")
    categories = ["All categories", *sorted(frame["category_label"].dropna().unique())]
    category = controls[1].selectbox("Category", categories, key="explorer_category")
    merchants = ["All merchants"]
    if "merchant_id" in frame:
        merchants += sorted(frame["merchant_id"].dropna().astype(str).unique())
    merchant = controls[2].selectbox("Merchant", merchants, key="explorer_merchant")
    sort_columns = [column for column in ("product_title", "category_label", "merchant_id", "product_id") if column in frame]
    sort_column = controls[3].selectbox(
        "Sort by", sort_columns, format_func=lambda column: column.replace("_", " ").title()
    )
    direction, page_size = st.columns([1, 1])
    ascending = direction.selectbox("Order", ["Ascending", "Descending"]) == "Ascending"
    rows_per_page = page_size.selectbox("Rows per page", [20, 50, 100], index=0)
    filtered = _filtered_products(frame, search, category, merchant, sort_column, ascending)
    total_pages = max(1, (len(filtered) + rows_per_page - 1) // rows_per_page)
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
    st.caption(f"{len(filtered):,} offers match these filters · page {page} of {total_pages}")
    start = (page - 1) * rows_per_page
    page_frame = filtered.iloc[start : start + rows_per_page]
    st.dataframe(
        page_frame[["product_id", "product_title", "category_label", "merchant_id"]],
        hide_index=True,
        use_container_width=True,
    )

    if not page_frame.empty:
        product_ids = page_frame["product_id"].tolist()
        selected_id = st.selectbox("Inspect a product", product_ids, format_func=lambda pid: _product_label(frame.loc[frame["product_id"] == pid].iloc[0]), key="explorer_selected_product")
        selected = frame.loc[frame["product_id"] == selected_id].iloc[0]
        left, right = st.columns([1.3, 0.7], gap="large")
        with left:
            section_title("Product details", "Fields below are present in the uploaded dataset.")
            st.markdown(f"**{selected['product_title']}**")
            detail_columns = st.columns(3)
            for index, field in enumerate(("product_id", "category_label", "merchant_id")):
                if field in selected.index:
                    detail_columns[index].caption(field.replace("_", " ").title())
                    detail_columns[index].write(str(selected[field]))
            st.caption(f"Normalized description · {selected['normalized_title']}")
        with right:
            section_title("Similar offers", "Top text-similar offers from the same category at other merchants.")
            if st.button("Find similar products", key="explorer_find_similar"):
                try:
                    st.session_state["explorer_similar"] = find_candidates(
                        frame, selected.name, limit=5
                    )
                    st.session_state["explorer_similar_source"] = selected.name
                except Exception as exc:
                    st.error(f"Could not retrieve recommendations: {exc}")
            if st.session_state.get("explorer_similar_source") == selected.name:
                recommendations = st.session_state.get("explorer_similar", [])
                if recommendations:
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {
                                    "Product": item["other_product"],
                                    "Merchant": item["merchant_id"],
                                    "Similarity": item["score"],
                                }
                                for item in recommendations
                            ]
                        ),
                        hide_index=True,
                        use_container_width=True,
                    )
                else:
                    st.caption("No recommendations were retrieved for this product.")
    st.download_button(
        "Export filtered offers",
        filtered.to_csv(index=False).encode("utf-8"),
        file_name="meridian_filtered_products.csv",
        mime="text/csv",
        key="explorer_export",
    )


def render_clustering_studio():
    frame = _frame()
    page_header("Clustering analysis", "Explore product groups with DBSCAN over character-level TF-IDF and cosine distance.")
    if "cluster_eps" not in st.session_state:
        st.session_state["cluster_eps"] = 0.35
        st.session_state["cluster_min_samples"] = 5
        st.session_state["cluster_sample_size"] = min(2500, len(frame))
    reset, _ = st.columns([1, 4])
    if reset.button("Reset parameters", key="cluster_reset"):
        st.session_state["cluster_eps"] = 0.35
        st.session_state["cluster_min_samples"] = 5
        st.session_state["cluster_sample_size"] = min(2500, len(frame))
        st.session_state.pop("cluster_result", None)
        st.rerun()
    with st.form("cluster_controls"):
        params = st.columns(3)
        eps = params[0].slider("Epsilon · cosine distance", 0.05, 0.95, step=0.01, key="cluster_eps")
        min_samples = params[1].slider("Minimum samples", 2, 20, key="cluster_min_samples")
        sample_size = params[2].slider(
            "Sample size", min_value=min(500, len(frame)), max_value=min(5000, len(frame)),
            step=100, key="cluster_sample_size",
        )
        run = st.form_submit_button("Run DBSCAN", type="primary")
    if run:
        try:
            _run_cluster(eps, min_samples, sample_size)
        except Exception as exc:
            st.error(f"DBSCAN did not complete: {exc}")
    cluster = _cluster_state()
    if not cluster:
        empty_state(
            "No clustering result yet",
            "Choose a neighborhood radius, minimum group size, and sample size, then run DBSCAN.",
        )
        return

    a, b, c, d = st.columns(4)
    with a:
        metric_card("Products analyzed", f"{cluster['sample_size']:,}")
    with b:
        metric_card("Detected clusters", f"{len(cluster['cluster_counts']):,}")
    with c:
        metric_card("Noise points", f"{cluster['noise_count']:,}")
    with d:
        metric_card("Noise share", f"{cluster['noise_count'] / max(cluster['sample_size'], 1):.1%}")
    st.caption(f"Representation: {cluster['representation']} · Metric: {cluster['metric']} · ε={cluster['eps']:.2f} · min_samples={cluster['min_samples']}")
    if cluster["silhouette"] is not None:
        st.caption(f"Cosine silhouette on assigned products only: {cluster['silhouette']:.3f}. Noise points are excluded from this score.")
    chart_col, noise_col = st.columns([1.45, 0.75], gap="large")
    with chart_col:
        section_title("Cluster sizes", "Counts exclude DBSCAN noise label −1. Select a cluster below to inspect its members.")
        if cluster["cluster_counts"]:
            st.plotly_chart(
                cluster_distribution(cluster["subset"]),
                use_container_width=True,
                key="cluster_size_chart",
            )
        else:
            st.info("These parameters classified every analyzed product as noise.")
    with noise_col:
        section_title("Assigned offers and noise", "Share of sampled offers assigned to a DBSCAN cluster versus noise.")
        st.plotly_chart(
            cluster_noise_donut(cluster["assigned_count"], cluster["noise_count"]),
            use_container_width=True,
        )

    if not cluster["cluster_counts"]:
        return
    chosen_cluster = st.selectbox(
        "Inspect cluster",
        sorted(cluster["cluster_counts"]),
        format_func=lambda cluster_id: f"Cluster {cluster_id} · {cluster['cluster_counts'][cluster_id]:,} products",
        key="cluster_selected_id",
    )
    representative = cluster["representatives"][chosen_cluster]
    st.markdown(f"**Representative offer** · {representative.get('product_title', '')}")
    members = cluster["subset"].loc[cluster["subset"]["cluster_id_dbscan"] == chosen_cluster]
    st.dataframe(
        members[[column for column in ("product_id", "product_title", "category_label", "merchant_id") if column in members]],
        hide_index=True,
        use_container_width=True,
    )


def render_classification_lab():
    frame = _frame()
    page_header("Classification lab", "Train a category model, review its held-out performance, and classify new product descriptions.")
    model = _model()
    if model:
        st.success(
            f"Model trained · {model['training_rows']:,} records · "
            f"{model['n_test']:,}-offer hold-out · macro F1 {model['f1']:.3f}"
        )
    else:
        st.info("No compatible model artifact is loaded. Start a training run to create one from this dataset.")
    with st.expander("Training configuration", expanded=not bool(model)):
        training_size = st.slider(
            "Maximum training records",
            min_value=5000,
            max_value=len(frame),
            value=min(12_000, len(frame)),
            step=1000 if len(frame) >= 6000 else 500,
            key="classification_training_size",
        )
        test_size = st.slider("Hold-out proportion", 0.1, 0.35, 0.2, 0.05, key="classification_test_size")
        if st.button("Train and evaluate model", type="primary", key="train_classifier"):
            try:
                model = _train_model(test_size=test_size, sample_limit=training_size)
                st.rerun()
            except Exception as exc:
                st.error(f"Model training failed: {exc}")
    if model:
        metric_cols = st.columns(4)
        for column, label, value in zip(
            metric_cols,
            ("Accuracy", "Macro precision", "Macro recall", "Macro F1"),
            (model["accuracy"], model["precision"], model["recall"], model["f1"]),
        ):
            with column:
                metric_card(label, f"{value:.3f}")
        left, right = st.columns([1.1, 0.9], gap="large")
        with left:
            section_title("Category performance", "Precision, recall and F1 on the stratified held-out split.")
            st.plotly_chart(per_category_performance(model), use_container_width=True)
        with right:
            section_title("Confusion matrix", "Rows are actual categories; columns are predicted categories. Select a cell to inspect examples.")
            selected = st.plotly_chart(
                confusion_matrix_chart(model),
                use_container_width=True,
                on_select="rerun",
                selection_mode="points",
                key="classification_confusion",
            )
            if selected.selection.points:
                point = selected.selection.points[0]
                examples = confusion_point_examples(model, str(point.get("y")), str(point.get("x")))
                st.caption(f"{len(examples):,} held-out examples · actual {point.get('y')} → predicted {point.get('x')}")
                st.dataframe(examples.head(8), hide_index=True, use_container_width=True)
        pr = precision_recall_curve(model)
        if pr is not None:
            with st.container(border=True):
                section_title("Precision–recall", "One-vs-rest curves use predicted class probabilities from the held-out split.")
                st.plotly_chart(pr, use_container_width=True)

        section_title("Classify descriptions", "Enter one product description per line. Logistic Regression supports the displayed confidence values.")
        raw_texts = st.text_area(
            "Product descriptions",
            placeholder="Example: samsung 55 inch 4k smart television",
            key="classification_input",
            height=115,
        )
        if st.button("Classify offers", key="predict_products"):
            descriptions = [value.strip() for value in raw_texts.splitlines() if value.strip()]
            if not descriptions:
                st.warning("Enter at least one product description.")
            else:
                try:
                    labels, probabilities = predict_category(model["pipeline"], descriptions)
                    predictions = pd.DataFrame(
                        {
                            "Product description": descriptions,
                            "Predicted category": labels,
                            "Confidence": probabilities.max(axis=1),
                        }
                    )
                    st.session_state["classification_predictions"] = predictions
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")
        if "classification_predictions" in st.session_state:
            predictions = st.session_state["classification_predictions"]
            st.dataframe(predictions, hide_index=True, use_container_width=True)
            st.download_button(
                "Export predictions",
                predictions.to_csv(index=False).encode("utf-8"),
                file_name="meridian_classification_predictions.csv",
                mime="text/csv",
                key="classification_export",
            )


def _product_picker(frame, label, key, default_index=0):
    search = st.text_input(f"Find {label.lower()}", key=f"{key}_search", placeholder="Search product title")
    matches = frame
    if search.strip():
        matches = frame.loc[
            frame["product_title"].str.contains(search.strip(), case=False, regex=False, na=False)
        ]
    options = matches.head(300)
    if options.empty:
        st.warning(f"No products match “{search}”.")
        return None
    selected_id = st.selectbox(
        label,
        options["product_id"].tolist(),
        index=min(default_index, len(options) - 1),
        format_func=lambda product_id: _product_label(frame.loc[frame["product_id"] == product_id].iloc[0]),
        key=key,
    )
    return frame.loc[frame["product_id"] == selected_id].iloc[0]


def _show_offer(row, title):
    with st.container(border=True):
        section_title(title)
        st.markdown(f"**{row['product_title']}**")
        st.caption(f"Normalized · {row['normalized_title']}")
        attributes = [
            (name, str(row.get(field)))
            for name, field in (
                ("Product ID", "product_id"),
                ("Merchant ID", "merchant_id"),
                ("Category", "category_label"),
                ("Reference group", "cluster_id"),
            )
            if field in row.index
        ]
        st.dataframe(pd.DataFrame(attributes, columns=["Available attribute", "Value"]), hide_index=True, use_container_width=True)


def render_entity_resolution():
    frame = _frame()
    page_header("Review potential matches", "Compare actual catalog offers, inspect similarity evidence, and record session-local review decisions.")
    threshold_col, review_col = st.columns(2)
    strong_threshold = threshold_col.slider("Strong candidate threshold", 0.5, 0.98, 0.8, 0.01, key="match_strong_threshold")
    review_threshold = review_col.slider(
        "Review threshold",
        0.3,
        min(0.95, strong_threshold),
        min(0.6, strong_threshold - 0.01),
        0.01,
        key="match_review_threshold",
    )
    left, right = st.columns(2, gap="large")
    with left:
        product_a = _product_picker(frame, "Offer A", "match_product_a")
    with right:
        product_b = _product_picker(frame, "Offer B", "match_product_b", default_index=1)
    if product_a is None or product_b is None:
        return
    offer_columns = st.columns(2, gap="large")
    with offer_columns[0]:
        _show_offer(product_a, "Offer A")
    with offer_columns[1]:
        _show_offer(product_b, "Offer B")
    actions = st.columns(4)
    if actions[0].button("Compare offers", type="primary", key="compare_selected_offers"):
        try:
            match = compare_offers(
                product_a["product_title"],
                product_b["product_title"],
                strong_threshold,
                review_threshold,
            )
            match["product_a_id"] = product_a["product_id"]
            match["product_b_id"] = product_b["product_id"]
            st.session_state["current_comparison"] = match
        except Exception as exc:
            st.error(f"Comparison failed: {exc}")
    current = st.session_state.get("current_comparison")
    if current and current["product_a_id"] == product_a["product_id"] and current["product_b_id"] == product_b["product_id"]:
        st.markdown(f"### {current['decision']} · {current['score']:.3f}")
        st.caption("Heuristic similarity score, not a calibrated probability. Use the reference grouping in this dataset as a validation label where available.")
        features = pd.DataFrame(
            [
                ("Word/character TF-IDF cosine", current["tfidf_cosine"]),
                ("Character n-gram overlap", current["char_ngram"]),
                ("Token Jaccard overlap", current["token_overlap"]),
                ("Normalized text sequence", current["normalized_text_similarity"]),
            ],
            columns=["Signal", "Score"],
        )
        st.dataframe(features, hide_index=True, use_container_width=True)
        if "cluster_id" in product_a.index and "cluster_id" in product_b.index:
            same_group = product_a["cluster_id"] == product_b["cluster_id"]
            st.info(
                "Dataset reference groups agree for this pair."
                if same_group
                else "Dataset reference groups differ for this pair."
            )
    review_key = f"{product_a['product_id']}:{product_b['product_id']}"
    review_state = st.session_state.setdefault("match_reviews", {})
    if current and current["product_a_id"] == product_a["product_id"] and current["product_b_id"] == product_b["product_id"]:
        action_cols = st.columns(4)
        for column, label, decision in zip(
            action_cols,
            ("Accept match", "Reject match", "Flag for review"),
            ("accepted", "rejected", "flagged"),
        ):
            if column.button(label, key=f"review_{decision}"):
                review_state[review_key] = decision
        if action_cols[3].button("Reset comparison", key="reset_comparison"):
            st.session_state.pop("current_comparison", None)
            review_state.pop(review_key, None)
            st.rerun()
        if review_key in review_state:
            st.success(f"Decision recorded for this session: {review_state[review_key]}. It is not saved to a shared database.")

    st.divider()
    section_title("Find candidates for an offer", "Retrieve nearest text-similar listings from different merchants in the same category.")
    seed = _product_picker(frame, "Source offer", "candidate_source")
    if seed is None:
        return
    if st.button("Generate candidate matches", type="primary", key="generate_candidates"):
        try:
            candidates = find_candidates(
                frame,
                seed.name,
                limit=15,
                strong_threshold=strong_threshold,
                review_threshold=review_threshold,
            )
            st.session_state["candidate_matches"] = candidates
            st.session_state["candidate_source_index"] = seed.name
        except Exception as exc:
            st.error(f"Candidate generation failed: {exc}")
    candidates = st.session_state.get("candidate_matches", [])
    if candidates and st.session_state.get("candidate_source_index") == seed.name:
        candidate_table = pd.DataFrame(
            [
                {
                    "Product ID": item["product_id"],
                    "Product": item["other_product"],
                    "Merchant": item["merchant_id"],
                    "Similarity": item["score"],
                    "Decision": "Strong candidate" if item["score"] >= strong_threshold else ("Review" if item["score"] >= review_threshold else "Low similarity"),
                    "Reference group match": item.get("same_reference_group", None),
                }
                for item in candidates
            ]
        )
        st.plotly_chart(
            matching_distribution(candidates, review_threshold),
            use_container_width=True,
            key="candidate_score_distribution",
        )
        selection = st.dataframe(
            candidate_table,
            hide_index=True,
            use_container_width=True,
            on_select="rerun",
            selection_mode="single-row",
            key="candidate_matches_table",
        )
        st.download_button(
            "Export candidate results",
            candidate_table.to_csv(index=False).encode("utf-8"),
            file_name="meridian_candidate_matches.csv",
            mime="text/csv",
            key="candidate_export",
        )
        if "cluster_id" in frame.columns and candidates:
            predicted = np.array([item["score"] >= review_threshold for item in candidates])
            actual = np.array([item["same_reference_group"] for item in candidates], dtype=bool)
            true_positives = int(np.count_nonzero(predicted & actual))
            precision = true_positives / max(int(predicted.sum()), 1)
            recall = true_positives / max(int(actual.sum()), 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-12)
            st.caption(
                f"On this retrieved candidate set only · precision {precision:.2f}, recall {recall:.2f}, "
                f"F1 {f1:.2f}. Recall is not exhaustive over all possible catalog pairs."
            )
        selected_rows = selection.selection.rows
        if selected_rows:
            candidate = candidates[selected_rows[0]]
            candidate_product = frame.loc[candidate["index"]]
            st.markdown("**Selected candidate pair**")
            pair_cols = st.columns(2)
            with pair_cols[0]:
                _show_offer(seed, "Source offer")
            with pair_cols[1]:
                _show_offer(candidate_product, "Candidate offer")
            pair_key = f"{seed['product_id']}:{candidate_product['product_id']}"
            st.caption(f"Similarity {candidate['score']:.3f} · {candidate['decision']}")
            review_cols = st.columns(3)
            for col, label, value in zip(
                review_cols,
                ("Accept", "Reject", "Flag for review"),
                ("accepted", "rejected", "flagged"),
            ):
                if col.button(label, key=f"candidate_{value}_{pair_key}"):
                    st.session_state.setdefault("match_reviews", {})[pair_key] = value
            if pair_key in st.session_state.get("match_reviews", {}):
                st.success(f"Session-local decision: {st.session_state['match_reviews'][pair_key]}.")
    elif st.session_state.get("candidate_source_index") == seed.name:
        st.info("No cross-merchant candidates were found for this offer.")


def render_embedding_explorer():
    frame = _frame()
    page_header("Product embedding explorer", "Map product-title TF-IDF vectors into two dimensions using TruncatedSVD.")
    controls = st.columns([1, 1, 1])
    categories = ["All categories", *sorted(frame["category_label"].dropna().unique())]
    category = controls[0].selectbox("Category filter", categories, key="embedding_category")
    merchants = ["All merchants"]
    if "merchant_id" in frame:
        merchants.extend(sorted(frame["merchant_id"].dropna().astype(str).unique()))
    merchant = controls[1].selectbox("Merchant filter", merchants, key="embedding_merchant")
    sample_size = controls[2].selectbox("Maximum points", [500, 1000, 1800], index=2, key="embedding_sample_size")
    filtered = frame
    if category != "All categories":
        filtered = filtered.loc[filtered["category_label"] == category]
    if merchant != "All merchants":
        filtered = filtered.loc[filtered["merchant_id"].astype(str) == merchant]
    st.caption(f"Projection uses a reproducible sample of up to {sample_size:,} offers from {len(filtered):,} filtered records.")
    if st.button("Generate product map", type="primary", key="build_embedding_map"):
        try:
            with st.spinner("Building text features and reducing them with TruncatedSVD…"):
                embedding = cached_embedding(filtered, min(sample_size, len(filtered)))
            cluster = _cluster_state()
            if cluster is not None and "product_id" in cluster["subset"]:
                labels_by_id = cluster["subset"].set_index("product_id")["cluster_id_dbscan"]
                embedding["cluster_id_dbscan"] = embedding["product_id"].map(labels_by_id).fillna(-2).astype(int)
            else:
                embedding["cluster_id_dbscan"] = -2
            st.session_state["product_embedding"] = embedding
            st.session_state["product_embedding_filter"] = (category, merchant, sample_size)
        except Exception as exc:
            st.error(f"Embedding map could not be generated: {exc}")
    embedding = st.session_state.get("product_embedding")
    if embedding is None or st.session_state.get("product_embedding_filter") != (category, merchant, sample_size):
        empty_state("No current map", "Choose the filters, then generate a fresh two-dimensional text projection.")
        return
    section_title("Text projection", "Points closer together share more text-feature structure in the reduced space; 2D distances are approximate.")
    chart = st.plotly_chart(
        product_embedding_map(embedding),
        use_container_width=True,
        on_select="rerun",
        selection_mode="points",
        key="product_embedding_scatter",
    )
    if chart.selection.points:
        point = chart.selection.points[0]
        if point.get("customdata"):
            product_id = point["customdata"][0]
            selected = embedding.loc[embedding["product_id"] == product_id]
            if not selected.empty:
                row = selected.iloc[0]
                st.session_state["embedding_selected_product"] = row["product_id"]
    selected_id = st.session_state.get("embedding_selected_product")
    if selected_id is not None:
        selected = embedding.loc[embedding["product_id"] == selected_id]
        if not selected.empty:
            row = selected.iloc[0]
            st.markdown(f"**Selected offer · {row['product_title']}**")
            st.caption(
                f"Product {row.get('product_id')} · {row.get('category_label')} · Merchant {row.get('merchant_id')} · "
                f"DBSCAN label {row.get('cluster_id_dbscan', 'not assigned in latest run')}"
            )


def render_model_evaluation():
    frame = _frame()
    model = _model()
    cluster = _cluster_state()
    page_header("Model evaluation", "Review genuine held-out classification results and the latest DBSCAN run.")
    if model:
        a, b, c, d = st.columns(4)
        for column, label, value in zip(
            (a, b, c, d),
            ("Accuracy", "Macro precision", "Macro recall", "Macro F1"),
            (model["accuracy"], model["precision"], model["recall"], model["f1"]),
        ):
            with column:
                metric_card(label, f"{value:.3f}")
        chart, perf = st.columns([1.1, 0.9], gap="large")
        with chart:
            section_title("Confusion matrix", "Counts from the stratified hold-out set.")
            selected = st.plotly_chart(
                confusion_matrix_chart(model),
                use_container_width=True,
                on_select="rerun",
                selection_mode="points",
                key="evaluation_confusion",
            )
            if selected.selection.points:
                point = selected.selection.points[0]
                examples = confusion_point_examples(model, str(point.get("y")), str(point.get("x")))
                st.dataframe(examples.head(10), hide_index=True, use_container_width=True)
        with perf:
            section_title("Per-category metrics", "Precision, recall and F1 per category.")
            st.plotly_chart(per_category_performance(model), use_container_width=True)
        pr = precision_recall_curve(model)
        if pr is not None:
            section_title("Precision–recall curves", "One-vs-rest scores computed from held-out Logistic Regression probabilities.")
            st.plotly_chart(pr, use_container_width=True)
        st.caption(
            f"Method: {model['n_train']:,} train / {model['n_test']:,} test records · "
            f"stratified random split · random_state={model['random_state']}."
        )
    else:
        empty_state(
            "Classification has not been evaluated",
            "Train the classifier to calculate test-set metrics and render the confusion matrix.",
            "Open classification lab",
            key="evaluation_train_link",
        )
        if st.session_state.get("evaluation_train_link"):
            _navigate("classification_lab")

    st.divider()
    section_title("DBSCAN statistics", "Cosine-distance clustering on actual character TF-IDF representations.")
    if cluster:
        stats = st.columns(4)
        for column, label, value in zip(
            stats,
            ("Analyzed offers", "Clusters", "Noise points", "Noise share"),
            (
                f"{cluster['sample_size']:,}",
                f"{len(cluster['cluster_counts']):,}",
                f"{cluster['noise_count']:,}",
                f"{cluster['noise_count'] / max(cluster['sample_size'], 1):.1%}",
            ),
        ):
            with column:
                metric_card(label, value)
        if cluster["silhouette"] is not None:
            st.caption(f"Silhouette (assigned products only, cosine distance): {cluster['silhouette']:.3f}")
        st.caption(f"Parameters: ε={cluster['eps']:.2f}, min_samples={cluster['min_samples']}; sample size={cluster['sample_size']:,}.")
    else:
        empty_state(
            "No clustering result",
            "DBSCAN metrics become available after running a clustering analysis.",
            "Open clustering studio",
            key="evaluation_cluster_link",
        )
        if st.session_state.get("evaluation_cluster_link"):
            _navigate("clustering_studio")
    if "cluster_id" in frame.columns:
        st.caption("The uploaded data contains Cluster ID reference groups. Pair-level checks are reported only for generated candidate sets in Entity Resolution.")


def render_dataset_explorer():
    frame = _frame()
    raw_columns = st.session_state["source_columns"]
    raw_frame = frame[raw_columns]
    page_header("Dataset quality", "Inspect the uploaded UCI records, schema, missingness, duplication, and product-description lengths.")
    a, b, c, d = st.columns(4)
    with a:
        metric_card("Records", f"{len(frame):,}")
    with b:
        metric_card("Original columns", f"{len(raw_columns)}")
    with c:
        metric_card("Exact duplicate rows", f"{raw_frame.duplicated().sum():,}")
    with d:
        metric_card("Missing cells", f"{int(raw_frame.isna().sum().sum()):,}")
    if raw_frame.isna().sum().sum():
        st.warning("Missing values are present in the uploaded data; normalized text fields treat missing descriptions as empty strings.")
    else:
        st.success("No missing values were found in the uploaded source columns.")
    if raw_frame.duplicated().sum():
        st.warning("Exact duplicate rows are present. The source records remain unchanged in the application.")
    else:
        st.info("No exact duplicate rows were found. Product-title repeats may still represent distinct merchant offers.")
    chart_col, schema_col = st.columns([1, 1], gap="large")
    with chart_col:
        section_title("Description lengths", "Character counts after consistent lower-case text normalization.")
        st.plotly_chart(text_length_distribution(frame), use_container_width=True)
    with schema_col:
        section_title("Source schema", "Original uploaded columns; derived helper columns are excluded.")
        schema = pd.DataFrame(
            {
                "Column": raw_columns,
                "Type": [str(raw_frame[column].dtype) for column in raw_columns],
                "Missing": [int(raw_frame[column].isna().sum()) for column in raw_columns],
                "Unique values": [int(raw_frame[column].nunique(dropna=True)) for column in raw_columns],
            }
        )
        st.dataframe(schema, hide_index=True, use_container_width=True)
    st.plotly_chart(category_distribution(frame), use_container_width=True)
    filters = st.columns(3)
    category = filters[0].selectbox("Export category", ["All categories", *sorted(frame["category_label"].unique())])
    merchant_values = ["All merchants", *sorted(frame["merchant_id"].astype(str).unique())] if "merchant_id" in frame else ["All merchants"]
    merchant = filters[1].selectbox("Export merchant", merchant_values)
    search = filters[2].text_input("Filter titles", key="dataset_title_filter")
    filtered = _filtered_products(frame, search, category, merchant, "product_title", True)
    st.caption(f"{len(filtered):,} filtered records shown.")
    preview_columns = [column for column in raw_columns if column in filtered.columns]
    st.dataframe(filtered[preview_columns].head(100), hide_index=True, use_container_width=True)
    st.download_button(
        "Export filtered source records",
        filtered[preview_columns].to_csv(index=False).encode("utf-8"),
        file_name="meridian_dataset_export.csv",
        mime="text/csv",
        key="dataset_export",
    )


def render_model_settings():
    page_header("Model settings", "Review model status and run a reproducible training configuration against the uploaded dataset.")
    model = _model()
    if model:
        st.success(
            f"Loaded model artifact matches this dataset · {model['training_rows']:,} training records · "
            f"macro F1 {model['f1']:.3f}."
        )
    else:
        st.info("No trained model artifact exists for the currently loaded dataset.")
    st.markdown("**Classifier** · word-level TF-IDF (1–2 grams) with balanced Logistic Regression.")
    st.markdown("**Evaluation** · stratified train/test split with accuracy, macro precision/recall/F1, per-class metrics and held-out probabilities.")
    st.markdown(f"**Artifact location** · `{st.session_state['artifact_path']}`")
    if st.button("Train classifier on current dataset", type="primary", key="settings_train_model"):
        try:
            result = _train_model(test_size=0.2, sample_limit=len(_frame()))
            st.success(f"Training completed with {result['training_rows']:,} records.")
        except Exception as exc:
            st.error(f"Training failed: {exc}")
    if st.button("Reload compatible model artifact", key="reload_artifact"):
        from src.classification import load_model_artifact

        try:
            result = load_model_artifact(
                st.session_state["dataset_fingerprint"], st.session_state["artifact_path"]
            )
            if result is None:
                st.warning("No compatible artifact was found for this dataset.")
            else:
                st.session_state["model_result"] = result
                st.success("Compatible model artifact loaded.")
        except Exception as exc:
            st.error(f"Artifact could not be loaded: {exc}")


PAGE_RENDERERS = {
    "overview": render_overview,
    "product_explorer": render_product_explorer,
    "clustering_studio": render_clustering_studio,
    "classification_lab": render_classification_lab,
    "entity_resolution": render_entity_resolution,
    "embedding_explorer": render_embedding_explorer,
    "model_evaluation": render_model_evaluation,
    "dataset_explorer": render_dataset_explorer,
    "model_settings": render_model_settings,
}


def render_page(page_name):
    PAGE_RENDERERS[page_name]()
