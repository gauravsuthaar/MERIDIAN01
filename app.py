from pathlib import Path

import streamlit as st

from components.layout import apply_global_theme, render_sidebar
from src.classification import load_model_artifact
from src.data_loader import dataset_fingerprint, find_dataset_path, load_dataset
from src.preprocessing import prepare_dataframe


st.set_page_config(
    page_title="MERIDIAN · Product Intelligence",
    page_icon="M",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_global_theme()

ARTIFACT_PATH = Path(__file__).resolve().parent / ".meridian" / "artifacts" / "classifier.joblib"
dataset_path = find_dataset_path()
if dataset_path is None:
    st.title("MERIDIAN")
    st.error(
        "The uploaded dataset could not be found. Place `pricerunner_aggregate.csv` "
        "on the Desktop or set `MERIDIAN_DATA_PATH` to the uploaded CSV/ZIP."
    )
    st.stop()

try:
    fingerprint = dataset_fingerprint(dataset_path)
    if (
        st.session_state.get("dataset_fingerprint") != fingerprint
        or "dataset_df" not in st.session_state
    ):
        raw_frame = load_dataset(dataset_path)
        prepared_frame = prepare_dataframe(raw_frame)
        st.session_state["dataset_df"] = prepared_frame
        st.session_state["source_columns"] = list(raw_frame.columns)
        st.session_state["dataset_path"] = str(dataset_path)
        st.session_state["dataset_fingerprint"] = fingerprint
        st.session_state.pop("model_result", None)
        st.session_state.pop("cluster_result", None)
        st.session_state.pop("candidate_matches", None)
        st.session_state.pop("product_embedding", None)
    st.session_state["artifact_path"] = str(ARTIFACT_PATH)
except Exception as exc:
    st.title("MERIDIAN")
    st.error(f"Dataset loading failed: {exc}")
    st.caption("The application does not substitute synthetic or unrelated data.")
    st.stop()

if "model_result" not in st.session_state:
    try:
        st.session_state["model_result"] = load_model_artifact(
            st.session_state["dataset_fingerprint"], ARTIFACT_PATH
        )
    except Exception as exc:
        st.session_state["model_result"] = None
        st.session_state["artifact_load_error"] = str(exc)

frame = st.session_state["dataset_df"]
render_sidebar(
    Path(st.session_state["dataset_path"]).name,
    len(frame),
    frame["category_label"].nunique(),
    st.session_state.get("model_result") is not None,
)

navigation = st.navigation(
    {
        "Workspace": [
            st.Page("pages/overview.py", title="Overview", icon=":material/dashboard:", default=True),
            st.Page("pages/product_explorer.py", title="Product explorer", icon=":material/search:"),
            st.Page("pages/entity_resolution.py", title="Entity resolution", icon=":material/compare_arrows:"),
        ],
        "Modeling": [
            st.Page("pages/classification_lab.py", title="Classification lab", icon=":material/category:"),
            st.Page("pages/clustering_studio.py", title="Clustering studio", icon=":material/hub:"),
            st.Page("pages/embedding_explorer.py", title="Embedding explorer", icon=":material/scatter_plot:"),
        ],
        "Data & evaluation": [
            st.Page("pages/model_evaluation.py", title="Model evaluation", icon=":material/analytics:"),
            st.Page("pages/dataset_explorer.py", title="Dataset explorer", icon=":material/table_view:"),
            st.Page("pages/model_settings.py", title="Model settings", icon=":material/settings:"),
        ],
    }
)
navigation.run()
