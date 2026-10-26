import hashlib
import os
from pathlib import Path
from zipfile import ZipFile

import pandas as pd


UPLOADED_DATASET = Path("/Users/gauravsuthar/Desktop/pricerunner_aggregate.csv")


def default_dataset_candidates():
    env_path = os.getenv("MERIDIAN_DATA_PATH")
    candidates = [Path(env_path)] if env_path else []
    candidates.extend(
        [
            UPLOADED_DATASET,
            Path.cwd() / "data" / "pricerunner_aggregate.csv",
            Path.cwd() / "data" / "product+classification+and+clustering.zip",
            Path("/Users/gauravsuthar/Desktop/product+classification+and+clustering.zip"),
        ]
    )
    return candidates


def find_dataset_path():
    for candidate in default_dataset_candidates():
        if candidate.is_file():
            return candidate
    return None


def dataset_fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as dataset_file:
        for chunk in iter(lambda: dataset_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset(path=None):
    dataset_path = Path(path) if path else find_dataset_path()
    if dataset_path is None:
        raise FileNotFoundError(
            "Dataset not found. The uploaded file should be "
            f"{UPLOADED_DATASET}, or set MERIDIAN_DATA_PATH to a CSV/ZIP file."
        )

    suffix = dataset_path.suffix.lower()
    if suffix == ".zip":
        with ZipFile(dataset_path) as archive:
            members = [member for member in archive.namelist() if member.lower().endswith(".csv")]
            if not members:
                raise ValueError(f"No CSV file found in dataset archive: {dataset_path}")
            with archive.open(members[0]) as source:
                frame = pd.read_csv(source)
    elif suffix == ".csv":
        frame = pd.read_csv(dataset_path)
    else:
        raise ValueError(f"Unsupported dataset file type: {suffix or '(no extension)'}")

    if frame.empty:
        raise ValueError(f"The dataset contains no records: {dataset_path}")
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame
