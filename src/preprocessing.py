import re


def normalize_text(text):
    if text is None:
        return ""
    text = str(text).lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def prepare_dataframe(frame):
    result = frame.copy()
    result.columns = [str(column).strip() for column in result.columns]
    columns = {column.casefold(): column for column in result.columns}

    title_column = next(
        (
            columns[key]
            for key in ("product title", "product_title", "product description", "description")
            if key in columns
        ),
        None,
    )
    if title_column is None:
        raise ValueError("The uploaded dataset has no product title/description column.")

    result["product_title"] = result[title_column].fillna("").astype(str)
    category_column = next(
        (columns[key] for key in ("category label", "category", "category_label") if key in columns),
        None,
    )
    result["category_label"] = (
        result[category_column].fillna("Unknown").astype(str)
        if category_column
        else "Unknown"
    )

    aliases = {
        "product_id": ("product id", "product_id", "id"),
        "merchant_id": ("merchant id", "merchant_id", "merchant"),
        "cluster_id": ("cluster id", "cluster_id"),
        "cluster_label": ("cluster label", "cluster_label"),
        "category_id": ("category id", "category_id"),
    }
    for output_column, possible_names in aliases.items():
        source_column = next((columns[name] for name in possible_names if name in columns), None)
        if source_column:
            result[output_column] = result[source_column]

    result["normalized_title"] = result["product_title"].map(normalize_text)
    result["title_length"] = result["normalized_title"].str.len()
    return result
