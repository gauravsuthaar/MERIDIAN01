import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


PALETTE = ["#31584d", "#78917f", "#aab9a9", "#c18468", "#777d77", "#d0b28b"]


def style_figure(fig, height=350):
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=18, r=20, t=16, b=28),
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0.25)",
        font=dict(family="Inter, ui-sans-serif, system-ui, sans-serif", color="#343933", size=12),
        hoverlabel=dict(bgcolor="#fff", bordercolor="#deded6", font_color="#252923"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        transition=dict(duration=260, easing="cubic-in-out"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eeede7", zerolinecolor="#ddddd4", showline=True, linecolor="#e5e4dd")
    fig.update_yaxes(showgrid=False, showline=True, linecolor="#e5e4dd")
    fig.update_layout(
        xaxis_title_font=dict(color="#74776f"),
        yaxis_title_font=dict(color="#74776f"),
        xaxis_tickfont=dict(color="#74776f"),
        yaxis_tickfont=dict(color="#74776f"),
    )
    return fig


def category_distribution(frame, selected_category=None):
    counts = frame["category_label"].value_counts().rename_axis("Category").reset_index(name="Offers")
    counts = counts.sort_values("Offers", ascending=True)
    fig = px.bar(
        counts,
        x="Offers",
        y="Category",
        orientation="h",
        color_discrete_sequence=[PALETTE[0]],
        custom_data=["Offers"],
    )
    fig.update_traces(
        marker_line_color="#fff",
        marker_line_width=1,
        opacity=0.9,
        hovertemplate="<b>%{y}</b><br>Product offers: %{x:,}<extra></extra>",
    )
    if selected_category and selected_category in counts["Category"].values:
        fig.update_traces(
            marker_color=[
                "#31584d" if category == selected_category else "#b8c5bc"
                for category in counts["Category"]
            ]
        )
    fig.update_layout(showlegend=False, xaxis_title="Product offers", yaxis_title=None)
    return style_figure(fig, max(300, 30 * len(counts) + 90))


def cluster_distribution(cluster_frame):
    counts = (
        cluster_frame.loc[cluster_frame["cluster_id_dbscan"] != -1, "cluster_id_dbscan"]
        .value_counts()
        .rename_axis("Cluster")
        .reset_index(name="Products")
        .sort_values("Products", ascending=True)
    )
    fig = px.bar(
        counts,
        x="Products",
        y="Cluster",
        orientation="h",
        color_discrete_sequence=[PALETTE[0]],
        custom_data=["Products"],
    )
    fig.update_traces(
        marker_line_color="#fff",
        marker_line_width=1,
        hovertemplate="Cluster %{y}<br>Products: %{x:,}<extra></extra>",
    )
    fig.update_layout(showlegend=False, xaxis_title="Assigned product offers", yaxis_title=None)
    return style_figure(fig, max(300, min(760, 24 * len(counts) + 80)))


def cluster_noise_donut(assigned, noise):
    fig = px.pie(
        pd.DataFrame({"Assignment": ["Assigned to a cluster", "Noise"], "Products": [assigned, noise]}),
        names="Assignment",
        values="Products",
        hole=0.68,
        color="Assignment",
        color_discrete_map={"Assigned to a cluster": "#31584d", "Noise": "#c18468"},
    )
    fig.update_traces(
        textinfo="percent",
        textposition="inside",
        hovertemplate="%{label}<br>%{value:,} offers · %{percent}<extra></extra>",
        marker_line_color="#fff",
        marker_line_width=2,
    )
    fig.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.1, x=0))
    return style_figure(fig, 310)


def confusion_matrix_chart(model_result):
    labels = model_result["labels"]
    matrix = model_result["confusion_matrix"]
    fig = px.imshow(
        matrix,
        x=labels,
        y=labels,
        aspect="auto",
        color_continuous_scale=[[0, "#f1f3ef"], [0.55, "#a6b9ad"], [1, "#31584d"]],
        labels={"x": "Predicted category", "y": "Actual category", "color": "Test offers"},
    )
    fig.update_traces(
        text=matrix,
        texttemplate="%{text}",
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Test offers: %{z:,}<extra></extra>",
    )
    fig.update_layout(
        xaxis_tickangle=-35,
        coloraxis_colorbar=dict(title="Offers", outlinecolor="#e5e4dd", outlinewidth=1),
    )
    return style_figure(fig, max(390, 34 * len(labels) + 170))


def per_category_performance(model_result):
    rows = []
    for label in model_result["labels"]:
        metrics = model_result["report"][label]
        rows.append(
            {
                "Category": label,
                "Metric": "Precision",
                "Score": metrics["precision"],
            }
        )
        rows.append({"Category": label, "Metric": "Recall", "Score": metrics["recall"]})
        rows.append({"Category": label, "Metric": "F1", "Score": metrics["f1-score"]})
    fig = px.bar(
        pd.DataFrame(rows),
        x="Score",
        y="Category",
        color="Metric",
        barmode="group",
        orientation="h",
        color_discrete_map={"Precision": "#31584d", "Recall": "#91a394", "F1": "#c18468"},
        hover_data={"Score": ":.3f"},
    )
    fig.update_traces(
        marker_line_color="#fff",
        marker_line_width=0.7,
        hovertemplate="%{y}<br>%{fullData.name}: %{x:.3f}<extra></extra>",
    )
    fig.update_layout(xaxis_title="Score (0–1)", yaxis_title=None, xaxis_range=[0, 1.04])
    return style_figure(fig, max(360, 32 * len(model_result["labels"]) + 100))


def matching_distribution(candidates, threshold):
    scores = [candidate["score"] for candidate in candidates]
    frame = pd.DataFrame({"Similarity score": scores})
    fig = px.histogram(
        frame,
        x="Similarity score",
        nbins=18,
        color_discrete_sequence=[PALETTE[0]],
        opacity=0.86,
    )
    fig.update_traces(
        marker_line_color="#fff",
        marker_line_width=1,
        hovertemplate="Similarity: %{x:.2f}<br>Candidate pairs: %{y}<extra></extra>",
    )
    fig.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="#c18468",
        annotation_text=f"Review threshold {threshold:.2f}",
        annotation_position="top",
    )
    fig.update_layout(showlegend=False, xaxis_title="Heuristic similarity score (0–1)", yaxis_title="Candidate pairs")
    return style_figure(fig, 320)


def product_embedding_map(frame):
    color_column = "category_label" if "category_label" in frame else None
    fig = px.scatter(
        frame,
        x="embedding_x",
        y="embedding_y",
        color=color_column,
        color_discrete_sequence=PALETTE,
        opacity=0.72,
        custom_data=[
            column
            for column in ["product_id", "product_title", "category_label", "merchant_id", "cluster_id_dbscan"]
            if column in frame.columns
        ],
    )
    fig.update_traces(
        marker=dict(size=7, line=dict(width=0.5, color="rgba(255,255,255,.8)")),
        hovertemplate="<b>%{customdata[1]}</b><br>Category: %{customdata[2]}<br>Merchant: %{customdata[3]}<br>Product ID: %{customdata[0]}<extra></extra>",
        selected=dict(marker=dict(size=12, opacity=1, color="#c18468")),
        unselected=dict(marker=dict(opacity=0.3)),
    )
    fig.update_layout(
        xaxis_title="Text embedding dimension 1",
        yaxis_title="Text embedding dimension 2",
        legend_title="Category",
        dragmode="select",
    )
    return style_figure(fig, 570)


def precision_recall_curve(model_result):
    from sklearn.metrics import precision_recall_curve

    y_true = pd.Series(model_result["y_test"]).astype(str)
    labels = model_result["labels"]
    probabilities = model_result.get("probabilities")
    if probabilities is None or len(labels) != probabilities.shape[1]:
        return None
    curves = []
    for index, label in enumerate(labels):
        binary = (y_true.to_numpy() == label).astype(int)
        if binary.sum() == 0 or binary.sum() == len(binary):
            continue
        precision, recall, thresholds = precision_recall_curve(binary, probabilities[:, index])
        for i, (p, r) in enumerate(zip(precision, recall)):
            curves.append(
                {
                    "Category": label,
                    "Recall": r,
                    "Precision": p,
                    "Threshold": thresholds[i] if i < len(thresholds) else 1.0,
                }
            )
    if not curves:
        return None
    fig = px.line(
        pd.DataFrame(curves),
        x="Recall",
        y="Precision",
        color="Category",
        color_discrete_sequence=PALETTE,
        custom_data=["Threshold"],
    )
    fig.update_traces(
        hovertemplate="%{fullData.name}<br>Recall: %{x:.3f}<br>Precision: %{y:.3f}<br>Threshold: %{customdata[0]:.3f}<extra></extra>",
        line=dict(width=2),
    )
    fig.update_layout(xaxis_title="Recall", yaxis_title="Precision", xaxis_range=[0, 1], yaxis_range=[0, 1])
    return style_figure(fig, 390)


def text_length_distribution(frame):
    fig = px.histogram(
        frame,
        x="title_length",
        nbins=35,
        color_discrete_sequence=[PALETTE[1]],
        labels={"title_length": "Normalized description length (characters)", "count": "Offers"},
    )
    fig.update_traces(
        marker_line_color="#fff",
        marker_line_width=1,
        hovertemplate="Description length: %{x} characters<br>Offers: %{y:,}<extra></extra>",
    )
    fig.update_layout(showlegend=False, xaxis_title="Normalized description length (characters)", yaxis_title="Product offers")
    return style_figure(fig, 320)


def confusion_point_examples(model_result, actual, predicted):
    y_true = pd.Series(model_result["y_test"]).astype(str)
    y_pred = pd.Series(model_result["predictions"]).astype(str)
    mask = (y_true == actual) & (y_pred == predicted)
    return pd.DataFrame(
        {
            "Product description": model_result["X_test"][mask].values,
            "Actual category": y_true[mask].values,
            "Predicted category": y_pred[mask].values,
        }
    )
