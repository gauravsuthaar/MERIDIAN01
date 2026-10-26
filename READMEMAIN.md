MERIDIAN AI

Product Classification, Clustering & Entity Resolution

A machine learning framework for analyzing product offers through text classification, density-based clustering, and entity matching.

MERIDIAN AI is a Python-based application designed to explore product catalogs, identify patterns in product descriptions, group similar listings, and discover potential duplicate products across merchants. It combines machine learning workflows with an interactive Streamlit interface and analytical visualizations.

⸻

Table of Contents

* Overview
* Problem Statement
* Project Objectives
* Dataset
* Core Features
* System Architecture
* Machine Learning Methodology
* Data Visualization
* Technology Stack
* Project Structure
* Installation
* Running the Application
* Application Modules
* Evaluation Metrics
* Limitations
* Future Enhancements
* Team Members
* References

⸻

Overview

Product catalogs often contain thousands of listings with inconsistent descriptions, different naming conventions, and repeated offers for equivalent products.

Identifying these relationships manually can be time-consuming and difficult to scale.

MERIDIAN AI addresses this challenge through three interconnected machine learning tasks:

1. Product Classification: Assign product offers to appropriate categories using their textual descriptions.
2. Product Clustering: Discover groups of similar product offers using text-derived feature representations.
3. Entity Matching: Identify candidate duplicate or equivalent products across different merchants.

The application provides an interactive environment for exploring the dataset, executing machine learning workflows, inspecting clustering results, comparing product offers, and evaluating model performance.

⸻

Problem Statement

Develop an intelligent, scalable and text-aware machine learning framework that automatically classifies and clusters product offers based on their short textual descriptions and subsequently performs entity matching to identify duplicate or equivalent products across different merchants.

⸻

Project Objectives

The main objectives of this project are:

* Analyze product offers using their textual descriptions.
* Preprocess and normalize product text for machine learning.
* Represent product descriptions as numerical feature vectors.
* Develop a text classification pipeline using available category labels.
* Apply DBSCAN to identify density-based clusters of similar product offers.
* Identify potential duplicate or equivalent product listings.
* Explore product similarities through interactive visualizations.
* Evaluate classification and clustering performance using appropriate metrics.
* Develop a usable interface for inspecting data and model results.
* Provide a modular foundation for future improvements in product intelligence.

⸻

Dataset

Dataset Name: Product Classification and Clustering

Author: L. Akritidis

Repository: UCI Machine Learning Repository

Year: 2020

Official Dataset Page: https://archive.ics.uci.edu/dataset/837/product+classification+and+clustering

DOI: https://doi.org/10.24432/C5M91Z

The project uses the specified UCI dataset as its primary data source.

The dataset contains product-offer information intended for product classification and clustering research. Its actual schema must be inspected before selecting the fields used by each machine learning task.

Dataset Preparation

The data preparation workflow may include:

* Loading the original dataset.
* Inspecting columns and data types.
* Handling missing or malformed values.
* Examining duplicate records.
* Identifying available product descriptions and category labels.
* Normalizing textual descriptions.
* Preparing features for classification and clustering.
* Preserving relevant identifiers and metadata for product inspection and matching.

Preprocessing decisions should be documented and applied consistently during training and inference.

The original dataset should remain unchanged.

Important: The fields available for merchant-level entity matching and ground-truth duplicate detection depend on the actual dataset schema. Matching capabilities and evaluation must not assume that unavailable identifiers, merchant fields, or labels exist.

⸻

Core Features

1. Overview Dashboard

The overview provides a summary of the available dataset and machine learning results.

Potential dashboard indicators include:

* Total product offers.
* Number of product categories.
* Number of detected clusters.
* Number of DBSCAN noise points.
* Candidate product matches.
* Model training and evaluation status.

Interactive visualizations summarize category distributions, clustering results, and matching analysis when the corresponding results are available.

All displayed statistics should be calculated from actual dataset contents or model outputs.

2. Product Explorer

The Product Explorer provides an interface for examining product listings.

Features include:

* Search product descriptions.
* Filter by available categories.
* Filter by merchant when merchant information exists.
* Sort available fields.
* Inspect individual product records.
* Review original and normalized descriptions.
* Explore relevant product metadata.
* Export filtered records to CSV.

The explorer supports data inspection and helps users understand the product catalog before running machine learning workflows.

3. Classification Lab

The Classification Lab supports product category prediction using textual descriptions.

The workflow consists of:

1. Preparing labeled product descriptions.
2. Splitting the data into training and evaluation sets.
3. Converting descriptions into numerical text features.
4. Training an appropriate classification algorithm.
5. Predicting categories for new descriptions.
6. Evaluating predictions using suitable metrics.
7. Exporting prediction results.

A TF-IDF-based classification pipeline can provide a suitable baseline for short product descriptions.

The classifier must be selected and evaluated according to the available labels and the characteristics of the dataset.

Confidence values should only be displayed when supported by the selected model and its implementation.

4. Clustering Studio

The Clustering Studio uses DBSCAN (Density-Based Spatial Clustering of Applications with Noise) to explore groups of similar product offers.

DBSCAN identifies dense regions in a feature space and labels points that do not belong to a qualifying cluster as noise.

The clustering workflow includes:

1. Preparing product descriptions.
2. Converting text into numerical representations.
3. Selecting an appropriate distance metric.
4. Configuring DBSCAN parameters.
5. Running the clustering algorithm.
6. Inspecting detected clusters and noise points.
7. Exploring the products belonging to each cluster.

Relevant parameters include:

* eps: Neighborhood radius.
* min_samples: Minimum number of samples required to form a dense region.

DBSCAN assigns the label -1 to noise points. These points should be analyzed separately from the detected clusters.

The selected text representation and distance metric must be appropriate for the feature space. High-dimensional sparse text vectors should not automatically be processed using an unsuitable distance configuration.

Where appropriate, K-Means may be used as a comparison baseline.

5. Entity Resolution

Entity Resolution identifies product listings that may represent the same or equivalent underlying product.

The workflow can include:

1. Selecting or entering product descriptions.
2. Normalizing textual content.
3. Generating candidate product pairs.
4. Calculating implemented similarity features.
5. Comparing scores against a configurable decision threshold.
6. Reviewing potential matches.
7. Accepting, rejecting, or flagging candidates when those controls are implemented.

Possible similarity techniques include:

* TF-IDF cosine similarity.
* Character n-gram similarity.
* Token overlap.
* Normalized text comparison.

The final matching strategy should be based on implemented features and evaluation results.

A similarity score is not automatically a calibrated probability that two products are identical.

Matching evaluation requires suitable ground-truth labels. If these are unavailable, the application should present candidate matching results without claiming validated matching accuracy.

6. Embedding Explorer

When product embeddings or suitable text vectors are available, the Embedding Explorer provides a visual representation of product relationships.

Possible capabilities include:

* Two-dimensional product projections.
* Interactive product selection.
* Category-based filtering.
* Cluster-based coloring.
* Product detail inspection.
* Exploration of nearby listings.

Dimensionality reduction techniques such as PCA or UMAP may be used when appropriate.

A two-dimensional projection is an approximation of the original feature space and should not be interpreted as a perfect representation of every product similarity.

7. Dataset Explorer

The Dataset Explorer supports inspection of the underlying data.

Potential capabilities include:

* Dataset dimensions.
* Column names and data types.
* Missing-value analysis.
* Duplicate-record analysis.
* Text-length distributions.
* Category distributions.
* Data previews.
* Filtered data exports.

These tools help identify data-quality issues and inform preprocessing decisions.

8. Model Evaluation

The Model Evaluation module provides task-specific performance analysis.

Classification evaluation

* Accuracy.
* Precision.
* Recall.
* F1 score.
* Confusion matrix.
* Per-category performance.

Clustering evaluation

* Number of detected clusters.
* Number of noise points.
* Noise ratio.
* Cluster-size distribution.
* Silhouette score where appropriate.

Entity matching evaluation

* Precision.
* Recall.
* F1 score.
* False positives.
* False negatives.
* Threshold-dependent performance.

Metrics must only be displayed when the required predictions, labels, and evaluation data are available.

⸻

System Architecture

MERIDIAN AI follows a modular Python-based architecture.

Data Layer

Responsible for:

* Loading the official dataset.
* Inspecting the schema.
* Handling missing values.
* Preparing product descriptions.
* Providing consistent data access.

Text Processing Layer

Responsible for:

* Text normalization.
* Feature extraction.
* TF-IDF vectorization or other selected text representations.
* Consistent transformation of training and inference data.

Machine Learning Layer

Contains separate workflows for:

* Product classification.
* DBSCAN clustering.
* Candidate generation and entity matching.
* Model evaluation.

Visualization Layer

Provides:

* Interactive analytical charts.
* Product tables.
* Cluster exploration.
* Classification evaluation.
* Similarity analysis.
* Matching inspection.

Application Layer

Streamlit provides the user interface and connects the data-processing, machine-learning, and visualization components.

⸻

Machine Learning Methodology

Text Representation

Product descriptions must be transformed into numerical representations before most machine learning algorithms can process them.

TF-IDF can represent the importance of terms within product descriptions relative to the collection of documents.

Depending on the task, character n-grams or sentence embeddings may also be evaluated.

The selected representation should be appropriate for the dataset and computational constraints.

Product Classification

A text classification pipeline can combine:

* Text preprocessing.
* TF-IDF feature extraction.
* A suitable classifier.
* A reproducible train/test split.
* Evaluation using appropriate classification metrics.

Potential baseline algorithms include Logistic Regression and Linear Support Vector Machines.

The final classifier should be selected based on actual experimental results rather than assumptions.

DBSCAN Clustering

DBSCAN groups observations according to density in a selected feature space.

Its principal parameters are:

* eps
* min_samples

The algorithm can discover clusters without requiring the number of clusters to be specified in advance.

It can also identify observations that do not belong to dense regions.

However, clustering performance depends on the feature representation, distance metric, parameter selection, and distribution of the data.

DBSCAN should therefore be evaluated using appropriate diagnostics rather than assuming that every resulting cluster is meaningful.

Entity Matching

Entity matching is distinct from clustering.

Clustering groups multiple observations, while entity matching evaluates whether individual product listings may refer to the same underlying entity.

A practical matching workflow can use candidate generation followed by similarity calculation and decision thresholds.

For larger datasets, candidate generation should reduce unnecessary pairwise comparisons rather than compare every possible pair indiscriminately.

Matching decisions should account for available product attributes and contradictory information wherever possible.

⸻

Data Visualization

The application uses interactive visualizations to make model results easier to inspect.

Depending on the available results, visualizations may include:

* Product category distribution.
* DBSCAN cluster-size distribution.
* Cluster noise analysis.
* Product embedding maps.
* Classification confusion matrices.
* Classification performance comparisons.
* Similarity-score distributions.
* Precision-recall analysis.
* Product relationship networks.

Visualizations should provide:

* Descriptive titles.
* Explanatory subtitles.
* Appropriate axis labels.
* Readable legends.
* Informative tooltips.
* Consistent styling.
* Clear handling of missing results.

Interactive controls should update the underlying displayed data rather than merely change the appearance of a chart.

Plotly is used for interactive visualizations where appropriate.

⸻

Technology Stack

Technology	Purpose
Python	Primary programming language
Streamlit	Application interface
Pandas	Dataset processing
NumPy	Numerical operations
Scikit-learn	Machine learning and evaluation
TF-IDF	Text feature extraction
DBSCAN	Density-based clustering
Plotly	Interactive visualizations
SciPy	Scientific computing where required
Joblib	Model artifact persistence where appropriate

Additional dependencies should only be included when required by the implementation.

⸻

Project Structure

The following is a suggested modular structure. The actual repository may differ.

MERIDIAN-AI/
├── app.py
├── pages/
│   ├── overview.py
│   ├── product_explorer.py
│   ├── classification_lab.py
│   ├── clustering_studio.py
│   ├── entity_resolution.py
│   ├── embedding_explorer.py
│   ├── model_evaluation.py
│   ├── dataset_explorer.py
│   └── model_settings.py
├── components/
│   ├── layout.py
│   ├── metrics.py
│   ├── charts.py
│   ├── tables.py
│   └── empty_states.py
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── classification.py
│   ├── clustering.py
│   ├── entity_matching.py
│   └── evaluation.py
├── data/
│   └── README.md
├── artifacts/
├── tests/
├── requirements.txt
└── README.md

Keep large datasets and generated artifacts out of version control unless there is a specific reason to include them.

⸻

Installation

Prerequisites

* Python 3.10 or a compatible version supported by the project’s dependencies.
* pip.
* Git, if cloning the repository.

1. Clone the Repository

If the project is hosted in a Git repository:

git clone <repository-url>
cd MERIDIAN-AI

Alternatively, open the existing project directory in your terminal.

2. Create a Virtual Environment

macOS / Linux

python3 -m venv .venv
source .venv/bin/activate

Windows PowerShell

python -m venv .venv
.venv\Scripts\Activate.ps1

3. Install Dependencies

python -m pip install --upgrade pip
pip install -r requirements.txt

Ensure requirements.txt contains the dependencies actually used by the application.

4. Obtain the Dataset

Download the official dataset from:

https://archive.ics.uci.edu/dataset/837/product+classification+and+clustering

Follow the repository’s download instructions and place the dataset in the location expected by the application’s data-loading configuration.

If the application expects a different path, update its configuration rather than renaming or modifying the original data without justification.

⸻

Running the Application

From the project root, activate the virtual environment and run:

streamlit run app.py

Streamlit will display the local application address in the terminal.

Open the displayed address in your browser.

Typical Workflow

1. Load and inspect the dataset.
2. Review the available columns and category labels.
3. Train the classification pipeline.
4. Run DBSCAN clustering.
5. Inspect detected clusters and noise points.
6. Compare candidate product offers.
7. Review model evaluation results.
8. Explore the visualizations and export supported results.

The exact workflow depends on the features implemented in the current version of the application.

⸻

Evaluation Metrics

Classification

Accuracy

The proportion of predictions that are correct.

Precision

The proportion of positive predictions that are correct for the relevant class.

Recall

The proportion of relevant positive examples successfully identified.

F1 Score

The harmonic mean of precision and recall.

Confusion Matrix

A table comparing actual labels against predicted labels.

For multiclass classification, the averaging method used for precision, recall and F1 should be documented.

Clustering

Number of Clusters

The number of distinct non-noise clusters identified by DBSCAN.

Noise Ratio

The proportion of observations assigned the DBSCAN noise label -1.

Silhouette Score

A clustering quality measure that compares within-cluster cohesion with separation from other clusters. Its use depends on the number of valid clusters and the suitability of the underlying distance representation.

Noise points should be handled explicitly when calculating clustering metrics.

Entity Matching

Precision

The proportion of predicted matches that are genuine matches.

Recall

The proportion of genuine matches that are successfully identified.

F1 Score

The harmonic mean of matching precision and recall.

These metrics require appropriate ground-truth matching labels. Without such labels, the system should report descriptive candidate-matching statistics rather than validated predictive performance.

⸻

Limitations

Potential limitations include:

* Product descriptions may be short, inconsistent, or ambiguous.
* Text similarity does not always establish product equivalence.
* DBSCAN can be sensitive to its distance metric and parameter choices.
* High-dimensional text representations can complicate density-based clustering.
* Some products may not form meaningful density-based clusters.
* Entity matching may require additional product attributes or ground-truth labels.
* Dimensionality-reduction plots approximate relationships in the original feature space.
* Computational cost may increase as the number of product offers grows.
* Matching thresholds may require validation against labeled examples.

These limitations should be refined according to the actual dataset, experimental results, and implemented functionality.

⸻

Future Enhancements

Potential future improvements include:

* Sentence-transformer embeddings for semantic product similarity.
* HDBSCAN for investigating clusters with varying density.
* More advanced candidate-generation methods.
* Attribute-aware product matching.
* Human-in-the-loop match review.
* Persistent storage for reviewed matching decisions.
* Model comparison and experiment tracking.
* Threshold calibration using validation data.
* More efficient similarity search for larger catalogs.
* Additional explainability tools.
* Improved handling of multilingual or noisy product descriptions.

Future work should be prioritized according to experimental evidence and project requirements.

⸻

Team Members

Group 7

Student ID	Name
2427010550	Gaurav Suthar
2427010462	Neel Sanjaykumar Patel
2427010521	Rishit Miglani

⸻

References

1. L. Akritidis, Product Classification and Clustering, UCI Machine Learning Repository, 2020.
    https://doi.org/10.24432/C5M91Z
2. Scikit-learn documentation.
    https://scikit-learn.org/stable/
3. Streamlit documentation.
    https://docs.streamlit.io/
4. Plotly Python documentation.
    https://plotly.com/python/

⸻

Project Note

MERIDIAN AI is an academic machine learning project focused on product classification, density-based clustering and entity matching using textual product descriptions.

All reported metrics, model predictions, cluster assignments and matching results should correspond to actual execution of the implemented pipeline.

The project should distinguish clearly between completed functionality, illustrative examples and proposed future enhancements.
