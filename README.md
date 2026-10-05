[🇬🇧 English](README.md) | [🇪🇸 Español](README.es.md)

# AutoML Classification Engine & Model Governance Pipeline

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NLTK](https://img.shields.io/badge/NLTK-NLP-046A38?style=for-the-badge)](https://www.nltk.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

> **Decoupled, configuration-driven AutoML framework featuring automated model governance, dynamic hyperparameter optimization, and champion-challenger lifecycle registry.**

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Engineering Highlights](#-engineering-highlights)
- [System Architecture](#-system-architecture)
- [Project Structure](#-project-structure)
- [Algorithms & Preprocessing Matrix](#-algorithms--preprocessing-matrix)
- [`configuration.json` Specification](#-configurationjson-specification)
- [Installation & Setup](#-installation--setup)
- [Quickstart CLI Guide](#-quickstart-cli-guide)
- [Metrics, Governance & Traceability](#-metrics-governance--traceability)
- [Technical Maturity & Future Extensions](#-technical-maturity--future-extensions)
- [Authors & Attribution](#-authors--attribution)
- [License](#-license)

---

## 🚀 Overview

**AutoML Classification Engine** is an end-to-end Machine Learning Engineering pipeline designed to bridge the gap between ad-hoc notebook experimentation and production-ready model governance.

Built with complete separation between execution logic and hyperparameter definitions via a declarative `configuration.json` file, the framework orchestrates:
1. **Dynamic Preprocessing:** Adaptive text/tabular transformations (Tokenization, Lemmatization, BoW, TF-IDF, Rescaling).
2. **Exhaustive Hyperparameter Sweeps:** Parallelized Grid Search across multiple algorithm families.
3. **Champion-Challenger Tournament:** Evaluates candidate models against an immutable validation benchmark. Only candidates outperforming the active production model are promoted to `best_model/`, while historical iterations are archived in `archivo_versiones/` for full compliance and reproducibility.
4. **Out-of-Sample Inference:** Standalone evaluation pipeline (`test.py`) with zero data leakage, computing multi-class confusion matrices and performance metrics.

---

## 💡 Engineering Highlights

* ⚙️ **100% Declarative Configuration:** Zero hardcoded hyperparameters in Python scripts. Changing algorithms, grid parameters, or feature scalers requires only editing `configuration.json`.
* 🛡️ **Automated Model Governance:** Implements an enterprise champion-challenger pattern. If a newly trained model fails to exceed the current macro F1-score record, the system rejects production deployment, preserving rollback safety.
* 📦 **Atomic Artifact Bundling:** Serializes trained models alongside their fitted preprocessor pipelines (`bestmodel.sav`, `bestmodel_preproc.sav`) using `pickle`, ensuring absolute reproducibility in downstream inference.
* 📊 **Audit Trail Logging:** Every experiment automatically appends timestamped executions, preprocessing flags, and evaluation metrics to `ultimos_resultados.csv`.
* 🔒 **Data Leakage Prevention:** Scalers and vectorizers are fitted strictly on training folds and subsequently applied to validation/test sets without target contamination.

---

## 📐 System Architecture

```text
               ┌───────────────────────┐
               │  configuration.json   │
               └──────────┬────────────┘
                          │ (Defines grids, scalers & targets)
                          ▼
┌──────────────┐     ┌───────────┐     ┌───────────────────────────────────┐
│ Training CSV │ ──> │ train.py  │ ──> │ Grid Search & Evaluation Pipeline │
└──────────────┘     └─────┬─────┘     └─────────────────┬─────────────────┘
                           │                             │
                           │                    (Evaluates Macro F1)
                           │                             │
                           │                             ▼
                           │             ┌───────────────────────────────┐
                           │             │ Did it beat production record? │
                           │             └───────┬───────────────┬───────┘
                           │                     │ YES           │ NO
                           │                     ▼               ▼
                           │             ┌───────────────┐ ┌───────────────────┐
                           │             │  best_model/  │ │ archivo_versiones/│
                           │             └───────┬───────┘ └───────────────────┘
                           │                     │ (Deploy)
                           ▼                     ▼
┌──────────────┐     ┌───────────┐     ┌───────────────────────────────┐
│ Test Set CSV │ ──> │  test.py  │ ──> │ Metrics & Confusion Matrix    │
└──────────────┘     └───────────┘     └───────────────────────────────┘
```

---

## 📂 Project Structure

```text
automl-classification-engine/
├── configuration.json         # Declarative AutoML schema & hyperparameter grids
├── requirements.txt           # Production dependencies manifest
├── train.py                   # Automated training, grid search & governance engine
├── test.py                    # Inference pipeline & production evaluation script
├── best_model/                # Active production champion model & preprocessor
│   ├── bestmodel.sav
│   └── bestmodel_preproc.sav
├── archivo_versiones/         # Historical challenger models & audit snapshots
├── ultimos_resultados.csv     # Immutable experimentation audit log
├── .gitignore                 # Clean Git exclusions
└── LICENSE                    # MIT License
```

---

## 📊 Algorithms & Preprocessing Matrix

### Supported Classifiers
* **K-Nearest Neighbors (KNN):** Distance metrics (`minkowski`, `manhattan`, `euclidean`), weights (`uniform`, `distance`), variable neighbors $K \in [3, 11]$.
* **Decision Trees (CART):** Splitting criteria (`gini`, `entropy`), maximum depths, minimum sample splits.
* **Random Forest:** Ensemble bagging sweeps, estimator counts, bootstrapping, maximum feature subsets.
* **Naive Bayes:** Gaussian, Multinomial, and Complement variants suited for high-dimensional sparse representations.
* **Logistic Regression:** Regularization penalties ($L_1$, $L_2$, ElasticNet), solver variants (`lbfgs`, `saga`), and tolerance parameters.

### Integrated Feature Transformations
* **NLP & Text Vectorization:** Bag of Words (BoW / CountVectorizer), Term Frequency-Inverse Document Frequency (TF-IDF), Lemmatization via NLTK WordNet, English stopword filtration.
* **Numeric Scaling & Normalization:** `StandardScaler` (z-score), `MinMaxScaler` ($[0, 1]$), `RobustScaler` (quantile-based outlier resistance).

---

## ⚙️ `configuration.json` Specification

```json
{
  "dataset": {
    "train_path": "train.csv",
    "target_column": "Label",
    "text_column": "Review"
  },
  "preprocessing": {
    "vectorizer": "tfidf",
    "max_features": 5000,
    "ngram_range": [1, 2],
    "use_lemmatization": true,
    "scaler": "standard"
  },
  "automl": {
    "cv_folds": 5,
    "scoring_metric": "f1_macro",
    "algorithms": ["random_forest", "knn", "naive_bayes"]
  }
}
```

---

## 🛠️ Installation & Setup

### 1. Clone repository & create virtual environment:
```bash
git clone https://github.com/aimarlarriba/automl-classification-engine.git
cd automl-classification-engine

python -m venv venv

# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

---

## 💻 Quickstart CLI Guide

### 1. Run AutoML Training & Tournament:
```bash
python train.py
```
*Parses `configuration.json`, executes parallelized Grid Search, benchmarks candidates, updates `best_model/` upon new record, and appends results to `ultimos_resultados.csv`.*

### 2. Run Production Inference & Evaluation:
```bash
python test.py
```
*Loads the active champion from `best_model/`, transforms unseen test instances without data leakage, and outputs classification reports and confusion matrices.*

---

## 📈 Metrics & Traceability

```text
======================================================
CHAMPION MODEL EVALUATION SUMMARY
======================================================
Algorithm:         RandomForestClassifier
Macro F1-Score:    0.8924
Weighted Accuracy: 0.9015
Log Loss:          0.2310
Status:            Active Production Champion (Promoted)
======================================================
```

---

## 👥 Authors & Attribution

Developed by **[Aimar Larriba](https://github.com/aimarlarriba)**. 
Originally conceptualized within the **Decision Support Systems (SAD)** curriculum at the **University of the Basque Country (UPV/EHU)**, refactored and maintained as an open-source model governance pipeline.

---

## 📜 License

Distributed under the **MIT** License. See [LICENSE](LICENSE) for more details.
