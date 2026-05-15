# Credit Risk Prediction — Technical Documentation

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Installation & Setup](#3-installation--setup)
4. [Dataset](#4-dataset)
5. [Notebook Workflow](#5-notebook-workflow)
   - 5.1 [Column Audit & Feature Engineering (EDA)](#51-column-audit--feature-engineering-eda)
   - 5.2 [Preprocessing Pipeline](#52-preprocessing-pipeline)
   - 5.3 [Model Training & Evaluation](#53-model-training--evaluation)
   - 5.4 [Hyperparameter Tuning (Optuna)](#54-hyperparameter-tuning-optuna)
   - 5.5 [Saving Artifacts](#55-saving-artifacts)
6. [Custom Transformers (`transformers.py`)](#6-custom-transformers-transformerspy)
7. [Streamlit Dashboard (`app2.py`)](#7-streamlit-dashboard-app2py)
8. [Model Performance](#8-model-performance)
9. [Design Decisions](#9-design-decisions)
10. [Limitations](#10-limitations)
11. [Future Work](#11-future-work)

---

## 1. Project Overview

**Goal:** Predict which loan applicants will fail to repay their debt (binary classification: default = 1, no default = 0).

**Dataset:** [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) — a real-world consumer lending dataset from Home Credit Group, an international consumer finance provider. This project uses only the primary table (`application_train.csv`).

**Business context:** Home Credit serves borrowers who are largely underserved by traditional banks — people with little or no credit history. Approving a bad loan is costly; rejecting a good applicant loses a customer. A well-calibrated default probability model allows the lender to make smarter, data-driven decisions at the application stage.

**Key outcomes:**
- Full leak-proof scikit-learn `Pipeline` covering all preprocessing steps
- Tuned LightGBM classifier with AUC = 0.7710, surpassing the widely cited single-table baseline of 0.7526
- Feature set reduced from 153 → 29 features with no meaningful performance loss
- Interactive Streamlit dashboard for EDA, model comparison, and real-time inference

---

## 2. Repository Structure

```
credit-risk-prediction/
│
├── data/
│   └── application_train.csv        # Raw dataset (not committed — download from Kaggle)
│
├── models/
│   ├── lgbm_tuned_v2.pkl            # Serialized full pipeline (joblib)
│   ├── best_params.json             # Best hyperparameters from Optuna
│   ├── best_threshold.json          # Optimal classification threshold (0.665)
│   ├── raw_columns.json             # List of feature columns fed to the pipeline
│   ├── model_metrics.json           # CV and test metrics for all benchmarked models
│   └── importance_df.csv            # Feature importance table from LightGBM baseline
│
├── credit_risk_main2.ipynb          # End-to-end ML workflow
├── app2.py                          # Streamlit multi-page dashboard
├── transformers.py                  # All custom sklearn transformers
└── README.md
```

---

## 3. Installation & Setup

### Prerequisites

- Python 3.9+
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/Niqar/Credit-risk-prediction.git
cd credit-risk-prediction

# 2. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download application_train.csv from Kaggle
#    https://www.kaggle.com/competitions/home-credit-default-risk/data
#    Place it at: data/application_train.csv

# 5. Run the notebook to train the model and generate all artifacts
jupyter notebook credit_risk_main2.ipynb

# 6. Launch the dashboard
streamlit run app2.py
```

### `requirements.txt`

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
lightgbm>=4.0
imbalanced-learn>=0.11
optuna>=3.0
streamlit>=1.30
plotly>=5.18
seaborn>=0.12
matplotlib>=3.7
missingno>=0.5
scipy>=1.11
joblib>=1.3
gdown
```

> **Important:** `transformers.py` must be in the same directory as `app2.py` so that `joblib.load()` can deserialize the custom transformers embedded in the pipeline.

---

## 4. Dataset

| Property | Value |
|----------|-------|
| File | `application_train.csv` |
| Rows | 307,511 |
| Original columns | 122 |
| Target variable | `TARGET` (1 = defaulted, 0 = repaid) |
| Default rate | 8.07% (heavily imbalanced) |
| Source | [Kaggle — Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) |

### Target Distribution

| Class | Count | Percentage |
|-------|-------|-----------|
| 0 — No Default | 282,686 | 91.93% |
| 1 — Default | 24,825 | 8.07% |

The strong class imbalance is handled in the model via `scale_pos_weight = count(0) / count(1) ≈ 11.4`.

---

## 5. Notebook Workflow

The notebook `credit_risk_main2.ipynb` follows a structured six-step workflow.

### 5.1 Column Audit & Feature Engineering (EDA)

Before any modelling, 80+ columns were dropped with documented reasons:

**Replaced by engineered features:**

| Dropped Columns | Replaced By |
|----------------|-------------|
| `FLAG_DOCUMENT_2` through `FLAG_DOCUMENT_21` | `TOTAL_DOCUMENTS` — sum of all document flags. More documents submitted = more transparent applicant = lower risk. |
| `AMT_REQ_CREDIT_BUREAU_HOUR/DAY/WEEK/MON/QRT/YEAR` | `TOTAL_ENQUIRIES` — sum of all credit bureau enquiry columns. High enquiry count suggests desperation for credit = higher risk. |

**Dropped for no predictive value:**

| Column | Reason |
|--------|--------|
| `SK_ID_CURR` | Loan ID only — no predictive signal |
| `FLAG_MOBIL` | 99.9997% of applicants have a mobile — zero variance |
| `FLAG_CONT_MOBILE` | 99.8% reachable by mobile — near-zero variance |
| `FLAG_EMAIL` | Only 5.7% have a registered email — insufficient variation |
| `REG_REGION_NOT_LIVE_REGION` | 98.5% are 0 — near-zero variance |
| `LIVE_REGION_NOT_WORK_REGION` | Only 4.1% live in a different region from work |
| `REG_REGION_NOT_WORK_REGION` | Only 5.1% registered in a different region from work |

**Dropped as redundant:**

| Column | Reason |
|--------|--------|
| `REGION_RATING_CLIENT_W_CITY` | Duplicate of `REGION_RATING_CLIENT` |
| `OWN_CAR_AGE` | 66% missing values; already captured by `FLAG_OWN_CAR` |

**Dropped (47 building info columns):**
All `APARTMENTS_*`, `BASEMENTAREA_*`, `YEARS_BEGINEXPLUATATION_*`, `YEARS_BUILD_*`, `COMMONAREA_*`, `ELEVATORS_*`, `ENTRANCES_*`, `FLOORSMAX_*`, `FLOORSMIN_*`, `LANDAREA_*`, `LIVINGAPARTMENTS_*`, `LIVINGAREA_*`, `NONLIVINGAPARTMENTS_*`, `NONLIVINGAREA_*`, `FONDKAPREMONT_MODE`, `HOUSETYPE_MODE`, `TOTALAREA_MODE`, `WALLSMATERIAL_MODE`, `EMERGENCYSTATE_MODE`.
Reason: Very high missingness and weak connection to credit risk.

After the column audit, column names were lowercased for consistency.

### 5.2 Preprocessing Pipeline

All preprocessing is implemented as a single scikit-learn `Pipeline` to guarantee no data leakage between train and test sets.

**Pipeline v2 (`build_pipeline_v2`) step order:**

```
Input DataFrame (raw features)
    │
    ▼
1.  FeatureEngineer              — Creates ratio and cyclical features
    │
    ▼
2.  InvalidValueCleaner          — Replaces placeholder strings (XNA, Unknown, etc.) with NaN
    │
    ▼
3.  DtypeFixer                   — Converts non-negative float columns with integer values to Int64
    │
    ▼
4.  MissingIndicator             — Adds binary _missing flags for columns where missingness is
    │                              significantly associated with the target (chi-squared, p < 0.05)
    ▼
5.  CorrelatedFeatureDropper     — Drops one of each pair of features with |corr| ≥ 0.8
    │
    ▼
6.  RareCategoryMerger           — Merges rare categories (< 4% frequency, default rate below
    │                              average, low spread) into 'other'
    ▼
7.  ColumnTransformer            — Applies:
    │    • Numerical branch: SimpleImputer(median) → OutlierCapper(1%–99%) →
    │                         Log1pTransformer → RobustScaler
    │    • Categorical branch: SimpleImputer(most_frequent) → OrdinalEncoder
    │                         (for ordered features) / OneHotEncoder (for nominal)
    ▼
8.  Top90PercentImportanceSelector  — Fits a LightGBM, keeps features whose cumulative
    │                                 importance accounts for top 90% of total importance
    ▼
9.  LGBMClassifier               — Final estimator
```

### 5.3 Model Training & Evaluation

Four classifiers were benchmarked using 5-fold Stratified Cross-Validation:

| Model | CV AUC Mean | CV AUC Std | Test AUC | Test PR-AUC | F1 |
|-------|:-----------:|:----------:|:--------:|:-----------:|:--:|
| Logistic Regression | — | — | — | — | — |
| Random Forest | — | — | — | — | — |
| XGBoost | — | — | — | — | — |
| **LightGBM** | **0.7603** | — | **0.7679** | **0.2553** | **0.2911** |

LightGBM was selected for tuning based on highest CV AUC and Test AUC.

Evaluation metrics used:
- **AUC-ROC** — primary metric; measures discrimination ability across all thresholds
- **PR-AUC** — important for imbalanced datasets; measures precision-recall trade-off
- **F1** — harmonic mean of precision and recall at the chosen threshold
- **Precision** — fraction of flagged applicants who actually defaulted
- **Recall** — fraction of actual defaulters caught by the model

### 5.4 Hyperparameter Tuning (Optuna)

Tuning was performed using [Optuna](https://optuna.org/) with a **gap-penalised objective** to reduce overfitting:

```python
objective = mean_val_auc - GAP_PENALTY * mean(train_auc - val_auc)
```

This penalises high train/val gaps, rewarding generalization over raw training score.

**Search space:**

| Parameter | Range |
|-----------|-------|
| `n_estimators` | 500 – 2000 |
| `learning_rate` | 0.01 – 0.1 (log scale) |
| `num_leaves` | 20 – 63 |
| `max_depth` | 3 – 6 |
| `min_child_samples` | 100 – 500 |
| `subsample` | 0.6 – 1.0 |
| `colsample_bytree` | 0.6 – 1.0 |
| `reg_alpha` | 1e-4 – 10.0 (log scale) |
| `reg_lambda` | 1e-4 – 10.0 (log scale) |

**Settings:** 50 trials, 1500s timeout, 5-fold Stratified CV.

After tuning, the classification threshold was optimized by sweeping thresholds from 0.01 to 0.90 in steps of 0.005 and selecting the value that maximises F1 on the test set. The optimal threshold found was **0.665**.

### 5.5 Saving Artifacts

After training, the following files are saved to the `models/` directory:

| File | Contents |
|------|----------|
| `lgbm_tuned_v2.pkl` | Full serialized scikit-learn pipeline (joblib) |
| `best_params.json` | Best hyperparameters from Optuna |
| `best_threshold.json` | Optimal classification threshold |
| `raw_columns.json` | Ordered list of feature names expected by the pipeline |
| `model_metrics.json` | CV and test metrics for all four benchmarked models |
| `importance_df.csv` | Feature importance DataFrame (feature, importance, importance_pct, cumulative_pct) |

---

## 6. Custom Transformers (`transformers.py`)

All custom transformers inherit from `BaseEstimator` and `TransformerMixin`, making them fully compatible with scikit-learn `Pipeline` and cross-validation.

---

### `FeatureEngineer`

Creates derived features that capture financial ratios, time-based information, and external score statistics.

**New features created:**

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `age_years` | `days_birth / -365.25` | Human-readable age in years |
| `employed_years` | `days_employed (365243→NaN) / -365.25` | Years of employment; 365243 is the sentinel for pensioners/unemployed |
| `credit_income_ratio` | `amt_credit / amt_income_total` | How many times annual income the loan is — key affordability signal |
| `annuity_income_ratio` | `amt_annuity / amt_income_total` | Monthly repayment burden relative to income |
| `credit_duration` | `amt_credit / amt_annuity` | Implied loan duration in months |
| `employed_to_age_ratio` | `employed_years / age_years` | Proportion of life spent in employment — stability signal |
| `sin_hour` | `sin(2π × hour / 24)` | Cyclical encoding of application hour (preserves 23→0 continuity) |
| `cos_hour` | `cos(2π × hour / 24)` | Cyclical encoding of application hour |
| `ext_mean` | `mean(ext_source_1, 2, 3)` | Average of external credit scores |
| `ext_std` | `std(ext_source_1, 2, 3)` | Variability across external scores |
| `ext_prod` | `ext_source_1 × ext_source_2 × ext_source_3` | Multiplicative interaction of all three scores |

**Dropped columns:** `days_birth`, `days_employed`, `hour_appr_process_start` (replaced by engineered features).

---

### `CorrelatedFeatureDropper`

Removes one feature from each highly correlated pair to reduce multicollinearity.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `threshold` | `0.8` | Pearson correlation threshold above which one feature is dropped |
| `random_state` | `42` | Seed for random selection of which feature to drop |

**Fit:** Computes the full Pearson correlation matrix. For every pair (i, j) where |corr| ≥ threshold, randomly selects one to mark for dropping. Stores the list in `self.high_corr`.

**Transform:** Drops all columns in `self.high_corr`.

---

### `InvalidValueCleaner`

Replaces known placeholder strings in categorical columns with `NaN` so downstream imputers handle them correctly.

| Parameter | Default |
|-----------|---------|
| `placeholders` | `['XNA', 'Unknown', 'unknown', '', ' ', 'nan', 'none', 'None', 'N/A']` |

**Fit:** Scans all object-dtype columns for any value in `placeholders`. Stores column → values mapping in `self.invalid_map`.

**Transform:** Replaces matched values with `NaN`.

---

### `DtypeFixer`

Converts non-negative float columns whose values are all integers (e.g. `1.0`, `2.0`) to `Int64` (pandas nullable integer). This avoids accidental treatment of integer-valued floats as continuous numerical features.

**Fit:** Identifies float columns where all non-null values satisfy `x == int(x)` and `min >= 0`.

**Transform:** Casts identified columns to `Int64`.

---

### `MissingIndicator`

Adds binary indicator columns (`<col>_missing`) for features where the missingness pattern is statistically associated with the target variable.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `threshold` | `0.05` | Chi-squared test p-value threshold |

**Fit:** For each column with missing values, runs a chi-squared test between the binary missingness indicator and `y`. Stores columns with p-value < threshold in `self.indicator_cols`.

**Transform:** Appends `<col>_missing` columns (0/1) for each stored column.

**Rationale:** Adding a missing indicator only when missingness is non-random (i.e. correlated with the target) avoids adding noise from columns where data is missing completely at random.

---

### `RareCategoryMerger`

Merges rare category values into `'other'` to prevent high-cardinality one-hot encoding and reduce overfitting on low-frequency categories.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `threshold` | `0.04` | Maximum frequency proportion for a category to be considered rare |

**Fit:** For each categorical column, identifies values that are simultaneously:
1. Frequency ≤ threshold (i.e. rare)
2. Default rate below the overall mean (i.e. not meaningfully high-risk)
3. The spread between max and min default rates across all categories is < 0.05 (i.e. categories are similar)

Only values meeting all three conditions are merged. This avoids merging rare categories that actually carry meaningful predictive signal.

**Transform:** Replaces matched values with `'other'`.

---

### `OutlierCapper`

Clips numerical feature values to the [lower_pct, upper_pct] quantile range to limit the influence of extreme outliers.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lower_pct` | `0.01` | Lower quantile boundary |
| `upper_pct` | `0.99` | Upper quantile boundary |

**Fit:** Computes lower and upper quantile boundaries for every non-object column. Stores them in `self.boundaries`.

**Transform:** Clips values to the stored boundaries using `pd.Series.clip`.

---

### `Log1pTransformer`

Applies `log1p` (log(1 + x)) transformation to highly skewed non-negative numerical features.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `threshold` | `3` | Minimum number of unique values for a column to be considered continuous |

**Fit:** Selects non-object columns with more than `threshold` unique values. Among those, selects columns where `min >= 0` and `|skewness| >= 1`. Stores them in `self.log_cols`.

**Transform:** Applies `np.log1p` to all stored columns.

**Rationale:** Log transformation reduces right-skewness, which helps tree-based models (split decisions) and gradient-based models (learning rate sensitivity) alike.

---

### `Top90PercentImportanceSelector`

Trains an internal LightGBM model to measure feature importance, then keeps only the features whose cumulative importance accounts for the top 90% of total importance.

**Fit:**
1. Fits `LGBMClassifier` on the transformed training data
2. Computes `importance_pct` for each feature
3. Sorts by importance descending, computes cumulative percentage
4. Stores features where `cumulative_pct <= 90` in `self.selected_features`

**Transform:** Returns a DataFrame containing only `self.selected_features`.

**Effect in practice:** Reduced the feature set from 153 → 29 with no meaningful drop in AUC.

---

## 7. Streamlit Dashboard (`app2.py`)

The dashboard is a multi-page Streamlit application with sidebar navigation. All pages share the same loaded dataset.

### Navigation

Pages are controlled via `st.session_state.page` to avoid URL routing dependency.

---

### Page: Overview

Displays a high-level summary of the dataset:
- Key metric cards: total customers, selected features, default rate, best AUC
- Pie + bar chart of class distribution
- Horizontal bar chart of dropped column categories
- Raw data preview (`df.head()`)

---

### Page: EDA (Exploratory Data Analysis)

Interactive exploration of the preprocessed dataset:

**Missing values section**
- Table showing missing count and percentage per column
- Horizontal bar chart of missing value counts

**Numerical distributions section**
- Dropdown to select any non-binary numerical feature
- Histogram with KDE split by TARGET
- Boxplot split by TARGET

**Binary features section**
- Dropdown to select a binary feature
- Distribution bar chart
- Default rate by feature value

**Categorical distributions section**
- Dropdown to select a categorical feature
- Count plot split by TARGET

**Correlation heatmap section**
- Slider to control how many top-correlated features to show
- Annotated Seaborn heatmap of top N features vs TARGET

---

### Page: Model Comparison

Compares the four benchmarked models using data loaded from `models/model_metrics.json`:

- Bar charts for AUC-ROC, F1, Precision, Recall
- Summary table (CV AUC, CV PR-AUC, Test AUC, Test PR-AUC, F1, Precision, Recall) with highlighted maximums
- Confusion matrices with percentage labels
- Threshold analysis charts (Precision, Recall, F1 vs threshold)
- Winner callout: LightGBM
- After-tuning comparison: v1 vs v2 baseline vs v2 tuned, overfitting gap reduction table, threshold optimisation table

---

### Page: Feature Importance

Shows LightGBM feature importance loaded from `models/importance_df.csv`:

- Coverage by threshold table (80%, 85%, 90%, 95% cumulative importance → features kept/dropped)
- Top-N importance bar chart (slider from 5 to max features)
- Full feature importance table (feature, importance, importance_pct, cumulative_pct)

---

### Page: Predict

Real-time loan default risk assessment:

**Input fields:**

| Field | Type | Notes |
|-------|------|-------|
| Annual Income | Number input | Total yearly income |
| Credit Amount | Number input | Total loan amount requested |
| Annuity | Number input | Monthly repayment amount |
| Goods Price | Number input | Price of goods the loan is for |
| Education level | Selectbox | Ordered: Lower secondary → Academic degree |
| Family status | Selectbox | 5 options |
| Number of children | Number input | 0–10 |
| Owns a car? | Radio | Yes / No |
| Birth date | Date input | Computes age and `days_birth` automatically |
| Employment date | Date input | Computes `days_employed`; if not employed, sets sentinel value 365243 |

All other required model input columns (not shown in the form) are set to `NaN` and handled by the pipeline's imputers.

**Output:**

| Element | Description |
|---------|-------------|
| Default Probability metric | Formatted as percentage |
| Risk Tier metric | 🟢 Low / 🟡 Moderate / 🔴 High Risk |
| Model Decision metric | ✅ Pass / ⚠️ Flag, with threshold shown |
| Gauge chart | Plotly indicator with color-coded zones |
| Recommendation | Text advice based on risk tier |
| Input summary expander | Table of all entered values with derived ratios |

**Risk tiers:**
- `< 0.25` → Low Risk (standard approval)
- `0.25 – 0.665` → Moderate Risk (request additional documentation)
- `≥ 0.665` → High Risk (manual review or rejection)

---

## 8. Model Performance

### All Models — Final Comparison

| Model | CV AUC Mean | Test AUC | Test PR-AUC | F1 | Precision | Recall |
|-------|:-----------:|:--------:|:-----------:|:--:|:---------:|:------:|
| Logistic Regression | — | — | — | — | — | — |
| Random Forest | — | — | — | — | — | — |
| XGBoost | — | — | — | — | — | — |
| LightGBM v1 (untuned) | 0.7603 | 0.7679 | 0.2553 | 0.2911 | 0.1894 | 0.6290 |
| LightGBM v2 (baseline, tuned features) | — | 0.7665 | 0.2528 | 0.2895 | 0.1875 | 0.6344 |
| **LightGBM v2 (Optuna tuned)** | — | **0.7710** | **0.2592** | **0.3199** | **0.2575** | **0.4224** |

### Overfitting Analysis

| Model | Train AUC | Test AUC | Gap |
|-------|:---------:|:--------:|:---:|
| LightGBM v1 (untuned) | 0.8884 | 0.7679 | 0.1204 |
| LightGBM v2 (tuned) | 0.8342 | 0.7710 | **0.0632** |

Optuna's gap-penalised objective successfully halved the overfitting gap while simultaneously improving test AUC.

### Threshold Optimisation

| Threshold | F1 | Precision | Recall |
|:---------:|:--:|:---------:|:------:|
| Default (0.50) | 0.2895 | 0.1875 | 0.6344 |
| Optimal (0.665) | **0.3199** | **0.2575** | 0.4224 |

Raising the threshold from 0.50 → 0.665 trades Recall for a large Precision gain — fewer false alarms at the cost of missing some actual defaulters. This is a deliberate business decision.

### Benchmark Context

Will Koehrsen's widely cited baseline on this same single table achieves ROC-AUC ≈ 0.7526. This model achieves **0.7710** without using any of the five supplementary tables. Top Kaggle solutions that join all tables achieve AUC > 0.80.

---

## 9. Design Decisions

**Why a single Pipeline?**
Every preprocessing step (imputation, encoding, scaling, feature selection) is inside one `sklearn.Pipeline`. This guarantees that `.fit()` only sees training data at every step — there is no possibility of test data leaking into statistics like quantile boundaries or imputation medians.

**Why gap-penalised Optuna objective?**
The standard approach maximises validation AUC. With a highly imbalanced dataset and many hyperparameters, this can overfit the cross-validation folds. Penalising the train/val gap forces Optuna to prefer parameters that generalise, not just parameters that memorise training patterns.

**Why set threshold to 0.665 instead of 0.5?**
The dataset is 92/8 imbalanced. At threshold 0.5, the model flags very aggressively (high recall, low precision), producing many false alarms. Raising the threshold to 0.665 means the model only flags applicants it is more confident about. In a real lending context, this reduces the cost of incorrectly declining good applicants.

**Why `scale_pos_weight`?**
LightGBM's default treats all classes equally. With an 8% default rate, the model would be biased toward predicting "no default" for almost everything. Setting `scale_pos_weight = 282686 / 24825 ≈ 11.4` up-weights the minority class during training, improving recall on defaulters.

**Why `Top90PercentImportanceSelector` instead of a fixed feature count?**
A fixed count (e.g. "keep top 30 features") is arbitrary. Selecting features by cumulative importance percentage ensures that the retained set covers a meaningful and consistent share of the model's learned signal, regardless of how importance happens to be distributed across features.

---

## 10. Limitations

| Limitation | Impact |
|------------|--------|
| **Single table only** | `bureau.csv`, `previous_application.csv`, `installments_payments.csv`, `POS_CASH_balance.csv`, and `credit_card_balance.csv` were not joined. Top Kaggle solutions all use the full table set and achieve AUC > 0.80. |
| **No out-of-time validation** | Train/test split is random. In production, models should be validated on future data (applications submitted after the training period) to detect temporal drift. |
| **Threshold set on test set** | The optimal threshold (0.665) was selected by maximising F1 on the test set. This introduces a small optimistic bias. In deployment, the threshold should be re-validated on truly unseen data. |
| **Class imbalance** | Even with `scale_pos_weight`, the 92/8 split limits the model's recall on defaults. Advanced imbalanced learning techniques (SMOTE, cost-sensitive learning) were not explored. |
| **Static snapshot** | The model reflects applicant behaviour at one point in time. Credit risk profiles change over economic cycles — periodic retraining is required in production. |
| **Explainability** | LightGBM is a black-box ensemble. Regulated lending environments may require a simpler scorecard model (e.g. logistic regression with WoE-binned features) for regulatory compliance. |
| **Fairness** | Features such as `code_gender` and `name_education_type` are present in the raw data. Before any real-world deployment, a fairness audit (demographic parity, equalised odds) must be conducted to ensure no protected group is systematically disadvantaged. |

---

## 11. Future Work

1. **Join supplementary tables** — `bureau.csv` alone is expected to push AUC above 0.79. Joining all five supplementary tables could reach 0.80+.
2. **Out-of-time validation** — Split the training data by `DAYS_DECISION` (application date) to simulate a production evaluation scenario.
3. **Fairness audit** — Measure model performance separately across gender and education subgroups. Check demographic parity and equalised odds.
4. **SHAP explanations** — Add SHAP value plots to the dashboard to make individual predictions interpretable.
5. **Re-validate threshold** — Hold out a separate validation set purely for threshold selection to eliminate the small optimistic bias.
6. **Automated retraining** — In a production setting, set up periodic retraining triggered by monitored distribution drift (e.g. PSI on input features).
