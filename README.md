# Credit-risk-prediction
ML pipeline for credit default prediction using Home Credit dataset. LightGBM (AUC 0.771), Optuna tuning, feature engineering (153→29 features), scikit-learn Pipeline + Streamlit dashboard.
# 🏦 Credit Risk Prediction

A machine learning pipeline that predicts the probability of loan default using the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) dataset. The project includes a full scikit-learn preprocessing pipeline, Optuna hyperparameter tuning, and an interactive Streamlit dashboard.

---

## 📊 Results

| Metric    | LightGBM v2 Tuned |
|-----------|:-----------------:|
| AUC       | **0.7710**        |
| PR-AUC    | **0.2592**        |
| F1        | **0.3199**        |
| Precision | **0.2575**        |
| Recall    | **0.4224**        |
| Threshold | **0.665**         |

> Baseline (Will Koehrsen, single-table): ROC-AUC ≈ 0.7526. This model achieves **0.7710** without using any supplementary tables.

---

## 🗂️ Project Structure

```
credit-risk-prediction/
│
├── data/
│   └── application_train.csv        # Raw dataset (download from Kaggle)
│
├── models/
│   ├── lgbm_tuned_v2.pkl            # Trained pipeline (joblib)
│   ├── best_params.json             # Optuna best hyperparameters
│   ├── best_threshold.json          # Optimal classification threshold
│   ├── raw_columns.json             # Feature column names
│   ├── model_metrics.json           # All model evaluation metrics
│   └── importance_df.csv            # Feature importances
│
├── credit_risk_main2.ipynb          # Full ML workflow notebook
├── app2.py                          # Streamlit dashboard
├── transformers.py                  # Custom scikit-learn transformers
└── README.md
```

---

## ⚙️ Setup & Installation

**1. Clone the repository**
```bash
git clone https://github.com/Niqar/Credit-risk-prediction.git
cd credit-risk-prediction
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download the dataset**

Download `application_train.csv` from [Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk/data) and place it in the `data/` folder.

**4. Run the notebook**

Open `credit_risk_main2.ipynb` in Jupyter and run all cells. This will train the model and save all output files to the `models/` folder.

**5. Launch the Streamlit app**
```bash
streamlit run app2.py
```

---

## 📦 Requirements

```
pandas
numpy
scikit-learn
lightgbm
imbalanced-learn
optuna
streamlit
plotly
seaborn
matplotlib
missingno
scipy
joblib
gdown
```

---

## 🖥️ Dashboard Pages

| Page | Description |
|------|-------------|
| **Overview** | Dataset statistics, class distribution, dropped column summary |
| **EDA** | Interactive distributions, missing value heatmap, correlation matrix |
| **Model Comparison** | AUC, F1, confusion matrices, threshold analysis for all 4 models |
| **Feature Importance** | Top-N importance chart, cumulative coverage table |
| **Predict** | Real-time default probability for a new loan applicant |

---

## 📋 Requirements File

Save the following as `requirements.txt`:

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

---

## ⚠️ Limitations

- Uses only `application_train.csv` — joining supplementary tables (e.g. `bureau.csv`) is expected to push AUC above 0.79
- Train/test split is random — out-of-time validation should be used in production
- The optimal threshold (0.665) was selected on the test set and should be re-validated on truly unseen data
- Features like `code_gender` are present in the raw data — a fairness audit should be conducted before any real deployment

---

## 📄 License

MIT
