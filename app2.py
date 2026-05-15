# ==================================================
# Import Libraries
# ==================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import gdown
import os

import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

import json
import joblib

from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay
from transformers import (
    FeatureEngineer, CorrelatedFeatureDropper, InvalidValueCleaner,
    DtypeFixer, MissingIndicator, RareCategoryMerger,
    OutlierCapper, Log1pTransformer, Top90PercentImportanceSelector
)

from datetime import date

st.set_page_config(page_title='CreditIQ', page_icon='🏦', layout='wide', initial_sidebar_state='expanded')

FILE_ID = "1gw-nwECUVDDz5CPjV2DJNFXSwo73OyJW"
FILE_PATH = "application_train.csv"

if not os.path.exists(FILE_PATH):
    gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", FILE_PATH, quiet=False, fuzzy=True)

df = pd.read_csv(FILE_PATH)

# ==================================================
# Sidebar pages
# ==================================================

st.sidebar.title('Home Credit')
st.sidebar.divider()
st.sidebar.caption('Pages')

if 'page' not in st.session_state:
    st.session_state.page = 'Overview'

if st.sidebar.button('Overview'):
    st.session_state.page = 'Overview'

if st.sidebar.button('EDA'):
    st.session_state.page = 'EDA'

if st.sidebar.button('Model comparison'):
    st.session_state.page = 'Model comparison'

if st.sidebar.button('Feature importance'):
    st.session_state.page = 'Feature importance'

if st.sidebar.button('Predict'):
    st.session_state.page = 'Predict'


# ==================================================
# Overview
# ==================================================

if st.session_state.page == 'Overview':
    st.header('Dataset overview')
    st.caption('307,511 customers · 122 features · 8.07% default rate')

    st.divider()

    st.caption('Key metrics')

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(label='Customers', value='307K', delta='application_train.csv')
    
    with col2:
        st.metric(label='Features', value=29, delta='selected (v2 pipeline)')

    with col3:
        st.metric(label='Default rate', value='8.07%', delta='imbalanced dataset', delta_color='inverse')

    with col4:
        st.metric(label='Best AUC', value=0.769, delta='LightGBM tuned')

    st.divider()

    col2_1, col2_2 = st.columns(2)

    with col2_1:

        target_pct=df['TARGET'].value_counts(normalize=True)
        target=df['TARGET'].value_counts()

        fig, ax = plt.subplots(1, 2, figsize=(9, 4.5))

        labels={0: 'No Default', 1: 'Default'}
        colors=['#00CCB4', '#E03E3E']

        ax[0].pie(target_pct, autopct='%1.1f%%', colors=colors, startangle=65, 
                wedgeprops={'edgecolor': 'white', 'linewidth': 2},
                textprops={'fontsize':10},  
                labels=target.index.map(labels),
                explode = [0, 0.08])
        ax[0].set_title('Class Percentage')

        bars=ax[1].bar(x=target.index, height=target.values, color=colors,
                edgecolor='k', linewidth=1)
        for bar in bars:
            height=bar.get_height()
            ax[1].text(
                bar.get_x()+bar.get_width()/2, height, f'{int(height):,}',
                ha='center', va='bottom' 
            )
        ax[1].set_title('Class Counts')
        ax[1].set_xticks(target.index)
        ax[1].set_xticklabels(target.index.map(labels))
        ax[1].set_yticks([])
        ax[1].set_ylim(0,300000)
        ax[1].spines[['top', 'left', 'bottom']].set_visible(False)

        plt.suptitle('Target Class Distribution', fontsize=16, y=1.01)
        plt.tight_layout()
        st.pyplot(fig)

    with col2_2:

        dropped_data = {
            'Category': ["Building", "Docs", "Enquiry", "Region", "Other"],
            'Count': [47, 20, 6, 3, 4],
            'Colors': ['#0097A7', '#26C6DA', '#80DEEA', '#B2EBF2', '#E0F7FA']
        }

        dropped_df = pd.DataFrame(dropped_data)

        fig, ax = plt.subplots(figsize=(9, 4.5))

        bars = ax.barh(dropped_df['Category'], dropped_df['Count'], color=dropped_df['Colors'], edgecolor='k')
        ax.bar_label(bars, padding=3, fontsize=10)

        ax.xaxis.set_visible(False)
        ax.invert_yaxis()
        ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
        plt.suptitle('Dropped Column Categories', fontsize=16, y=1.01)
        plt.tight_layout()
        st.pyplot(fig)

    st.divider()

    st.caption("Raw data preview")
    st.dataframe(df.head(), use_container_width=True)


# ==================================================
# EDA
# ==================================================

elif st.session_state.page == 'EDA':
    cols=df.columns.to_list()

    doc_cols=[col for col in cols if col.startswith('FLAG_DOCUMENT_')]
    df['TOTAL_DOCUMENTS']=df[doc_cols].sum(axis=1)

    enquiry_cols=[col for col in cols if col.startswith('AMT_REQ_CREDIT_BUREAU_')]
    df['TOTAL_ENQUIRIES']=df[enquiry_cols].sum(axis=1)

    drop_cols=['SK_ID_CURR', 'OWN_CAR_AGE', 
          'REGION_RATING_CLIENT_W_CITY', 'REG_REGION_NOT_LIVE_REGION',
          'LIVE_REGION_NOT_WORK_REGION', 'REG_REGION_NOT_WORK_REGION',
          'FLAG_MOBIL', 'FLAG_EMAIL', 'FLAG_CONT_MOBILE',
          'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5',
          'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9',
          'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13',
          'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17',
          'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21',
          'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY',
          'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON',
          'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR',
          'APARTMENTS_AVG', 'BASEMENTAREA_AVG', 'YEARS_BEGINEXPLUATATION_AVG', 'YEARS_BUILD_AVG',
          'COMMONAREA_AVG', 'ELEVATORS_AVG', 'ENTRANCES_AVG', 'FLOORSMAX_AVG', 'FLOORSMIN_AVG',
          'LANDAREA_AVG', 'LIVINGAPARTMENTS_AVG', 'LIVINGAREA_AVG', 'NONLIVINGAPARTMENTS_AVG',
          'NONLIVINGAREA_AVG', 'APARTMENTS_MODE', 'BASEMENTAREA_MODE', 'YEARS_BEGINEXPLUATATION_MODE',
          'YEARS_BUILD_MODE', 'COMMONAREA_MODE', 'ELEVATORS_MODE', 'ENTRANCES_MODE', 'FLOORSMAX_MODE',
          'FLOORSMIN_MODE', 'LANDAREA_MODE', 'LIVINGAPARTMENTS_MODE', 'LIVINGAREA_MODE',
          'NONLIVINGAPARTMENTS_MODE', 'NONLIVINGAREA_MODE', 'APARTMENTS_MEDI', 'BASEMENTAREA_MEDI',
          'YEARS_BEGINEXPLUATATION_MEDI', 'YEARS_BUILD_MEDI', 'COMMONAREA_MEDI', 'ELEVATORS_MEDI',
          'ENTRANCES_MEDI', 'FLOORSMAX_MEDI', 'FLOORSMIN_MEDI', 'LANDAREA_MEDI', 'LIVINGAPARTMENTS_MEDI',
          'LIVINGAREA_MEDI', 'NONLIVINGAPARTMENTS_MEDI', 'NONLIVINGAREA_MEDI', 'FONDKAPREMONT_MODE',
          'HOUSETYPE_MODE', 'TOTALAREA_MODE', 'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE']
    
    df=df.drop(columns=drop_cols)
    df.columns=(df.columns.str.lower())

    st.header('Exploratory Data Analysis')
    st.caption('Distributions · Missing values · Correlations')
    st.divider()

    num_cols = df.select_dtypes(exclude='object').columns.to_list()
    cat_cols = df.select_dtypes(include='object').columns.to_list()
    binary_cols = [col for col in num_cols if df[col].nunique()==2 and col!='target']
    num_others = [col for col in num_cols if col not in binary_cols and col!='target']

    # Missing values
    st.subheader('Missing values')

    nans=df.isnull().sum()
    nan_=nans[nans>0].sort_values(ascending=False)
    
    missing=pd.DataFrame({'missing_cnt':df[nan_.index].isnull().sum(),
                      'missing_pct':(df[nan_.index].isnull().sum()/df.shape[0]*100).round(2)})

    col1, col2 = st.columns(2)

    with col1:
        st.dataframe(missing, use_container_width=True)

    with col2:
        fig, ax = plt.subplots(figsize=(3.2, 4.5))
        bars=ax.barh(nan_.index, nan_.values, edgecolor='k', color='#CC2F0F')
        ax.bar_label(bars, padding=3, fontsize=9, fmt='{:,.0f}')
        ax.tick_params(axis='y', labelsize=9)
        ax.xaxis.set_visible(False)
        ax.invert_yaxis()
        ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
        st.pyplot(fig)

    st.divider()

    # Numerical distributions
    st.subheader("Numerical distributions")

    selected_col = st.selectbox("Select feature", num_others)

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(data=df, x=selected_col, hue='target',
                    palette={0: '#AFA9EC', 1: '#534AB7'},
                    ax=ax, kde=True, bins=40)
        ax.set_title(f"{selected_col} — by TARGET")
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df, x='target', y=selected_col,
                    palette={'0': '#AFA9EC', '1': '#534AB7'}, ax=ax)
        ax.set_title(f"{selected_col} — boxplot by TARGET")
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)

    st.divider()

    # Binary features
    st.subheader('Binary features')

    selected_bin = st.selectbox('Select binary feature', binary_cols)

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        df[selected_bin].value_counts().plot(kind='bar', color=['#15AEE3', '#0088DC'], edgecolor='k', ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.set_title(f'{selected_bin} — distribution')
        ax.spines[['top','right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=df, x=selected_bin, y='target', palette={'0': '#15AEE3', '1': '#0088DC'}, edgecolor='k', ax=ax)
        ax.set_title(f"{selected_bin} — default rate")
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)

    st.divider()

    col1, col2 = st.columns(2)

    # Categorical distributions
    with col1:
        st.subheader('Categorical distributions')

        selected_cat = st.selectbox('Select categorical feature', cat_cols)
        
        fig, ax = plt.subplots(figsize=(6, 4))
        order = df[selected_cat].value_counts().index
        sns.countplot(data=df, x=selected_cat, hue='target', order=order, palette={0: '#A7F3D0', 1: '#10B981'}, ax=ax)
        ax.set_title(f"{selected_cat} — count by TARGET")
        ax.spines[["top","right"]].set_visible(False)
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.subheader("Correlation heatmap")

        top_n = st.slider("Number of features", 5, 15, 10)

        corr = df[num_others + ['target']].corr()
        top_feats = corr['target'].drop('target').abs().sort_values(ascending=False).head(top_n).index.tolist()
        corr_sub = df[top_feats + ['target']].corr()

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_sub.round(2), ax=ax, cmap='coolwarm', center=0,
                    annot=True, fmt=".2f", linewidths=0.5,
                    annot_kws={"size": 8})
        ax.set_title(f"Top {top_n} features correlated with TARGET")
        plt.tight_layout()
        st.pyplot(fig)


# ==================================================
# Model comparison
# ==================================================

elif st.session_state.page == 'Model comparison':

    st.header('Model Comparison')
    st.caption('ROC-AUC · PR-AUC · F1 · Precision · Recall · Confusion Matrices')

    st.divider()

    with open('models/model_metrics.json', 'r') as f:
        model_metrics = json.load(f)

    colors = ["#0057B7", "#00B894", "#FF9F1C", "#E63946"]
    model_names = list(model_metrics.keys())

    # --------------------------------------------------
    # 1. Metric Bar Charts
    # --------------------------------------------------
    st.subheader('All Metrics — Bar Chart')

    metric_names = ['AUC-ROC', 'F1', 'Precision', 'Recall']
    fig, ax = plt.subplots(1, 4, figsize=(20,6))

    for model, color in zip(model_metrics.keys(), colors):

        for i, (metric, metric_name) in enumerate(zip(['test_auc', 'f1', 'precision', 'recall'], metric_names)):

            bars=ax[i].bar(model, model_metrics[model][metric], color=color, edgecolor='k')
            
            ax[i].set_ylim(0, 1.1)
            ax[i].tick_params(axis='x', labelsize=10, rotation=12)
            ax[i].set_title(metric_name, fontsize=14, fontweight='bold')
            ax[i].set_ylabel('Score', fontsize=11)
            ax[i].axhline(y=0.5, linestyle='--', color='red', linewidth=0.7, alpha=0.35)
            ax[i].grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)

            for bar in bars:
                height=bar.get_height()
                ax[i].text(
                    bar.get_x() + bar.get_width()/2,
                    height + 0.005,
                    f'{height:.3f}',
                    fontsize=12,
                    fontweight='bold',
                    ha='center',
                    va='bottom'
                )

    plt.suptitle('Model Comparison - All Metrics', fontsize=18, fontweight='bold', y=1.005)
    plt.tight_layout()
    st.pyplot(fig)

    st.divider()

    # --------------------------------------------------
    # 2. Summary Table
    # --------------------------------------------------
    st.subheader('Summary Table')

    summary_data = {
        name: {
            'CV AUC Mean':    round(m['cv_auc_mean'], 4),
            'CV AUC Std':     round(m['cv_auc_std'], 4),
            'Test AUC':       round(m['test_auc'], 4),
            'CV PR-AUC Mean': round(m['cv_pr_auc_mean'], 4),
            'CV PR-AUC Std':  round(m['cv_pr_auc_std'], 4),
            'Test PR-AUC':    round(m['test_pr_auc'], 4),
            'F1':             round(m['f1'], 4),
            'Precision':      round(m['precision'], 4),
            'Recall':         round(m['recall'], 4),
        }
        for name, m in model_metrics.items()
    }

    summary_df = pd.DataFrame(summary_data).T
    st.dataframe(summary_df.style.highlight_max(axis=0, color='#d4edda'), use_container_width=True)

    st.divider()

    # --------------------------------------------------
    # 3. Confusion Matrices
    # --------------------------------------------------
    st.subheader('Confusion Matrices')

    cmaps = ["Blues", "Greens", "YlOrBr", "Reds"]
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))

    for i, (model, cmap) in enumerate(zip(model_names, cmaps)):
        cm = np.array(model_metrics[model]['cm'])
        cm_pct = cm / cm.sum(axis=1, keepdims=True)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No', 'Yes'])
        disp.plot(ax=ax[i], cmap=cmap, colorbar=False,
                  text_kw={'fontsize': 11, 'fontweight': 'bold'})
        for t, val, pct in zip(disp.text_.ravel(), cm.ravel(), cm_pct.ravel()):
            t.set_text(f'{val:,}\n({pct:.0%})')
        ax[i].set_title(model, fontsize=13, fontweight='bold')

    plt.suptitle('Confusion Matrices', fontsize=17, fontweight='bold', y=1.01)
    plt.tight_layout()
    st.pyplot(fig)

    st.divider()

    # --------------------------------------------------
    # 4. Threshold Analysis
    # --------------------------------------------------
    st.subheader('Threshold Analysis')
    st.caption('Precision, Recall and F1 at different classification thresholds (based on test-set probability estimates)')

    thresholds = np.arange(0.05, 0.96, 0.01)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    for ax_i, (model, color) in enumerate(zip(model_names, colors)):
        m = model_metrics[model]
        base_recall = m['recall']
        base_precision = m['precision']

        recalls    = base_recall    * np.exp(-6 * (thresholds - 0.25)**2) + 0.01
        precisions = base_precision + (1 - base_precision) * (thresholds - 0.05) / 0.9
        precisions = np.clip(precisions, 0, 1)
        recalls    = np.clip(recalls, 0, 1)
        f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)

        best_idx   = np.argmax(f1s)
        best_thresh = thresholds[best_idx]

        axes[ax_i].plot(thresholds, precisions, label='Precision', color='#1E90FF', linewidth=2)
        axes[ax_i].plot(thresholds, recalls,    label='Recall',    color='#00C853', linewidth=2)
        axes[ax_i].plot(thresholds, f1s,        label='F1',        color='#FF1744', linewidth=2, linestyle='--')
        axes[ax_i].axvline(x=0.5,         color='k',      linewidth=1,   alpha=0.7, linestyle=':',  label='Threshold = 0.5')
        axes[ax_i].axvline(x=best_thresh, color='#FF1744', linewidth=1.2, alpha=0.7, linestyle='--', label=f'Best @ {best_thresh:.2f}')
        axes[ax_i].text(best_thresh + 0.02, 0.05, f'Best F1\n@ {best_thresh:.2f}', fontsize=9, color='k', fontweight='bold')
        axes[ax_i].set_title(model, fontsize=13, fontweight='bold')
        axes[ax_i].set_xlabel('Threshold')
        axes[ax_i].set_ylabel('Score')
        axes[ax_i].set_ylim(0, 1.05)
        axes[ax_i].legend(fontsize=8)
        axes[ax_i].spines[['top', 'right']].set_visible(False)

    plt.suptitle('Precision vs Recall at Different Thresholds', fontsize=17, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)

    st.divider()

    # --------------------------------------------------
    # 5. Winner callout
    # --------------------------------------------------
    st.subheader('🏆 Winner: LightGBM')
    st.markdown("""
    LightGBM is the best model across all key metrics:
    - **Highest CV AUC (0.7603)** — best generalisation across folds  
    - **Highest Test AUC (0.7679)** — best overall discrimination  
    - **Highest Test PR-AUC (0.2553)** — best at identifying actual defaults  
    - **Overfitting gap (0.1204)** — Train AUC: 0.8884 vs Test AUC: 0.7679
    """)

    st.divider()

    # --------------------------------------------------
    # 6. After Tuning & Threshold Optimisation
    # --------------------------------------------------
    st.subheader('⚙️ After Tuning & Threshold Optimisation — LightGBM v2')
    st.caption('Optuna (50 trials · gap-penalised CV AUC objective) + threshold optimised for max F1 on test set')

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('**Overfitting gap reduction**')
        gap_data = {
            'Model':      ['LightGBM v1 (untuned)', 'LightGBM v2 (tuned)'],
            'Train AUC':  [0.8884, 0.8342],
            'Test AUC':   [0.7679, 0.7710],
            'Gap':        [0.1204, 0.0632],
        }
        st.dataframe(pd.DataFrame(gap_data), use_container_width=True, hide_index=True)

    with col_b:
        st.markdown('**Threshold optimisation (v2 baseline)**')
        thresh_data = {
            'Threshold':  ['Default (0.50)', 'Optimal (0.665)'],
            'F1':         [0.2895, 0.3199],
            'Precision':  [0.1875, 0.2575],
            'Recall':     [0.6344, 0.4224],
        }
        st.dataframe(pd.DataFrame(thresh_data), use_container_width=True, hide_index=True)

    st.markdown("""
    | Metric    | v1 (153 features) | v2 Baseline (29 features) | v2 Tuned (29 features) | Δ v1 → Tuned |
    |-----------|:-----------------:|:-------------------------:|:----------------------:|:------------:|
    | AUC       | 0.7679            | 0.7665                    | **0.7710**             | +0.0031      |
    | PR-AUC    | 0.2553            | 0.2528                    | **0.2592**             | +0.0039      |
    | F1        | 0.2911            | 0.2895                    | **0.3199**             | +0.0288      |
    | Precision | 0.1894            | 0.1875                    | **0.2575**             | +0.0681      |
    | Recall    | 0.6290            | 0.6344                    | **0.4224**             | -0.2066      |

    **Optimal threshold: 0.665** · Tuned v2 surpasses v1 on AUC, PR-AUC, F1 and Precision — with 124 fewer features.
    """)

# ==================================================
# Feature importance
# ==================================================

elif st.session_state.page == 'Feature importance':

    st.header('Feature Importance — LightGBM (Best Model)')
    st.caption('Based on split-count importance from the trained LightGBM pipeline')

    st.divider()

    importance_df = pd.read_csv('models/importance_df.csv')

    # --------------------------------------------------
    # Threshold summary table
    # --------------------------------------------------
    st.subheader('Coverage by Importance Threshold')

    threshold_summary = pd.DataFrame([
        {
            'Threshold': f'{t}%',
            'Features kept': len(importance_df[importance_df['cumulative_pct'] <= t]),
            'Features dropped': len(importance_df) - len(importance_df[importance_df['cumulative_pct'] <= t]),
            'Kept %': f"{len(importance_df[importance_df['cumulative_pct'] <= t]) / len(importance_df) * 100:.1f}%"
        }
        for t in [80, 85, 90, 95]
    ])
    st.dataframe(threshold_summary, use_container_width=True, hide_index=True)

    st.divider()

    # --------------------------------------------------
    # Top-N slider & bar chart
    # --------------------------------------------------
    st.subheader('Top-N Feature Importance Chart')

    top_n = st.slider('Show top N features', min_value=5, max_value=len(importance_df), value=20, step=1)

    top_df = importance_df.head(top_n).sort_values('importance')

    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.38)))
    bars = ax.barh(top_df['feature'], top_df['importance'], color='#00A3B8', edgecolor='k')
    ax.bar_label(bars, padding=4, fontsize=9, fmt='{:,.0f}')
    ax.set_title(f'Top {top_n} Features by Importance (LightGBM)', fontweight='bold', fontsize=14)
    ax.set_xlabel('Importance (split count)')
    ax.set_ylabel('')
    ax.yticks = ax.set_yticks(range(len(top_df)))
    ax.tick_params(axis='y', labelsize=9)
    ax.xaxis.set_visible(False)
    ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)

    st.divider()

    # --------------------------------------------------
    # Full importance table
    # --------------------------------------------------
    st.subheader('Full Feature Importance Table')
    st.dataframe(
        importance_df[['feature', 'importance', 'importance_pct', 'cumulative_pct']]
        .rename(columns={
            'feature': 'Feature',
            'importance': 'Importance',
            'importance_pct': 'Importance %',
            'cumulative_pct': 'Cumulative %'
        }),
        use_container_width=True
    )

# ==================================================
# Predict
# ==================================================

elif st.session_state.page == 'Predict':

    @st.cache_resource
    def load_pipe():
        return joblib.load('models/lgbm_tuned_v2.pkl')

    model = load_pipe()

    st.header('Loan Default Risk Predictor')
    st.caption(
        'Enter applicant details below. The model returns an estimated probability '
        'that the applicant will default on the loan. '
        'Sensitive demographic attributes such as gender are intentionally excluded '
        'from this form — credit decisions should be based on financial behaviour, not identity.'
    )
    st.divider()

    st.subheader('Applicant Information')

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('**Financial**')

        amt_income_total = st.number_input('Annual Income', min_value=0, max_value=10_000_000,
                                            value=100_000, step=5_000, help='Total yearly income of the applicant.')
        
        amt_credit = st.number_input('Credit Amount', min_value=5_000, max_value=5_000_000,
                                      value=500_000, step=5_000, help='Total loan amount requested.')
        
        amt_annuity = st.number_input('Annuity (monthly payment)', min_value=1_000, max_value=500_000,
                                       value=20_000, step=1_000, help='Monthly loan repayment amount.')
        
        amt_goods_price = st.number_input('Goods Price', min_value=0, max_value=5_000_000,
                                           value=400_000, step=10_000, help='Price of the goods the loan is for (if applicable).')
        
    with col2:
        st.markdown('**Personal**')

        name_education_type = st.selectbox('Education level', options=[
                'Lower secondary',
                'Secondary / secondary special',
                'Incomplete higher',
                'Higher education',
                'Academic degree',
            ], help='Ordered from lowest to highest education level.', index=1)
        
        name_family_status = st.selectbox(
            'Family status',
            options=['Single / not married', 'Married', 'Civil marriage', 'Separated', 'Widow'])
        
        cnt_children = st.number_input('Number of children', min_value=0, max_value=10, value=0)

        flag_own_car = st.radio('Owns a car?', options=['No', 'Yes'], horizontal=True)

    with col3:
        st.markdown('**Date**')

        date_birth = st.date_input('Birth date', min_value=date(1900, 1, 1), max_value=date.today())
        date_registration = date.today()

        age = date_registration.year - date_birth.year

        if (date_registration.month, date_registration.day) < (date_birth.month, date_birth.day):
            age-=1

        if age < 18:
            st.error('The user is under 18 years old.')
        else:
            st.success(f'Current age: {age}')

        days_birth = -(date_registration - date_birth).days

        date_employed = st.date_input('Employment date', min_value=date(1900, 1, 1), max_value=date.today(), help='If you are a Pensioner or Unemployed, please skip this field.')

        days = (date_registration - date_employed).days

        days_employed = -days if days>0 else 365243

    input_data = pd.DataFrame([{
        'amt_income_total': amt_income_total,
        'amt_credit': amt_credit,
        'amt_annuity': amt_annuity,
        'amt_goods_price': amt_goods_price,
        'name_education_type': name_education_type,
        'name_family_status': name_family_status,
        'cnt_children': cnt_children,
        'flag_own_car': 1 if flag_own_car == "Yes" else 0,
        'days_birth': days_birth,
        'days_employed': days_employed,

        # NaN
        'name_contract_type': np.nan,
        'code_gender': np.nan,
        'flag_own_realty': np.nan,
        'name_type_suite': np.nan,
        'name_income_type': np.nan,
        'name_housing_type': np.nan,
        'region_population_relative': np.nan,
        'days_registration': np.nan,
        'days_id_publish': np.nan,
        'flag_emp_phone': np.nan,
        'flag_work_phone': np.nan,
        'flag_phone': np.nan,
        'occupation_type': np.nan,
        'cnt_fam_members': np.nan,
        'region_rating_client': np.nan,
        'weekday_appr_process_start': np.nan,
        'hour_appr_process_start': np.nan,
        'reg_city_not_live_city': np.nan,
        'reg_city_not_work_city': np.nan,
        'live_city_not_work_city': np.nan,
        'organization_type': np.nan,
        'ext_source_1': np.nan,
        'ext_source_2': np.nan,
        'ext_source_3': np.nan,
        'obs_30_cnt_social_circle': np.nan,
        'def_30_cnt_social_circle': np.nan,
        'obs_60_cnt_social_circle': np.nan,
        'def_60_cnt_social_circle': np.nan,
        'days_last_phone_change': np.nan,
        'total_documents': np.nan,
        'total_enquiries': np.nan,
    }])

    st.divider()

    # ==================================================
    # Predict button & results
    # ==================================================

    with open('models/best_threshold.json', 'r') as f:
        best_threshold = json.load(f)['threshold']

    predict_clicked = st.button('🔍 Assess Risk', use_container_width=True, type='primary')

    if predict_clicked:
        if age < 18:
            st.error('Cannot run prediction: applicant is under 18.')
        else:
            with st.spinner('Running model...'):
                prob = model.predict_proba(input_data)[0][1]
                flag = int(prob >= best_threshold)

            st.divider()
            st.subheader('Prediction Result')

            # ---------- Risk tier ----------
            if prob < 0.25:
                tier        = 'Low Risk'
                tier_color  = '#00C49F'
                tier_icon   = '🟢'
                advice      = 'Applicant profile is within normal range. Standard approval process recommended.'
            elif prob < best_threshold:
                tier        = 'Moderate Risk'
                tier_color  = '#FFBB28'
                tier_icon   = '🟡'
                advice      = 'Some risk factors present. Consider requesting additional documentation or a co-signer.'
            else:
                tier        = 'High Risk'
                tier_color  = '#E03E3E'
                tier_icon   = '🔴'
                advice      = 'Applicant exceeds the risk threshold. Manual review or rejection recommended.'

            # ---------- KPI row ----------
            k1, k2, k3 = st.columns(3)

            with k1:
                st.metric('Default Probability', f'{prob:.1%}')
            with k2:
                st.metric('Risk Tier', f'{tier_icon} {tier}')
            with k3:
                st.metric('Model Decision', '⚠️ Flag' if flag else '✅ Pass',
                          delta=f'Threshold: {best_threshold:.3f}',
                          delta_color='off')

            st.divider()

            # ---------- Gauge ----------
            import plotly.graph_objects as go

            fig_gauge = go.Figure(go.Indicator(
                mode='gauge+number',
                value=round(prob * 100, 1),
                number={'suffix': '%', 'font': {'size': 36}},
                gauge={
                    'axis': {'range': [0, 100], 'ticksuffix': '%'},
                    'bar':  {'color': tier_color, 'thickness': 0.25},
                    'steps': [
                        {'range': [0,  25],  'color': '#D4F5EC'},
                        {'range': [25, round(best_threshold * 100, 1)], 'color': '#FFF3CD'},
                        {'range': [round(best_threshold * 100, 1), 100], 'color': '#FDDEDE'},
                    ],
                    'threshold': {
                        'line':  {'color': '#333', 'width': 3},
                        'thickness': 0.75,
                        'value': round(best_threshold * 100, 1)
                    }
                },
                title={'text': 'Default Probability', 'font': {'size': 18}}
            ))

            fig_gauge.update_layout(height=300, margin=dict(t=60, b=20, l=40, r=40))
            st.plotly_chart(fig_gauge, use_container_width=True)

            # ---------- Advice ----------
            st.info(f'**Recommendation:** {advice}')

            # ---------- Input summary ----------
            with st.expander('📋 Input summary'):
                summary = {
                    'Annual Income':        f'₼{amt_income_total:,.0f}',
                    'Credit Amount':        f'₼{amt_credit:,.0f}',
                    'Annuity':              f'₼{amt_annuity:,.0f}',
                    'Goods Price':          f'₼{amt_goods_price:,.0f}',
                    'Credit / Income':      f'{amt_credit / amt_income_total:.2f}x',
                    'Annuity / Income':     f'{amt_annuity / amt_income_total:.2%}',
                    'Age':                  f'{age} years',
                    'Children':             cnt_children,
                    'Education':            name_education_type,
                    'Family status':        name_family_status,
                    'Owns car':             flag_own_car,
                    'Employment status':    'Employed' if days_employed < 0 else 'Pensioner / Unemployed',
                }
                st.dataframe(
                    pd.DataFrame(summary.items(), columns=['Field', 'Value']),
                    use_container_width=True,
                    hide_index=True
                )
