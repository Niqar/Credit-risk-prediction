# transformers.py
# ---------------------------------------------------------------
# All custom sklearn transformers used in lgbm_tuned_v2.pkl.
# This file must be importable from the same directory as app1.py
# so that joblib.load() can deserialise the pipeline correctly.
#
# Usage in app1.py (add near the top, before joblib.load):
#   from transformers import (
#       FeatureEngineer, CorrelatedFeatureDropper, InvalidValueCleaner,
#       DtypeFixer, MissingIndicator, RareCategoryMerger,
#       OutlierCapper, Log1pTransformer, Top90PercentImportanceSelector
#   )
# ---------------------------------------------------------------

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.base import BaseEstimator, TransformerMixin
from lightgbm import LGBMClassifier


class FeatureEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        df=pd.DataFrame(X).copy()

        df['age_years'] = df['days_birth'] / -365.25
        df['employed_years'] = df['days_employed'].replace(365243, np.nan) / -365.25
        df['credit_income_ratio'] = df['amt_credit'] / df['amt_income_total']
        df['annuity_income_ratio'] = df['amt_annuity'] / df['amt_income_total']
        df['credit_duration'] = df['amt_credit'] / df['amt_annuity']
        df['employed_to_age_ratio'] = df['employed_years'] / df['age_years']

        df['sin_hour'] = np.sin(2 * np.pi * df['hour_appr_process_start'] / 24)
        df['cos_hour'] = np.cos(2 * np.pi * df['hour_appr_process_start'] / 24)

        df['ext_mean'] = df[['ext_source_1','ext_source_2','ext_source_3']].mean(axis=1)
        df['ext_std']  = df[['ext_source_1','ext_source_2','ext_source_3']].std(axis=1)
        df['ext_prod'] = df['ext_source_1'] * df['ext_source_2'] * df['ext_source_3']

        df.drop(columns=['days_birth', 'days_employed', 'hour_appr_process_start'], errors='ignore', inplace=True)

        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        return df


class CorrelatedFeatureDropper(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.8, random_state=42):
        self.threshold = threshold
        self.random_state = random_state

    def fit(self, X, y=None):
        self.high_corr = []
        df = pd.DataFrame(X)
        corr = df.corr(numeric_only=True)
        rng = np.random.default_rng(self.random_state)
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                if abs(corr.iloc[i, j]) >= self.threshold:
                    feat_1 = corr.columns[i]
                    feat_2 = corr.columns[j]
                    group = [feat_1, feat_2]
                    keep = rng.choice(group)
                    for f in group:
                        if f != keep:
                            self.high_corr.append(f)
        self.high_corr = list(set(self.high_corr))
        return self
    
    def transform(self, X):
        df=pd.DataFrame(X).copy()
        df.drop(columns = self.high_corr, inplace=True, errors='ignore')
        return df

class InvalidValueCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, placeholders = None):
        self.placeholders = placeholders or ['XNA', 'Unknown', 'unknown', '', ' ', 'nan', 'none', 'None', 'N/A']
    
    def fit(self, X, y=None):
        self.invalid_map={}
        X = pd.DataFrame(X)
        cat_cols = X.select_dtypes(include='object').columns.to_list()
        for col in cat_cols:
            uniques=X[col].unique()
            for unique in uniques:
                if unique in self.placeholders:
                    if col not in self.invalid_map:
                        self.invalid_map[col] = []
                    self.invalid_map[col].append(unique)
        return self
    
    def transform(self, X, y=None):
        X = pd.DataFrame(X).copy()
        for col in self.invalid_map.keys():
            X[col] = X[col].replace(self.invalid_map[col], np.nan)
        return X


class DtypeFixer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.convert_int = []
        X = pd.DataFrame(X)
        float_cols = X.select_dtypes(include='float').columns
        for col in float_cols:
            if X[col].min() >= 0:
                if X[col].dropna().apply(lambda x: x==int(x)).all():
                    self.convert_int.append(col)
        return self
    
    def transform(self, X, y=None):
        X = pd.DataFrame(X).copy()
        for col in self.convert_int:
            X[col]=X[col].astype('Int64')
        return X


class MissingIndicator(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.05):
        self.threshold = threshold

    def fit(self, X, y=None):
        self.indicator_cols = []
        X = pd.DataFrame(X)
        cols = X.columns
        for col in cols:
            nans = X[col].isnull().sum()
            if nans > 0:
                missing_indicator = X[col].isnull().astype(int)
                contingency_table = pd.crosstab(missing_indicator, y)
                chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                if p_value < self.threshold:
                    self.indicator_cols.append(col)
        return self
    
    def transform(self, X, y=None):
        X = pd.DataFrame(X).copy()
        for col in self.indicator_cols:
            X[str(col)+'_missing'] = X[col].isnull().astype(int)
        return X


class RareCategoryMerger(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.04):
        self.threshold = threshold

    def fit(self, X, y):
        self.merge_map = {}
        X = pd.DataFrame(X)
        cat_cols = X.select_dtypes(include='object').columns.to_list()
        for col in cat_cols:
            proportions = X[col].value_counts(normalize=True)
            default_rates = y.groupby(X[col]).mean()
            for value in proportions.index:
                if proportions[value] <= self.threshold and default_rates[value] < y.mean() and default_rates.max() - default_rates.min() < 0.05:
                    if col not in self.merge_map:
                        self.merge_map[col] = []
                    self.merge_map[col].append(value)
        return self
    
    def transform(self, X, y=None):
        X = pd.DataFrame(X).copy()
        for col in self.merge_map.keys():
            X[col] = X[col].replace(self.merge_map[col], 'other')
        return X


class OutlierCapper(BaseEstimator, TransformerMixin):
    def __init__(self, lower_pct=0.01, upper_pct=0.99):
        self.lower_pct = lower_pct
        self.upper_pct = upper_pct

    def fit(self, X, y=None):
        self.boundaries = {}
        X = pd.DataFrame(X)
        for col in X.select_dtypes(exclude='object'):
            self.boundaries[col] = {
            'lower_boundary' : X[col].quantile(self.lower_pct),
            'upper_boundary' : X[col].quantile(self.upper_pct)}
        return self
    
    def transform(self, X, y=None):
        X = pd.DataFrame(X).copy()
        for col in X.select_dtypes(exclude='object'):
            if col in self.boundaries:
                X[col] = X[col].clip(
                    lower = self.boundaries[col]['lower_boundary'],
                    upper = self.boundaries[col]['upper_boundary']
                )
        return X

class Log1pTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=3):
        self.threshold=threshold

    def fit(self, X, y=None):
        self.log_cols = []
        X = pd.DataFrame(X)
        num_cols = X.select_dtypes(exclude = 'object').columns
        num_others = [col for col in num_cols if X[col].nunique() > self.threshold]
        for col in num_others:
            skewness = abs(X[col].skew())
            if X[col].min() >= 0 and skewness >= 1:
                self.log_cols.append(col)
        return self
    
    def transform(self, X, y=None):
        X = pd.DataFrame(X).copy()
        for col in self.log_cols:
            X[col] = np.log1p(X[col])
        return X


class Top90PercentImportanceSelector(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.model = LGBMClassifier()

    def fit(self, X, y):
        X = pd.DataFrame(X)

        self.model.fit(X, y)

        importances = self.model.feature_importances_
        features = X.columns

        importance_df = pd.DataFrame({
            'feature': features,
            'importance': importances,
            'importance_pct': importances / importances.sum()*100
        }).sort_values('importance', ascending=False).reset_index()

        importance_df['cum_pct'] = importance_df['importance_pct'].cumsum()
        
        self.selected_features = importance_df[importance_df['cum_pct'] <= 90]['feature'].tolist()

        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        return X[self.selected_features]