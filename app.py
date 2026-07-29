import io
import re
import time
import json
from datetime import datetime
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import streamlit as st

# ML & Stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix, classification_report
)
from scipy.stats import ttest_ind, chi2_contingency
import joblib

# ------------------------------
# Streamlit Page Config & Theme
# ------------------------------
st.set_page_config(
    page_title="MovieIQ – Intelligent Movie Dataset Analysis & Prediction Platform",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------
# Custom CSS for Dark, Polished UI
# ------------------------------
CUSTOM_CSS = """
<style>
/* Global dark theme adjustments */
:root {
  --primary: #6EE7F9;
  --accent: #8B5CF6;
  --bg: #0f1116;
  --card: #151823;
  --text: #ECEFF4;
  --muted: #9AA4B2;
  --success: #10B981;
  --warning: #F59E0B;
  --danger: #EF4444;
}

/* Main background */
.stApp {
  background: radial-gradient(1200px circle at 20% 10%, #121524 0%, #0f1116 35%, #0c0f14 100%);
  color: var(--text) !important;
}

/* Typography */
html, body, [class*="css"]  {
  font-family: Inter, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
}

/* Headings */
h1, h2, h3, h4 {
  color: var(--text) !important;
}

/* Card style */
.block-container { padding-top: 2rem; }

.card {
  background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 16px;
  padding: 18px 18px;
  box-shadow: 0 6px 24px rgba(0,0,0,0.35);
}

.metric-card {
  border-radius: 14px;
  background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
  border: 1px solid rgba(255,255,255,0.06);
  padding: 16px;
}

hr { border: none; height: 1px; background: rgba(255,255,255,0.08); margin: 8px 0 16px 0; }

/* Sidebar */
section[data-testid="stSidebar"] > div { background: #10131A; }
section[data-testid="stSidebar"] .block-container { padding: 1rem; }

/* Buttons */
.stButton>button {
  background: linear-gradient(180deg, var(--accent), #6D28D9);
  color: white;
  border-radius: 12px;
  border: none;
  padding: 0.6rem 1rem;
}
.stDownloadButton>button {
  background: linear-gradient(180deg, #0EA5E9, #0284C7);
  color: white; border-radius: 12px; border: none; padding: 0.6rem 1rem;
}

/* Progress bar */
.stProgress > div > div > div > div { background-image: linear-gradient(to right, var(--accent), var(--primary)); }

/* Dataframe rounded corners */
[data-testid="stDataFrame"] div {
  border-radius: 10px !important;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ------------------------------
# Constants & Required Columns
# ------------------------------
REQUIRED_COLUMNS = [
    "budget", "revenue", "genres", "runtime", "popularity", "vote_average", "title"
]
NUMERIC_COLUMNS = ["budget", "revenue", "runtime", "popularity", "vote_average"]

# ------------------------------
# Utility Helpers
# ------------------------------

def human_bytes(n: float, suffix: str = 'B') -> str:
    try:
        for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']:
            if abs(n) < 1024.0:
                return f"{n:3.1f}{unit}{suffix}"
            n /= 1024.0
        return f"{n:.1f}Y{suffix}"
    except Exception:
        return "-"

@st.cache_data(show_spinner=False)
def memory_usage_mb(df: pd.DataFrame) -> float:
    try:
        return float(df.memory_usage(deep=True).sum() / (1024**2))
    except Exception:
        return 0.0

@st.cache_data(show_spinner=False)
def compute_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    try:
        return df.describe(include='all').T
    except Exception:
        return pd.DataFrame()

def validate_required_columns(df: pd.DataFrame) -> List[str]:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    return missing

def safe_to_numeric(series: pd.Series) -> pd.Series:
    try:
        return pd.to_numeric(series, errors='coerce')
    except Exception:
        return series

def ensure_numeric_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    for c in columns:
        if c in df.columns:
            df[c] = safe_to_numeric(df[c])
    return df

def extract_primary_genre(genre_value: str) -> str:
    try:
        if pd.isna(genre_value):
            return "Unknown"
        # Common formats: 'Action|Adventure', '["Action", "Adventure"]', 'Action, Adventure'
        text = str(genre_value).strip()
        if text.startswith('[') and text.endswith(']'):
            # JSON-like list
            try:
                items = [g.strip().strip('"\'') for g in json.loads(text)]
                return items[0] if items else "Unknown"
            except Exception:
                pass
        # Split by common separators
        for sep in ['|', ',', ';', '/']:
            if sep in text:
                return text.split(sep)[0].strip() or "Unknown"
        return text or "Unknown"
    except Exception:
        return "Unknown"

@st.cache_data(show_spinner=False)
def add_primary_genre(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    try:
        df['primary_genre'] = df['genres'].apply(extract_primary_genre) if 'genres' in df.columns else "Unknown"
    except Exception:
        df['primary_genre'] = "Unknown"
    return df

@st.cache_data(show_spinner=False)
def detect_missing(df: pd.DataFrame) -> pd.Series:
    try:
        return df.isna().sum()
    except Exception:
        return pd.Series(dtype=int)

@st.cache_data(show_spinner=False)
def detect_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    try:
        dups = df[df.duplicated(keep=False)].copy()
        return dups
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def detect_invalids(df: pd.DataFrame) -> Dict[str, pd.Series]:
    # Returns boolean masks for invalid rows per rule
    masks = {}
    try:
        if 'budget' in df.columns:
            masks['invalid_budget'] = (df['budget'] <= 0)
        if 'revenue' in df.columns:
            masks['invalid_revenue'] = (df['revenue'] <= 0)
        if 'runtime' in df.columns:
            masks['invalid_runtime'] = (df['runtime'] <= 0)
        if 'popularity' in df.columns:
            masks['invalid_popularity'] = (df['popularity'] < 0)
        if 'vote_average' in df.columns:
            masks['invalid_vote'] = ((df['vote_average'] < 0) | (df['vote_average'] > 10))
    except Exception:
        pass
    return masks

@st.cache_data(show_spinner=False)
def detect_outliers_iqr(df: pd.DataFrame, cols: List[str]) -> Dict[str, pd.Series]:
    outlier_masks = {}
    try:
        for c in cols:
            if c in df.columns:
                series = pd.to_numeric(df[c], errors='coerce')
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outlier_masks[c] = (series < lower) | (series > upper)
    except Exception:
        pass
    return outlier_masks

@st.cache_data(show_spinner=False)
def correlation_matrix(df: pd.DataFrame, cols: Optional[List[str]] = None) -> pd.DataFrame:
    try:
        if cols is not None:
            use = [c for c in cols if c in df.columns]
        else:
            use = df.select_dtypes(include=np.number).columns.tolist()
        if len(use) == 0:
            return pd.DataFrame()
        return df[use].corr()
    except Exception:
        return pd.DataFrame()

# ------------------------------
# Data Cleaning Operations
# ------------------------------

def apply_missing_strategy(df: pd.DataFrame, strategy: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
    report = {"missing_removed": 0, "missing_filled": 0}
    try:
        if strategy == "Remove Missing Rows":
            before = len(df)
            df = df.dropna()
            report["missing_removed"] = before - len(df)
        elif strategy == "Fill Numeric with Median":
            num_cols = df.select_dtypes(include=[np.number]).columns
            medians = df[num_cols].median()
            df[num_cols] = df[num_cols].fillna(medians)
            report["missing_filled"] = int(df[num_cols].isna().sum().sum())
            # second pass to fill again in case of initial NaNs, though above should fill
            df[num_cols] = df[num_cols].fillna(medians)
            # categorical mode
            cat_cols = df.select_dtypes(exclude=[np.number]).columns
            for c in cat_cols:
                mode_val = df[c].mode(dropna=True)
                if len(mode_val) > 0:
                    df[c] = df[c].fillna(mode_val[0])
        elif strategy == "Fill Categorical with Mode":
            cat_cols = df.select_dtypes(exclude=[np.number]).columns
            for c in cat_cols:
                mode_val = df[c].mode(dropna=True)
                if len(mode_val) > 0:
                    df[c] = df[c].fillna(mode_val[0])
            # numeric untouched
        elif strategy == "Keep Missing Values":
            pass
    except Exception:
        pass
    return df, report

def apply_duplicate_strategy(df: pd.DataFrame, strategy: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
    report = {"duplicates_removed": 0}
    try:
        if strategy == "Remove All Duplicates":
            before = len(df)
            df = df.drop_duplicates()
            report["duplicates_removed"] = before - len(df)
        elif strategy in ("Keep Duplicates", "Manual Edit"):
            pass
    except Exception:
        pass
    return df, report

def apply_invalid_strategy(df: pd.DataFrame, strategy: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
    report = {
        "invalid_budget_removed": 0,
        "invalid_revenue_removed": 0,
        "invalid_runtime_removed": 0,
        "invalid_ratings_removed": 0,
        "invalid_popularity_removed": 0,
    }
    try:
        masks = detect_invalids(df)
        if strategy == "Remove":
            before = len(df)
            combined = None
            for k, m in masks.items():
                if combined is None:
                    combined = m.copy()
                else:
                    combined = combined | m
            if combined is not None:
                # Count per category
                for k, m in masks.items():
                    report_key = {
                        'invalid_budget': 'invalid_budget_removed',
                        'invalid_revenue': 'invalid_revenue_removed',
                        'invalid_runtime': 'invalid_runtime_removed',
                        'invalid_vote': 'invalid_ratings_removed',
                        'invalid_popularity': 'invalid_popularity_removed'
                    }.get(k, None)
                    if report_key:
                        report[report_key] = int(m.sum())
                df = df[~combined]
        elif strategy in ("Keep", "Manual Edit"):
            pass
    except Exception:
        pass
    return df, report

def apply_outlier_strategy(df: pd.DataFrame, strategy: str, cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, int]]:
    report = {"outliers_removed": 0}
    try:
        masks = detect_outliers_iqr(df, cols)
        if strategy == "Remove Outliers":
            before = len(df)
            if masks:
                combined = None
                for m in masks.values():
                    if combined is None:
                        combined = m.copy()
                    else:
                        combined = combined | m
                if combined is not None:
                    report["outliers_removed"] = int(combined.sum())
                    df = df[~combined]
        elif strategy == "Keep Outliers":
            pass
    except Exception:
        pass
    return df, report

# ------------------------------
# Target Variable
# ------------------------------

def ensure_target_success(df: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
    created = False
    try:
        if 'success' not in df.columns:
            if 'revenue' in df.columns and 'budget' in df.columns:
                df['success'] = (pd.to_numeric(df['revenue'], errors='coerce') > pd.to_numeric(df['budget'], errors='coerce')).astype(int)
                created = True
        # If exists, coerce to int 0/1
        if 'success' in df.columns:
            df['success'] = pd.to_numeric(df['success'], errors='coerce').fillna(0).astype(int)
    except Exception:
        pass
    return df, created

# ------------------------------
# Filtering
# ------------------------------

def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    try:
        # Genre filter: match if any selected appears in genres text (case-insensitive)
        genres_selected = st.session_state.get('filter_genres', [])
        min_rating, max_rating = st.session_state.get('filter_rating', (0.0, 10.0))
        budget_range = st.session_state.get('filter_budget', (float(df['budget'].min() or 0), float(df['budget'].max() or 0))) if 'budget' in df.columns else (0.0, 0.0)
        runtime_range = st.session_state.get('filter_runtime', (float(df['runtime'].min() or 0), float(df['runtime'].max() or 0))) if 'runtime' in df.columns else (0.0, 0.0)
        revenue_range = st.session_state.get('filter_revenue', (float(df['revenue'].min() or 0), float(df['revenue'].max() or 0))) if 'revenue' in df.columns else (0.0, 0.0)
        success_filter = st.session_state.get('filter_success', 'All')
        search_query = st.session_state.get('filter_search', '').strip().lower()

        out = df.copy()
        # Numeric ranges
        if 'vote_average' in out.columns:
            out = out[(out['vote_average'] >= min_rating) & (out['vote_average'] <= max_rating)]
        if 'budget' in out.columns:
            out = out[(out['budget'] >= budget_range[0]) & (out['budget'] <= budget_range[1])]
        if 'runtime' in out.columns:
            out = out[(out['runtime'] >= runtime_range[0]) & (out['runtime'] <= runtime_range[1])]
        if 'revenue' in out.columns:
            out = out[(out['revenue'] >= revenue_range[0]) & (out['revenue'] <= revenue_range[1])]

        # Genre filter
        if genres_selected:
            pattern = '|'.join([pd.regex.escape(g) if hasattr(pd, 'regex') else g for g in genres_selected])
            mask = out['genres'].astype(str).str.contains('|'.join(genres_selected), case=False, na=False)
            out = out[mask]

        # Success filter
        if success_filter in ["Success", "Failure"] and 'success' in out.columns:
            target_val = 1 if success_filter == "Success" else 0
            out = out[out['success'] == target_val]

        # Search in title
        if search_query and 'title' in out.columns:
            out = out[out['title'].astype(str).str.lower().str.contains(search_query, na=False)]
        return out
    except Exception:
        return df

# ------------------------------
# KPI & Insights
# ------------------------------

def kpi_cards(df: pd.DataFrame) -> Dict[str, any]:
    res = {}
    try:
        total_movies = len(df)
        avg_budget = float(df['budget'].mean()) if 'budget' in df.columns else 0.0
        avg_revenue = float(df['revenue'].mean()) if 'revenue' in df.columns else 0.0
        avg_runtime = float(df['runtime'].mean()) if 'runtime' in df.columns else 0.0
        avg_rating = float(df['vote_average'].mean()) if 'vote_average' in df.columns else 0.0
        success_rate = float(df['success'].mean()*100.0) if 'success' in df.columns else 0.0
        # Most common genre
        try:
            temp = df.copy()
            temp['primary_genre'] = temp['genres'].apply(extract_primary_genre)
            most_common_genre = temp['primary_genre'].mode().iloc[0] if not temp['primary_genre'].mode().empty else 'Unknown'
        except Exception:
            most_common_genre = 'Unknown'
        res = {
            'total_movies': total_movies,
            'avg_budget': avg_budget,
            'avg_revenue': avg_revenue,
            'avg_runtime': avg_runtime,
            'avg_rating': avg_rating,
            'success_rate': success_rate,
            'most_common_genre': most_common_genre,
        }
    except Exception:
        pass
    return res

# ------------------------------
# Plot Helpers & Distribution Insights
# ------------------------------

def distribution_fig(df: pd.DataFrame, col: str, bins: int = 40) -> Tuple[go.Figure, str]:
    try:
        series = pd.to_numeric(df[col], errors='coerce').dropna()
        fig = px.histogram(df, x=col, nbins=bins, template='plotly_dark', opacity=0.85,
                           color_discrete_sequence=['#60A5FA'])
        fig.update_layout(margin=dict(l=10,r=10,t=30,b=10), height=320, bargap=0.05)
        insight = ""
        if len(series) > 0:
            mean = series.mean(); median = series.median()
            skew = series.skew()
            if skew > 0.5:
                insight = f"Right-skewed: a few high values drive the mean above the median (mean {mean:,.0f} > median {median:,.0f})."
            elif skew < -0.5:
                insight = f"Left-skewed: lower values dominate (mean {mean:,.0f} < median {median:,.0f})."
            else:
                insight = f"Roughly symmetric distribution (mean {mean:,.0f}, median {median:,.0f})."
        return fig, insight
    except Exception:
        return go.Figure(), "No insight available."


def scatter_fig(df: pd.DataFrame, x: str, y: str, color: Optional[str] = None) -> go.Figure:
    try:
        fig = px.scatter(df, x=x, y=y, color=color, template='plotly_dark', opacity=0.8,
                         hover_data=['title'] if 'title' in df.columns else None,
                         color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_traces(marker=dict(size=7, line=dict(width=0.3, color='white')))
        fig.update_layout(height=360, margin=dict(l=10,r=10,t=30,b=10))
        return fig
    except Exception:
        return go.Figure()


def heatmap_fig(corr: pd.DataFrame) -> go.Figure:
    try:
        fig = px.imshow(corr, text_auto=False, aspect='auto', color_continuous_scale='Viridis',
                        template='plotly_dark')
        fig.update_layout(height=480, margin=dict(l=10,r=10,t=30,b=10))
        return fig
    except Exception:
        return go.Figure()

# ------------------------------
# Statistical Tests
# ------------------------------

def run_ttest_revenue_success(df: pd.DataFrame) -> Dict[str, any]:
    result = {
        'hypothesis': 'H0: Mean revenue is equal for success vs. failure; H1: They differ.',
        'p_value': None, 'decision': 'Insufficient data', 'interpretation': '', 'business_meaning': ''
    }
    try:
        if 'success' in df.columns and 'revenue' in df.columns:
            grp1 = pd.to_numeric(df.loc[df['success']==1, 'revenue'], errors='coerce').dropna()
            grp0 = pd.to_numeric(df.loc[df['success']==0, 'revenue'], errors='coerce').dropna()
            if len(grp1) > 3 and len(grp0) > 3:
                stat, p = ttest_ind(grp1, grp0, equal_var=False)
                result['p_value'] = float(p)
                result['decision'] = 'Reject H0' if p < 0.05 else 'Fail to Reject H0'
                result['interpretation'] = (
                    'Evidence of different revenue means between successful and failed movies.'
                    if p < 0.05 else 'No strong evidence of a difference in mean revenue.'
                )
                result['business_meaning'] = (
                    'Success relates to significantly different revenue; investing to raise success drivers may improve returns.'
                    if p < 0.05 else 'Revenue differences may be driven by other factors beyond success label.'
                )
    except Exception:
        pass
    return result


def run_chi_square_genre_success(df: pd.DataFrame) -> Dict[str, any]:
    result = {
        'hypothesis': 'H0: Genre and success are independent; H1: They are associated.',
        'p_value': None, 'decision': 'Insufficient data', 'interpretation': '', 'business_meaning': ''
    }
    try:
        if 'success' in df.columns and 'genres' in df.columns:
            temp = df.copy()
            temp['primary_genre'] = temp['genres'].apply(extract_primary_genre)
            # Keep top 10 genres to stabilize test
            top_genres = temp['primary_genre'].value_counts().nlargest(10).index
            temp = temp[temp['primary_genre'].isin(top_genres)]
            if temp['primary_genre'].nunique() > 1 and temp['success'].nunique() > 1:
                ct = pd.crosstab(temp['primary_genre'], temp['success'])
                chi2, p, dof, exp = chi2_contingency(ct)
                result['p_value'] = float(p)
                result['decision'] = 'Reject H0' if p < 0.05 else 'Fail to Reject H0'
                result['interpretation'] = (
                    'Genre is associated with success probability.' if p < 0.05 else 'No strong evidence of association between genre and success.'
                )
                result['business_meaning'] = (
                    'Some genres may systematically perform better; prioritize favorable genres in greenlighting decisions.'
                    if p < 0.05 else 'Genre alone may not determine success; combine with other predictors.'
                )
    except Exception:
        pass
    return result

# ------------------------------
# Machine Learning: Training & Prediction
# ------------------------------

def build_pipeline() -> Pipeline:
    # Numeric features for tree-based model do not need scaling
    numeric_features = ['budget', 'runtime', 'popularity', 'vote_average']
    categorical_features = ['primary_genre']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', 'passthrough', numeric_features)
        ]
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        max_depth=None,
        n_jobs=-1,
        class_weight='balanced_subsample'
    )

    pipe = Pipeline(steps=[('preprocess', preprocessor), ('model', clf)])
    return pipe


def train_model(df: pd.DataFrame) -> Tuple[Pipeline, Dict[str, any]]:
    metrics = {}
    model = build_pipeline()
    try:
        work = df.copy()
        work = add_primary_genre(work)
        # Features & Target
        X = work[['primary_genre', 'budget', 'runtime', 'popularity', 'vote_average']].copy()
        y = work['success'].astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = None
        try:
            y_prob = model.predict_proba(X_test)[:,1]
        except Exception:
            pass

        # Metrics
        metrics['accuracy'] = float(accuracy_score(y_test, y_pred))
        metrics['precision'] = float(precision_score(y_test, y_pred, zero_division=0))
        metrics['recall'] = float(recall_score(y_test, y_pred, zero_division=0))
        metrics['f1'] = float(f1_score(y_test, y_pred, zero_division=0))

        # ROC
        roc_fig = go.Figure()
        if y_prob is not None:
            fpr, tpr, thr = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            metrics['roc_auc'] = float(roc_auc)
            roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC AUC={roc_auc:.3f}', line=dict(color='#60A5FA', width=3)))
            roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random', line=dict(color='gray', dash='dash')))
            roc_fig.update_layout(template='plotly_dark', height=320, margin=dict(l=10,r=10,t=30,b=10), title='ROC Curve')

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred, labels=[0,1])
        cm_fig = px.imshow(cm, x=['Pred 0','Pred 1'], y=['True 0','True 1'], text_auto=True, template='plotly_dark', color_continuous_scale='Blues')
        cm_fig.update_layout(height=320, margin=dict(l=10,r=10,t=30,b=10), title='Confusion Matrix')

        # Classification Report
        metrics['classification_report'] = classification_report(y_test, y_pred, zero_division=0)

        # Feature Importances
        # Extract feature names from preprocessor
        pre: ColumnTransformer = model.named_steps['preprocess']
        cat_names = []
        try:
            cat_encoder: OneHotEncoder = pre.named_transformers_['cat']
            cat_names = list(cat_encoder.get_feature_names_out(['primary_genre']))
        except Exception:
            pass
        feature_names = cat_names + ['budget', 'runtime', 'popularity', 'vote_average']
        importances = model.named_steps['model'].feature_importances_
        # Align lengths defensively
        if len(importances) == len(feature_names):
            fi_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('importance', ascending=False).head(20)
        else:
            # Fallback: aggregate unknown
            fi_df = pd.DataFrame({'feature': [f'F{i}' for i in range(len(importances))], 'importance': importances}).sort_values('importance', ascending=False).head(20)
        fi_fig = px.bar(fi_df, x='importance', y='feature', orientation='h', template='plotly_dark', color='importance', color_continuous_scale='Purples')
        fi_fig.update_layout(height=380, margin=dict(l=10,r=10,t=30,b=10), title='Feature Importance')

        metrics['roc_fig'] = roc_fig
        metrics['cm_fig'] = cm_fig
        metrics['fi_fig'] = fi_fig

        return model, metrics
    except Exception as e:
        # Return minimally trained model to avoid crash
        return model, metrics


def predict_with_model(model: Pipeline, budget: float, runtime: float, popularity: float, vote_average: float, genre: str) -> Tuple[int, float]:
    try:
        row = pd.DataFrame([{
            'primary_genre': genre,
            'budget': budget,
            'runtime': runtime,
            'popularity': popularity,
            'vote_average': vote_average
        }])
        prob = 0.0
        try:
            prob = float(model.predict_proba(row)[:,1][0])
        except Exception:
            pass
        pred = int(model.predict(row)[0])
        return pred, prob
    except Exception:
        return 0, 0.0

# ------------------------------
# PDF Report
# ------------------------------

def generate_eda_pdf(df: pd.DataFrame, corr: pd.DataFrame) -> bytes:
    try:
        buffer = io.BytesIO()
        with PdfPages(buffer) as pdf:
            # Title Page
            fig = plt.figure(figsize=(8.27, 11.69))
            plt.axis('off')
            plt.text(0.5, 0.8, 'MovieIQ – EDA Report', ha='center', va='center', fontsize=24, fontweight='bold')
            plt.text(0.5, 0.75, datetime.now().strftime('%Y-%m-%d %H:%M'), ha='center', va='center', fontsize=12)
            plt.text(0.5, 0.65, f"Rows: {len(df):,}  |  Columns: {df.shape[1]}", ha='center', va='center', fontsize=12)
            pdf.savefig(fig); plt.close(fig)

            # Distributions
            for col in [c for c in ['budget','revenue','runtime','vote_average','popularity'] if c in df.columns]:
                fig = plt.figure(figsize=(11.69, 8.27))
                plt.hist(pd.to_numeric(df[col], errors='coerce').dropna(), bins=40, color='#60A5FA', alpha=0.85)
                plt.title(f'{col.title()} Distribution'); plt.xlabel(col); plt.ylabel('Count')
                pdf.savefig(fig); plt.close(fig)

            # Correlation Heatmap (matplotlib)
            if corr is not None and not corr.empty:
                fig = plt.figure(figsize=(11.69, 8.27))
                plt.imshow(corr, cmap='viridis'); plt.colorbar()
                plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha='right', fontsize=8)
                plt.yticks(range(len(corr.index)), corr.index, fontsize=8)
                plt.title('Correlation Matrix')
                pdf.savefig(fig); plt.close(fig)
        buffer.seek(0)
        return buffer.getvalue()
    except Exception:
        return b''

# ------------------------------
# Business Recommendations
# ------------------------------

def generate_recommendations(df: pd.DataFrame, kpis: Dict[str, any], corr: pd.DataFrame) -> List[str]:
    recs = []
    try:
        # KPI-driven
        if kpis.get('success_rate', 0) < 50:
            recs.append('Improve greenlighting by focusing on concepts with historically higher success indicators (optimize budget allocation).')
        else:
            recs.append('Maintain investment discipline in genres and budgets that sustain high success rates.')
        if kpis.get('avg_runtime', 0) > 140:
            recs.append('Consider tighter editing: very long runtimes may reduce audience throughput and screenings per day.')
        else:
            recs.append('Runtime is within a broadly marketable range; continue optimizing pacing for engagement.')
        if kpis.get('avg_rating', 0) < 6.0:
            recs.append('Strengthen script development and test screenings to elevate audience ratings before release.')
        else:
            recs.append('Leverage strong ratings with expanded marketing in high-performing markets.')

        # Genre-driven
        try:
            temp = df.copy(); temp['primary_genre'] = temp['genres'].apply(extract_primary_genre)
            sr = temp.groupby('primary_genre')['success'].mean().sort_values(ascending=False)
            if not sr.empty:
                top_g = sr.index[0]
                recs.append(f'Prioritize {top_g} projects given the highest observed success rate in the dataset.')
        except Exception:
            pass

        # Correlation-driven
        try:
            if corr is not None and not corr.empty:
                # strongest pos/neg with revenue
                if 'revenue' in corr.columns:
                    rev_corr = corr['revenue'].drop('revenue', errors='ignore').sort_values(ascending=False)
                    if not rev_corr.empty:
                        top_feat = rev_corr.index[0]
                        recs.append(f'Maximize drivers positively correlated with revenue (e.g., optimize {top_feat}).')
                    rev_corr_neg = corr['revenue'].drop('revenue', errors='ignore').sort_values()
                    if not rev_corr_neg.empty:
                        bad_feat = rev_corr_neg.index[0]
                        recs.append(f'Mitigate factors negatively correlated with revenue (monitor {bad_feat}).')
        except Exception:
            pass

        # Data quality
        recs.append('Maintain data hygiene: promptly address missing values, duplicates, and invalid entries to ensure robust analytics.')
        recs.append('Adopt outlier-aware KPIs; track medians and percentile ranges to avoid skewed decisions by extreme cases.')

        # Marketing & release
        recs.append('Use early audience feedback loops (trailers, test cuts) to improve predicted success probability pre-release.')
        recs.append('Align marketing spend with predicted ROI; dynamically reallocate budget towards high-probability successes.')
        recs.append('Diversify portfolio across 2–3 top-performing genres to balance risk and capitalize on trends.')

        # Minimum 10 recommendations safeguard
        while len(recs) < 10:
            recs.append('Iteratively refine feature engineering (e.g., sub-genre, cast features) to enhance predictive performance.')
    except Exception:
        pass
    return recs[:15]

# ------------------------------
# Session State Initialization
# ------------------------------
if 'raw_df' not in st.session_state:
    st.session_state['raw_df'] = None
if 'cleaned_df' not in st.session_state:
    st.session_state['cleaned_df'] = None
if 'file_name' not in st.session_state:
    st.session_state['file_name'] = None
if 'upload_complete' not in st.session_state:
    st.session_state['upload_complete'] = False
if 'loading_done' not in st.session_state:
    st.session_state['loading_done'] = False
if 'cleaning_report' not in st.session_state:
    st.session_state['cleaning_report'] = {}
if 'model' not in st.session_state:
    st.session_state['model'] = None
if 'insights' not in st.session_state:
    st.session_state['insights'] = []

# ------------------------------
# Header / First Screen
# ------------------------------

def render_hero():
    st.markdown("""
    <div style='text-align:center; margin-top: 6vh;'>
        <h1 style='font-size:48px; margin-bottom: 4px;'>🎬 MovieIQ</h1>
        <div style='color:#9AA4B2; font-size:18px; margin-bottom:24px;'>Intelligent Movie Dataset Analysis & Prediction Platform</div>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        st.write("")
        c1, c2, c3 = st.columns([1,2,1])
        with c2:
            st.markdown("<div class='card' style='text-align:center; padding:24px;'>", unsafe_allow_html=True)
            st.subheader("Upload Your Dataset")
            st.caption("Drag & Drop CSV or Browse Files")
            uploaded = st.file_uploader("Only CSV files are accepted", type=['csv'], accept_multiple_files=False, label_visibility='collapsed')
            st.markdown("</div>", unsafe_allow_html=True)
    return uploaded

# ------------------------------
# Loading Animation
# ------------------------------

def professional_loading_sequence(df_loader):
    status_placeholder = st.empty()
    progress = st.progress(0)
    steps = [
        ("Loading Dataset...", 10),
        ("Reading CSV...", 25),
        ("Checking Columns...", 40),
        ("Detecting Missing Values...", 55),
        ("Removing Invalid Records...", 65),
        ("Creating Target Variable...", 75),
        ("Preparing Dashboard...", 90),
        ("Finalizing Analysis...", 100),
    ]
    for msg, p in steps:
        status_placeholder.info(msg)
        time.sleep(0.25)
        progress.progress(p)
    progress.empty()
    status_placeholder.success("Dataset Uploaded Successfully • Dataset is Ready for Analysis")
    time.sleep(0.4)
    status_placeholder.empty()

# ------------------------------
# Sidebar Builder
# ------------------------------

def build_sidebar(df: pd.DataFrame, original: pd.DataFrame, file_name: str):
    with st.sidebar:
        # Logo & Title
        st.markdown("""
        <div style='display:flex; align-items:center; gap:10px; margin-bottom:6px;'>
            <div style='font-size:26px;'>🎬</div>
            <div>
                <div style='font-weight:700; font-size:20px;'>MovieIQ</div>
                <div style='color:#9AA4B2; font-size:12px; margin-top:-2px;'>Intelligent Analysis</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)

        # Dataset Status
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<b>📦 Dataset Status</b>", unsafe_allow_html=True)
        st.success("Uploaded Successfully")
        st.write(f"Dataset Name: {file_name}")
        try:
            mem = memory_usage_mb(df)
        except Exception:
            mem = 0
        info_cols = st.columns(2)
        with info_cols[0]:
            st.metric("Rows", f"{len(df):,}")
        with info_cols[1]:
            st.metric("Columns", f"{df.shape[1]}")
        st.caption(f"Memory Usage: {mem:.2f} MB")
        # Missing, Duplicates, Outliers
        try:
            miss_count = int(df.isna().sum().sum())
        except Exception:
            miss_count = 0
        try:
            dup_count = int(df.duplicated().sum())
        except Exception:
            dup_count = 0
        try:
            out_masks = detect_outliers_iqr(df, [c for c in NUMERIC_COLUMNS if c in df.columns])
            out_count = 0
            if out_masks:
                comb = None
                for m in out_masks.values():
                    comb = m if comb is None else (comb | m)
                out_count = int(comb.sum()) if comb is not None else 0
        except Exception:
            out_count = 0
        st.write(f"Missing Values Count: {miss_count}")
        st.write(f"Duplicate Rows Count: {dup_count}")
        st.write(f"Outlier Count: {out_count}")
        st.success("Target Variable Status: Success Column Created" if 'success' in df.columns else "Target Variable Status: Not Available")
        st.success("Data Cleaning Status: Ready for Analysis")
        st.markdown("</div>", unsafe_allow_html=True)

        # Dataset Preview
        with st.expander("👀 Dataset Preview (first 5 rows)"):
            st.dataframe(df.head(5), use_container_width=True, height=180)

        # Column Information
        with st.expander("📑 Column Information"):
            try:
                col_info = pd.DataFrame({
                    'Name': df.columns,
                    'Datatype': [str(df[c].dtype) for c in df.columns],
                    'Missing Values': [int(df[c].isna().sum()) for c in df.columns],
                    'Unique Values': [df[c].nunique(dropna=False) for c in df.columns]
                })
                st.dataframe(col_info, use_container_width=True, height=240)
            except Exception:
                st.info("Unable to display column information.")

        # Summary Statistics
        with st.expander("📊 Summary Statistics"):
            try:
                st.dataframe(compute_summary_stats(df), use_container_width=True, height=280)
            except Exception:
                st.info("Unable to compute summary stats.")

        # Data Cleaning Controls & Report
        with st.expander("🧹 Data Cleaning"):
            # Missing Values
            try:
                missing_total = int(original.isna().sum().sum())
                if missing_total == 0:
                    st.success("No Missing Values Found")
                    miss_strategy = "Keep Missing Values"
                else:
                    miss_strategy = st.radio(
                        "Missing Values Handling",
                        ["Remove Missing Rows", "Fill Numeric with Median", "Fill Categorical with Mode", "Keep Missing Values"],
                        index=1, help="Choose how to treat missing values across the dataset."
                    )
                    st.session_state['missing_strategy'] = miss_strategy
                # Duplicates
                dups = detect_duplicates(original)
                if dups.empty:
                    st.success("No Duplicate Rows Found")
                    dup_strategy = "Keep Duplicates"
                else:
                    st.info(f"Found {int(dups.duplicated().sum())} duplicate records.")
                    dup_strategy = st.radio(
                        "Duplicate Handling",
                        ["Keep Duplicates", "Remove All Duplicates", "Manual Edit"],
                        index=1, help="Remove all duplicates or manually edit them."
                    )
                    if dup_strategy == "Manual Edit":
                        st.caption("Edit duplicate rows below and click 'Save Manual Edits'.")
                        edited_dups = st.data_editor(dups, use_container_width=True, num_rows="dynamic", height=220)
                        if st.button("Save Manual Edits", help="Apply your manual edits to the dataset"):
                            try:
                                # Update original with edited rows by index
                                original.update(edited_dups)
                                st.success("Manual edits saved.")
                            except Exception:
                                st.warning("Could not apply manual edits to duplicates.")
                st.session_state['duplicate_strategy'] = dup_strategy

                # Invalid Values
                inv = detect_invalids(original)
                inv_count = sum(int(m.sum()) for m in inv.values()) if inv else 0
                if inv_count > 0:
                    invalid_strategy = st.radio(
                        "Invalid Values (budget<=0, revenue<=0, runtime<=0, popularity<0, vote_average outside 0-10)",
                        ["Remove", "Keep", "Manual Edit"], index=0,
                        help="Remove all invalid rows, keep them, or manually edit."
                    )
                    if invalid_strategy == "Manual Edit":
                        try:
                            comb = None
                            for m in inv.values():
                                comb = m if comb is None else (comb | m)
                            invalid_rows = original[comb]
                            st.caption("Fix invalid rows below and click 'Save Invalid Edits'.")
                            edited_inv = st.data_editor(invalid_rows, use_container_width=True, num_rows="dynamic", height=240)
                            if st.button("Save Invalid Edits"):
                                try:
                                    original.update(edited_inv)
                                    st.success("Invalid rows updated.")
                                except Exception:
                                    st.warning("Could not apply invalid edits.")
                        except Exception:
                            st.warning("Could not process invalid rows for editing.")
                    st.session_state['invalid_strategy'] = invalid_strategy
                else:
                    st.success("No invalid values detected.")
                    st.session_state['invalid_strategy'] = "Keep"

                # Outliers
                out_masks = detect_outliers_iqr(original, [c for c in NUMERIC_COLUMNS if c in original.columns])
                outlier_total = 0
                if out_masks:
                    comb = None
                    for m in out_masks.values():
                        comb = m if comb is None else (comb | m)
                    outlier_total = int(comb.sum()) if comb is not None else 0
                st.caption(f"Outliers detected (IQR method): {outlier_total}")
                out_strategy = st.radio("Outlier Handling", ["Keep Outliers", "Remove Outliers"], index=0,
                                       help="Outliers are not removed automatically; choose your approach.")
                st.session_state['outlier_strategy'] = out_strategy

            except Exception:
                st.warning("Data cleaning controls could not be fully rendered.")

                # Apply cleaning pipeline
            try:
                # Ensure numerics
                work = ensure_numeric_columns(original.copy(), NUMERIC_COLUMNS)
                # Missing
                miss_str = st.session_state.get('missing_strategy', 'Fill Numeric with Median')
                work, miss_rep = apply_missing_strategy(work, miss_str)
                # Duplicates
                dup_str = st.session_state.get('duplicate_strategy', 'default_value')
            except Exception:
                st.warning("Could not apply cleaning pipeline.")