"""
Thyroid Disease Detection — Production ML Pipeline v2
======================================================
A complete, production-quality pipeline for multi-class thyroid disease
classification using the real diagnostic target labels from the dataset.

Author: ML Pipeline Redesign
Date: 2026-07-18
"""

# ============================================================
# 0. CONFIGURATION
# ============================================================
import os
import time
import logging
import warnings
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_validate,
    learning_curve, validation_curve
)
from sklearn.preprocessing import (
    LabelEncoder, StandardScaler, OrdinalEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, balanced_accuracy_score,
    roc_auc_score, roc_curve, auc, precision_recall_curve,
    average_precision_score
)
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, HistGradientBoostingClassifier,
    VotingClassifier, StackingClassifier, BaggingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance

import joblib

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ── Config ──────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
N_JOBS = -1

# Paths (relative to project root)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_PATH = PROJECT_ROOT / "src" / "data" / "thyroidDF.csv"
MODEL_OUTPUT_PATH = SCRIPT_DIR / "thyroid_model_v2.pkl"
LABEL_ENCODER_PATH = SCRIPT_DIR / "label_encoder_v2.pkl"
FIGURES_DIR = SCRIPT_DIR / "figures"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)
np.random.seed(RANDOM_STATE)


# ============================================================
# 1. DATA LOADING & CLEANING
# ============================================================

def load_and_clean_data(path: Path) -> pd.DataFrame:
    """Load the raw thyroid dataset and perform initial cleaning."""
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path)
    logger.info(f"Raw dataset shape: {df.shape}")

    # ── Drop columns with >90% missing (TBG is 96% null) ───
    high_missing_cols = ['TBG', 'TBG_measured']
    df = df.drop(columns=high_missing_cols, errors='ignore')
    logger.info(f"Dropped high-missing columns: {high_missing_cols}")

    # ── Drop measurement flags (perfectly correlated with nulls) ──
    measurement_flags = [
        'TSH_measured', 'T3_measured', 'TT4_measured',
        'T4U_measured', 'FTI_measured'
    ]
    df = df.drop(columns=measurement_flags, errors='ignore')
    logger.info(f"Dropped redundant measurement flags: {measurement_flags}")

    # ── Drop patient_id (identifier, not a feature) ──
    df = df.drop(columns=['patient_id'], errors='ignore')

    # ── Clean age outliers (cap at 120) ──
    outlier_mask = df['age'] > 120
    n_outliers = outlier_mask.sum()
    if n_outliers > 0:
        df.loc[outlier_mask, 'age'] = np.nan  # Will be imputed
        logger.info(f"Set {n_outliers} age outliers (>120) to NaN for imputation")

    # ── Convert age to float for consistent handling ──
    df['age'] = df['age'].astype(float)

    logger.info(f"Cleaned dataset shape: {df.shape}")
    return df


# ============================================================
# 2. TARGET ENGINEERING
# ============================================================

# Medical reference for thyroid diagnosis codes:
# - "-"  = Negative (normal thyroid function)
# - A, B, C, D, AK, GK, GI, GD = Hyperthyroid conditions
# - E, F, G, H, S, FK, FI = Hypothyroid conditions
# - I, J, K, L, M, N, O, P, Q, R = Binding protein / replacement / other
# Multi-label codes like "C|I", "H|K", "MK" take the first condition

TARGET_GROUPING = {
    # Negative
    '-': 'Negative',
    # Hyperthyroid
    'A': 'Hyperthyroid', 'B': 'Hyperthyroid', 'C': 'Hyperthyroid',
    'D': 'Hyperthyroid', 'AK': 'Hyperthyroid', 'GK': 'Hyperthyroid',
    'GI': 'Hyperthyroid', 'GD': 'Hyperthyroid', 'GKJ': 'Hyperthyroid',
    # Hypothyroid
    'E': 'Hypothyroid', 'F': 'Hypothyroid', 'G': 'Hypothyroid',
    'H': 'Hypothyroid', 'S': 'Hypothyroid', 'FK': 'Hypothyroid',
    'FI': 'Hypothyroid',
    # Binding protein / sick euthyroid / other
    'I': 'Other', 'J': 'Other', 'K': 'Other', 'L': 'Other',
    'M': 'Other', 'N': 'Other', 'O': 'Other', 'P': 'Other',
    'Q': 'Other', 'R': 'Other',
    'C|I': 'Hyperthyroid', 'H|K': 'Hypothyroid',
    'MK': 'Other', 'KJ': 'Other', 'MI': 'Other',
    'LJ': 'Other', 'OI': 'Other', 'D|R': 'Hyperthyroid',
}


def engineer_target(df: pd.DataFrame) -> pd.DataFrame:
    """Map raw diagnostic codes to 4-class groups."""
    df = df.copy()
    df['target_original'] = df['target']
    df['target'] = df['target'].map(TARGET_GROUPING)

    unmapped = df['target'].isna().sum()
    if unmapped > 0:
        logger.warning(f"{unmapped} rows have unmapped target codes — dropping")
        df = df.dropna(subset=['target'])

    logger.info(f"Target distribution after grouping:\n{df['target'].value_counts().to_string()}")
    return df


# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create domain-informed features."""
    df = df.copy()

    # ── TSH/T3 ratio (clinically meaningful) ──
    df['TSH_T3_ratio'] = df['TSH'] / (df['T3'] + 1e-6)

    # ── TSH/TT4 ratio ──
    df['TSH_TT4_ratio'] = df['TSH'] / (df['TT4'] + 1e-6)

    # ── T3/TT4 ratio (free T3 proxy) ──
    df['T3_TT4_ratio'] = df['T3'] / (df['TT4'] + 1e-6)

    # ── FTI already exists but compute T4U*TT4 interaction ──
    df['T4U_TT4_interaction'] = df['T4U'] * df['TT4']

    # ── Age bins (pediatric, adult, elderly) ──
    df['age_group'] = pd.cut(
        df['age'],
        bins=[0, 18, 45, 65, 120],
        labels=['pediatric', 'young_adult', 'middle_aged', 'elderly'],
        right=True
    ).astype(str)
    df['age_group'] = df['age_group'].replace('nan', 'unknown')

    # ── Flag: is any lab value extreme? ──
    df['extreme_TSH'] = ((df['TSH'] > 10) | (df['TSH'] < 0.1)).astype(int)
    df['extreme_T3'] = ((df['T3'] > 4) | (df['T3'] < 0.5)).astype(int)
    df['extreme_TT4'] = ((df['TT4'] > 200) | (df['TT4'] < 40)).astype(int)

    # ── Missingness indicator features (the fact that a test wasn't ordered is informative) ──
    for col in ['TSH', 'T3', 'TT4', 'T4U', 'FTI']:
        df[f'{col}_missing'] = df[col].isna().astype(int)

    logger.info(f"Feature-engineered dataset shape: {df.shape}")
    return df


# ============================================================
# 4. PREPROCESSING PIPELINE
# ============================================================

# Define feature groups
NUMERIC_FEATURES = [
    'age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI',
    'TSH_T3_ratio', 'TSH_TT4_ratio', 'T3_TT4_ratio',
    'T4U_TT4_interaction'
]

BINARY_FEATURES = [
    'on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_meds',
    'sick', 'pregnant', 'thyroid_surgery', 'I131_treatment',
    'query_hypothyroid', 'query_hyperthyroid', 'lithium',
    'goitre', 'tumor', 'hypopituitary', 'psych',
    'extreme_TSH', 'extreme_T3', 'extreme_TT4',
    'TSH_missing', 'T3_missing', 'TT4_missing',
    'T4U_missing', 'FTI_missing'
]

CATEGORICAL_FEATURES = [
    'sex', 'referral_source', 'age_group'
]

DROP_FEATURES = ['target', 'target_original']


def build_preprocessor() -> ColumnTransformer:
    """Build a ColumnTransformer for all feature types."""
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    binary_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        # Binary features (t/f or 0/1) are already numeric after mapping
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, NUMERIC_FEATURES),
            ('bin', binary_pipeline, BINARY_FEATURES),
            ('cat', categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder='drop',
        n_jobs=N_JOBS
    )
    return preprocessor


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Map binary t/f columns to 0/1."""
    df = df.copy()
    binary_tf_cols = [
        'on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_meds',
        'sick', 'pregnant', 'thyroid_surgery', 'I131_treatment',
        'query_hypothyroid', 'query_hyperthyroid', 'lithium',
        'goitre', 'tumor', 'hypopituitary', 'psych'
    ]
    for col in binary_tf_cols:
        if col in df.columns:
            df[col] = df[col].map({'t': 1, 'f': 0}).fillna(0).astype(int)
    return df


# ============================================================
# 5. MODEL DEFINITIONS
# ============================================================

def get_models() -> Dict[str, Any]:
    """Return a dictionary of models to compare."""
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_split=5,
            min_samples_leaf=2, class_weight='balanced',
            random_state=RANDOM_STATE, n_jobs=N_JOBS
        ),
        'ExtraTrees': ExtraTreesClassifier(
            n_estimators=200, max_depth=15, min_samples_split=5,
            min_samples_leaf=2, class_weight='balanced',
            random_state=RANDOM_STATE, n_jobs=N_JOBS
        ),
        'HistGradientBoosting': HistGradientBoostingClassifier(
            max_iter=300, max_depth=8, learning_rate=0.1,
            min_samples_leaf=10, class_weight='balanced',
            random_state=RANDOM_STATE
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            min_samples_split=5, min_samples_leaf=2,
            random_state=RANDOM_STATE
        ),
        'LogisticRegression': LogisticRegression(
            max_iter=1000, class_weight='balanced', multi_class='multinomial',
            solver='lbfgs', random_state=RANDOM_STATE, n_jobs=N_JOBS
        ),
        'KNN': KNeighborsClassifier(
            n_neighbors=7, weights='distance', n_jobs=N_JOBS
        ),
        'DecisionTree': DecisionTreeClassifier(
            max_depth=10, min_samples_split=10, min_samples_leaf=5,
            class_weight='balanced', random_state=RANDOM_STATE
        ),
        'NaiveBayes': GaussianNB(),
        'BaggingRF': BaggingClassifier(
            estimator=RandomForestClassifier(
                n_estimators=50, max_depth=10, class_weight='balanced',
                random_state=RANDOM_STATE
            ),
            n_estimators=10, random_state=RANDOM_STATE, n_jobs=N_JOBS
        ),
    }

    # Conditionally add XGBoost, LightGBM, CatBoost
    try:
        from xgboost import XGBClassifier
        models['XGBoost'] = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, use_label_encoder=False,
            eval_metric='mlogloss', random_state=RANDOM_STATE,
            n_jobs=N_JOBS
        )
    except Exception as e:
        logger.warning(f"XGBoost unavailable — skipping ({e})")

    try:
        from lightgbm import LGBMClassifier
        models['LightGBM'] = LGBMClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.1,
            min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, class_weight='balanced',
            random_state=RANDOM_STATE, n_jobs=N_JOBS, verbose=-1
        )
    except Exception as e:
        logger.warning(f"LightGBM unavailable — skipping ({e})")

    try:
        from catboost import CatBoostClassifier
        models['CatBoost'] = CatBoostClassifier(
            iterations=200, depth=6, learning_rate=0.1,
            auto_class_weights='Balanced', random_seed=RANDOM_STATE,
            verbose=0
        )
    except Exception as e:
        logger.warning(f"CatBoost unavailable — skipping ({e})")

    return models


# ============================================================
# 6. EVALUATION
# ============================================================

def evaluate_model(
    model, X_test: np.ndarray, y_test: np.ndarray,
    label_encoder: LabelEncoder, model_name: str
) -> Dict[str, float]:
    """Compute comprehensive evaluation metrics."""
    y_pred = model.predict(X_test)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
        'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'precision_weighted': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall_weighted': recall_score(y_test, y_pred, average='weighted', zero_division=0),
    }

    # ROC-AUC (one-vs-rest)
    try:
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
            metrics['roc_auc_ovr'] = roc_auc_score(
                y_test, y_proba, multi_class='ovr', average='macro'
            )
        else:
            metrics['roc_auc_ovr'] = np.nan
    except Exception:
        metrics['roc_auc_ovr'] = np.nan

    return metrics


def print_classification_report(
    model, X_test: np.ndarray, y_test: np.ndarray,
    label_encoder: LabelEncoder, model_name: str
):
    """Print detailed classification report."""
    y_pred = model.predict(X_test)
    target_names = label_encoder.classes_
    print(f"\n{'=' * 60}")
    print(f"Classification Report: {model_name}")
    print(f"{'=' * 60}")
    print(classification_report(
        y_test, y_pred, target_names=target_names, zero_division=0
    ))


def plot_confusion_matrix(
    model, X_test: np.ndarray, y_test: np.ndarray,
    label_encoder: LabelEncoder, model_name: str, save_dir: Path
):
    """Plot and save confusion matrix."""
    y_pred = model.predict(X_test)
    target_names = label_encoder.classes_
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=target_names, yticklabels=target_names, ax=ax
    )
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix — {model_name}')
    plt.tight_layout()
    save_path = save_dir / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved confusion matrix: {save_path}")


def plot_roc_curves(
    model, X_test: np.ndarray, y_test: np.ndarray,
    label_encoder: LabelEncoder, model_name: str, save_dir: Path
):
    """Plot multi-class ROC curves (one-vs-rest)."""
    if not hasattr(model, 'predict_proba'):
        logger.info(f"Skipping ROC for {model_name} (no predict_proba)")
        return

    y_proba = model.predict_proba(X_test)
    n_classes = len(label_encoder.classes_)

    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(n_classes):
        y_binary = (y_test == i).astype(int)
        fpr, tpr, _ = roc_curve(y_binary, y_proba[:, i])
        roc_auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'{label_encoder.classes_[i]} (AUC={roc_auc_val:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curves — {model_name}')
    ax.legend(loc='lower right')
    plt.tight_layout()
    save_path = save_dir / f"roc_curves_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved ROC curves: {save_path}")


def plot_precision_recall_curves(
    model, X_test: np.ndarray, y_test: np.ndarray,
    label_encoder: LabelEncoder, model_name: str, save_dir: Path
):
    """Plot multi-class Precision-Recall curves."""
    if not hasattr(model, 'predict_proba'):
        return

    y_proba = model.predict_proba(X_test)
    n_classes = len(label_encoder.classes_)

    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(n_classes):
        y_binary = (y_test == i).astype(int)
        prec, rec, _ = precision_recall_curve(y_binary, y_proba[:, i])
        ap = average_precision_score(y_binary, y_proba[:, i])
        ax.plot(rec, prec, label=f'{label_encoder.classes_[i]} (AP={ap:.3f})')

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall Curves — {model_name}')
    ax.legend(loc='lower left')
    plt.tight_layout()
    save_path = save_dir / f"pr_curves_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved PR curves: {save_path}")


def plot_feature_importance(
    model, feature_names: List[str], model_name: str, save_dir: Path
):
    """Plot feature importance for tree-based models."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        logger.info(f"Skipping feature importance for {model_name}")
        return

    indices = np.argsort(importances)[::-1][:20]  # Top 20

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(
        range(len(indices)),
        importances[indices],
        align='center', color='steelblue'
    )
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    ax.set_title(f'Feature Importance (Top 20) — {model_name}')
    plt.tight_layout()
    save_path = save_dir / f"feature_importance_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved feature importance: {save_path}")


def plot_learning_curve_fig(
    model, X_train: np.ndarray, y_train: np.ndarray,
    model_name: str, save_dir: Path
):
    """Plot learning curve for the model."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train, cv=cv,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='f1_macro', n_jobs=N_JOBS
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='orange')
    ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
    ax.plot(train_sizes, val_mean, 'o-', color='orange', label='Validation score')
    ax.set_xlabel('Training set size')
    ax.set_ylabel('F1 Macro Score')
    ax.set_title(f'Learning Curve — {model_name}')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = save_dir / f"learning_curve_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved learning curve: {save_path}")


def plot_calibration_curve_fig(
    model, X_test: np.ndarray, y_test: np.ndarray,
    label_encoder: LabelEncoder, model_name: str, save_dir: Path
):
    """Plot calibration curves for each class."""
    if not hasattr(model, 'predict_proba'):
        return

    y_proba = model.predict_proba(X_test)
    n_classes = len(label_encoder.classes_)

    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(n_classes):
        y_binary = (y_test == i).astype(int)
        if y_binary.sum() < 5:
            continue
        prob_true, prob_pred = calibration_curve(y_binary, y_proba[:, i], n_bins=10)
        ax.plot(prob_pred, prob_true, 's-', label=label_encoder.classes_[i])

    ax.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')
    ax.set_title(f'Calibration Curve — {model_name}')
    ax.legend(loc='lower right')
    plt.tight_layout()
    save_path = save_dir / f"calibration_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved calibration curve: {save_path}")


# ============================================================
# 7. EXPLAINABILITY (SHAP)
# ============================================================

def run_shap_analysis(
    model, X_test: np.ndarray, feature_names: List[str],
    model_name: str, save_dir: Path
):
    """Run SHAP analysis for the best model."""
    try:
        import shap
    except ImportError:
        logger.warning("SHAP not installed — skipping explainability")
        return

    logger.info(f"Running SHAP analysis for {model_name}...")

    # Use a sample for speed
    sample_size = min(200, X_test.shape[0])
    X_sample = X_test[:sample_size]

    try:
        # Try TreeExplainer first (fast, exact, but limited compatibility)
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            logger.info("Using TreeExplainer (fast)")
        except Exception:
            # Fall back to KernelExplainer (universal, slower)
            logger.info("TreeExplainer failed, falling back to KernelExplainer...")
            background = shap.kmeans(X_test[:min(50, X_test.shape[0])], 20)
            explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_values = explainer.shap_values(X_sample, nsamples=100)

        # For multi-class, shap_values is a list of arrays (one per class)
        if isinstance(shap_values, list):
            mean_abs_shap = np.mean([np.abs(sv) for sv in shap_values], axis=0)
        else:
            mean_abs_shap = np.abs(shap_values)

        # Bar plot of mean |SHAP| values
        mean_importance = mean_abs_shap.mean(axis=0)
        sorted_idx = np.argsort(mean_importance)[::-1][:20]

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(
            range(len(sorted_idx)),
            mean_importance[sorted_idx],
            align='center', color='coral'
        )
        ax.set_yticks(range(len(sorted_idx)))
        ax.set_yticklabels([feature_names[i] for i in sorted_idx])
        ax.invert_yaxis()
        ax.set_xlabel('Mean |SHAP value|')
        ax.set_title(f'SHAP Feature Importance — {model_name}')
        plt.tight_layout()
        save_path = save_dir / f"shap_bar_{model_name.lower().replace(' ', '_')}.png"
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved SHAP bar plot: {save_path}")

        # Summary beeswarm for first class
        if isinstance(shap_values, list) and len(shap_values) > 0:
            fig, ax = plt.subplots(figsize=(12, 8))
            shap.summary_plot(
                shap_values[0], X_sample,
                feature_names=feature_names, show=False,
                max_display=20
            )
            plt.title(f'SHAP Summary (Class 0: {label_encoder.classes_[0]}) — {model_name}')
            plt.tight_layout()
            save_path = save_dir / f"shap_summary_{model_name.lower().replace(' ', '_')}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved SHAP summary: {save_path}")
    except Exception as e:
        logger.warning(f"SHAP analysis failed: {e}")


# ============================================================
# 8. HYPERPARAMETER TUNING (Optuna)
# ============================================================

def tune_with_optuna(
    X_train: np.ndarray, y_train: np.ndarray,
    n_trials: int = 50
) -> Dict[str, Any]:
    """Tune the best model using Optuna."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.warning("Optuna not installed — skipping hyperparameter tuning")
        return {}

    logger.info(f"Running Optuna hyperparameter tuning ({n_trials} trials)...")

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 4, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'class_weight': 'balanced',
            'random_state': RANDOM_STATE,
            'n_jobs': N_JOBS,
        }

        model = RandomForestClassifier(**params)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_validate(
            model, X_train, y_train, cv=cv,
            scoring='f1_macro', n_jobs=1
        )
        return scores['test_score'].mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info(f"Best Optuna trial: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")

    return study.best_params


# ============================================================
# 9. MAIN PIPELINE
# ============================================================

def main():
    """Execute the complete ML pipeline."""
    start_time = time.time()

    # ── Create output directories ──
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────────────────
    # STEP 1: Load & Clean
    # ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1: Loading and cleaning data")
    logger.info("=" * 60)
    df = load_and_clean_data(DATA_PATH)

    # ──────────────────────────────────────────────────────────
    # STEP 2: Target Engineering
    # ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2: Engineering target variable")
    logger.info("=" * 60)
    df = engineer_target(df)

    # ──────────────────────────────────────────────────────────
    # STEP 3: Feature Engineering
    # ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3: Feature engineering")
    logger.info("=" * 60)
    df = engineer_features(df)
    df = prepare_features(df)

    # ──────────────────────────────────────────────────────────
    # STEP 4: Encode target & Split
    # ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 4: Encoding target and splitting data")
    logger.info("=" * 60)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['target'])
    logger.info(f"Target classes: {list(label_encoder.classes_)}")

    X = df.drop(columns=DROP_FEATURES, errors='ignore')

    # Validate feature columns
    all_features = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES
    missing_features = [f for f in all_features if f not in X.columns]
    if missing_features:
        logger.error(f"Missing features: {missing_features}")
        raise ValueError(f"Missing features in dataframe: {missing_features}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    logger.info(f"Train target distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    logger.info(f"Test target distribution: {dict(zip(*np.unique(y_test, return_counts=True)))}")

    # ──────────────────────────────────────────────────────────
    # STEP 5: Preprocessing (fit on train only!)
    # ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 5: Preprocessing (fit on training data only)")
    logger.info("=" * 60)
    preprocessor = build_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Get feature names after transformation
    feature_names = (
        NUMERIC_FEATURES +
        BINARY_FEATURES +
        CATEGORICAL_FEATURES
    )
    logger.info(f"Processed features: {X_train_processed.shape[1]}")

    # ──────────────────────────────────────────────────────────
    # STEP 6: Handle Class Imbalance (SMOTE on train only)
    # ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 6: Handling class imbalance")
    logger.info("=" * 60)
    try:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=3)
        X_train_resampled, y_train_resampled = smote.fit_resample(
            X_train_processed, y_train
        )
        logger.info(f"Before SMOTE: {dict(zip(*np.unique(y_train, return_counts=True)))}")
        logger.info(f"After SMOTE:  {dict(zip(*np.unique(y_train_resampled, return_counts=True)))}")
    except ImportError:
        logger.warning("imbalanced-learn not installed — using class_weight='balanced' only")
        X_train_resampled, y_train_resampled = X_train_processed, y_train

    # ──────────────────────────────────────────────────────────
    # STEP 7: Model Comparison
    # ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 7: Comparing models")
    logger.info("=" * 60)

    models = get_models()
    results = {}

    for name, model in models.items():
        logger.info(f"Training {name}...")
        t0 = time.time()
        try:
            model.fit(X_train_resampled, y_train_resampled)
            train_time = time.time() - t0

            metrics = evaluate_model(
                model, X_test_processed, y_test, label_encoder, name
            )
            metrics['train_time_sec'] = round(train_time, 3)

            # Inference time
            t1 = time.time()
            _ = model.predict(X_test_processed)
            metrics['inference_time_sec'] = round(time.time() - t1, 4)

            results[name] = metrics
            logger.info(
                f"  {name}: Acc={metrics['accuracy']:.4f}, "
                f"F1_macro={metrics['f1_macro']:.4f}, "
                f"BAcc={metrics['balanced_accuracy']:.4f}, "
                f"Train={train_time:.2f}s"
            )
        except Exception as e:
            logger.error(f"  {name} failed: {e}")

    # ── Results comparison table ──
    results_df = pd.DataFrame(results).T.sort_values('f1_macro', ascending=False)
    print("\n" + "=" * 80)
    print("MODEL COMPARISON RESULTS")
    print("=" * 80)
    print(results_df.to_string(float_format='%.4f'))
    print()

    # ──────────────────────────────────────────────────────────
    # STEP 8: Cross-Validation on Top Models
    # ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 8: Cross-validation on top 3 models")
    logger.info("=" * 60)

    top_models = results_df.head(3).index.tolist()
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_results = {}

    for name in top_models:
        model = get_models()[name]
        try:
            scores = cross_validate(
                model, X_train_resampled, y_train_resampled, cv=cv,
                scoring=['accuracy', 'f1_macro', 'balanced_accuracy'],
                n_jobs=N_JOBS, return_train_score=True
            )
            cv_results[name] = {
                'cv_accuracy_mean': scores['test_accuracy'].mean(),
                'cv_accuracy_std': scores['test_accuracy'].std(),
                'cv_f1_macro_mean': scores['test_f1_macro'].mean(),
                'cv_f1_macro_std': scores['test_f1_macro'].std(),
                'cv_balanced_accuracy_mean': scores['test_balanced_accuracy'].mean(),
                'cv_balanced_accuracy_std': scores['test_balanced_accuracy'].std(),
                'train_f1_macro_mean': scores['train_f1_macro'].mean(),
            }
            logger.info(
                f"  {name}: CV F1_macro = "
                f"{scores['test_f1_macro'].mean():.4f} ± {scores['test_f1_macro'].std():.4f}"
            )
        except Exception as e:
            logger.error(f"  CV failed for {name}: {e}")

    cv_df = pd.DataFrame(cv_results).T
    print("\n" + "=" * 80)
    print("CROSS-VALIDATION RESULTS (Top 3)")
    print("=" * 80)
    print(cv_df.to_string(float_format='%.4f'))

    # ──────────────────────────────────────────────────────────
    # STEP 9: Select Best Model & Tune
    # ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 9: Selecting best model & hyperparameter tuning")
    logger.info("=" * 60)

    best_model_name = results_df.index[0]
    logger.info(f"Best model (by F1 macro): {best_model_name}")

    # Optuna tuning for RandomForest (our most reliably strong model)
    best_params = tune_with_optuna(X_train_resampled, y_train_resampled, n_trials=50)

    if best_params:
        best_params['class_weight'] = 'balanced'
        best_params['random_state'] = RANDOM_STATE
        best_params['n_jobs'] = N_JOBS
        tuned_model = RandomForestClassifier(**best_params)
        tuned_model.fit(X_train_resampled, y_train_resampled)
        tuned_metrics = evaluate_model(
            tuned_model, X_test_processed, y_test, label_encoder, 'Tuned_RF'
        )
        logger.info(
            f"Tuned RF: Acc={tuned_metrics['accuracy']:.4f}, "
            f"F1_macro={tuned_metrics['f1_macro']:.4f}"
        )

        # Compare tuned vs original best
        original_f1 = results[best_model_name]['f1_macro']
        tuned_f1 = tuned_metrics['f1_macro']
        if tuned_f1 > original_f1:
            logger.info(f"Tuned RF improved F1: {original_f1:.4f} → {tuned_f1:.4f}")
            final_model = tuned_model
            final_model_name = "Tuned_RandomForest"
        else:
            logger.info(f"Original {best_model_name} is still better: {original_f1:.4f} vs {tuned_f1:.4f}")
            final_model = models[best_model_name]
            final_model_name = best_model_name
    else:
        final_model = models[best_model_name]
        final_model_name = best_model_name

    # ──────────────────────────────────────────────────────────
    # STEP 10: Final Evaluation
    # ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 10: Final evaluation & visualization")
    logger.info("=" * 60)

    print_classification_report(
        final_model, X_test_processed, y_test, label_encoder, final_model_name
    )
    plot_confusion_matrix(
        final_model, X_test_processed, y_test, label_encoder,
        final_model_name, FIGURES_DIR
    )
    plot_roc_curves(
        final_model, X_test_processed, y_test, label_encoder,
        final_model_name, FIGURES_DIR
    )
    plot_precision_recall_curves(
        final_model, X_test_processed, y_test, label_encoder,
        final_model_name, FIGURES_DIR
    )
    plot_feature_importance(
        final_model, feature_names, final_model_name, FIGURES_DIR
    )
    plot_learning_curve_fig(
        final_model, X_train_resampled, y_train_resampled,
        final_model_name, FIGURES_DIR
    )
    plot_calibration_curve_fig(
        final_model, X_test_processed, y_test, label_encoder,
        final_model_name, FIGURES_DIR
    )

    # ──────────────────────────────────────────────────────────
    # STEP 11: SHAP Explainability
    # ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 11: Explainability (SHAP)")
    logger.info("=" * 60)
    run_shap_analysis(
        final_model, X_test_processed, feature_names,
        final_model_name, FIGURES_DIR
    )

    # ── Permutation Importance ──
    logger.info("Running permutation importance...")
    perm_imp = permutation_importance(
        final_model, X_test_processed, y_test,
        n_repeats=10, random_state=RANDOM_STATE, n_jobs=N_JOBS,
        scoring='f1_macro'
    )
    perm_imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': perm_imp.importances_mean,
        'importance_std': perm_imp.importances_std
    }).sort_values('importance_mean', ascending=False)
    print("\n" + "=" * 60)
    print("PERMUTATION IMPORTANCE (Top 15)")
    print("=" * 60)
    print(perm_imp_df.head(15).to_string(index=False))

    # ──────────────────────────────────────────────────────────
    # STEP 12: Build & Save Production Pipeline
    # ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 12: Building production pipeline")
    logger.info("=" * 60)

    # Create a full pipeline: preprocessor + model
    production_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', final_model)
    ])

    # Refit on full training data (preprocessor already fitted)
    # The pipeline is already trained, just save it
    joblib.dump(production_pipeline, MODEL_OUTPUT_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)
    logger.info(f"Saved production pipeline: {MODEL_OUTPUT_PATH}")
    logger.info(f"Saved label encoder: {LABEL_ENCODER_PATH}")

    # ── Save results summary ──
    summary = {
        'final_model': final_model_name,
        'test_metrics': evaluate_model(
            final_model, X_test_processed, y_test, label_encoder, final_model_name
        ),
        'classes': list(label_encoder.classes_),
        'n_features': X_train_processed.shape[1],
        'train_size': X_train.shape[0],
        'test_size': X_test.shape[0],
        'total_time_sec': round(time.time() - start_time, 1),
    }
    summary_path = SCRIPT_DIR / "pipeline_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Saved pipeline summary: {summary_path}")

    # ── Final Report ──
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"Best model: {final_model_name}")
    print(f"Test Accuracy: {summary['test_metrics']['accuracy']:.4f}")
    print(f"Test F1 (macro): {summary['test_metrics']['f1_macro']:.4f}")
    print(f"Test Balanced Accuracy: {summary['test_metrics']['balanced_accuracy']:.4f}")
    print(f"ROC-AUC (OVR): {summary['test_metrics'].get('roc_auc_ovr', 'N/A')}")
    print(f"Total pipeline time: {total_time:.1f}s")
    print(f"Model saved to: {MODEL_OUTPUT_PATH}")
    print(f"Figures saved to: {FIGURES_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
