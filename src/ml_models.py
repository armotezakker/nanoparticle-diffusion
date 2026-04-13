"""
ml_models.py
------------
Supervised classification of diffusion regimes using trajectory-derived
features. Two ensemble methods are compared: Random Forest and Gradient
Boosting. Grid search over hyperparameters is used to select the best model
for each algorithm.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import (
    GridSearchCV, train_test_split, learning_curve, cross_val_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score


def prepare_data(feature_df: pd.DataFrame,
                 target: str = 'diffusion_regime',
                 test_size: float = 0.2,
                 random_state: int = 42):
    """Prepare feature matrix and labels for supervised classification.

    Non-feature columns (run_id and the target) are dropped. All numeric
    feature columns and the physical parameters (polymer_conc, np_charge)
    are retained as predictors. The target labels are integer-encoded.
    A stratified train/test split preserves the class distribution in both
    subsets. Features are standardised (zero mean, unit variance) on the
    training set and the same transform is applied to the test set.

    Parameters
    ----------
    feature_df : pd.DataFrame
        Output of build_feature_matrix.
    target : str
        Column name of the class label. Default 'diffusion_regime'.
    test_size : float
        Fraction of data reserved for the test set. Default 0.2.
    random_state : int
        Random seed for reproducibility. Default 42.

    Returns
    -------
    X_train, X_test : np.ndarray
        Scaled feature matrices.
    y_train, y_test : np.ndarray
        Integer-encoded labels.
    scaler : StandardScaler
        Fitted scaler (apply to new data before prediction).
    feature_names : list of str
        Names of the feature columns, in the same order as X columns.
    label_encoder : LabelEncoder
        Fitted encoder (use .inverse_transform to recover string labels).
    """
    drop_cols = {'run_id', target}
    feature_names = [c for c in feature_df.columns if c not in drop_cols]

    X = feature_df[feature_names].to_numpy(dtype=float)
    y_raw = feature_df[target].to_numpy()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler, feature_names, label_encoder


def train_random_forest(X_train: np.ndarray,
                        y_train: np.ndarray,
                        random_state: int = 42) -> tuple:
    """Train a Random Forest classifier with hyperparameter search.

    Grid search explores combinations of tree count, depth, and minimum
    leaf size. Five-fold cross-validation is used for scoring; macro-averaged
    F1 is chosen because the class distribution may be unbalanced and we care
    equally about all four regimes.

    Parameters
    ----------
    X_train : np.ndarray
        Scaled training features.
    y_train : np.ndarray
        Integer-encoded training labels.
    random_state : int
        Random seed. Default 42.

    Returns
    -------
    best_model : RandomForestClassifier
        Fitted model with the best hyperparameters.
    cv_results_df : pd.DataFrame
        Full grid search results table.
    """
    param_grid = {
        'n_estimators'    : [100, 200, 300],
        'max_depth'       : [None, 10, 20],
        'min_samples_leaf': [1, 2, 4],
    }
    rf = RandomForestClassifier(random_state=random_state)
    gs = GridSearchCV(
        rf, param_grid, cv=5, scoring='f1_macro', n_jobs=-1, refit=True
    )
    gs.fit(X_train, y_train)

    print('Random Forest -- best parameters:', gs.best_params_)
    print(f'Random Forest -- best CV F1 macro: {gs.best_score_:.4f}')

    cv_results_df = pd.DataFrame(gs.cv_results_)
    return gs.best_estimator_, cv_results_df


def train_gradient_boosting(X_train: np.ndarray,
                             y_train: np.ndarray,
                             random_state: int = 42) -> tuple:
    """Train a Gradient Boosting classifier with hyperparameter search.

    Grid search explores learning rate, number of boosting rounds, and tree
    depth. Slower learning rates with more rounds often generalise better in
    low-sample-size regimes such as this one (128 training trajectories).

    Parameters
    ----------
    X_train : np.ndarray
        Scaled training features.
    y_train : np.ndarray
        Integer-encoded training labels.
    random_state : int
        Random seed. Default 42.

    Returns
    -------
    best_model : GradientBoostingClassifier
        Fitted model with the best hyperparameters.
    cv_results_df : pd.DataFrame
        Full grid search results table.
    """
    param_grid = {
        'n_estimators' : [100, 200],
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth'    : [3, 5],
    }
    gb = GradientBoostingClassifier(random_state=random_state)
    gs = GridSearchCV(
        gb, param_grid, cv=5, scoring='f1_macro', n_jobs=-1, refit=True
    )
    gs.fit(X_train, y_train)

    print('Gradient Boosting -- best parameters:', gs.best_params_)
    print(f'Gradient Boosting -- best CV F1 macro: {gs.best_score_:.4f}')

    cv_results_df = pd.DataFrame(gs.cv_results_)
    return gs.best_estimator_, cv_results_df


def evaluate_model(model,
                   X_test: np.ndarray,
                   y_test: np.ndarray,
                   label_encoder: LabelEncoder,
                   model_name: str) -> dict:
    """Evaluate a trained classifier on the test set.

    Prints a full per-class classification report and returns summary metrics.

    Parameters
    ----------
    model
        Fitted sklearn classifier.
    X_test : np.ndarray
        Scaled test features.
    y_test : np.ndarray
        Integer-encoded test labels.
    label_encoder : LabelEncoder
        Used to recover class name strings for the report.
    model_name : str
        Human-readable name printed in the report header.

    Returns
    -------
    dict with keys:
        accuracy, f1_macro, confusion_matrix (np.ndarray).
    """
    y_pred = model.predict(X_test)
    class_names = label_encoder.classes_

    print(f'\n=== {model_name} ===')
    print(classification_report(
        y_test, y_pred, target_names=class_names
    ))

    cm = confusion_matrix(y_test, y_pred)
    acc = (y_pred == y_test).mean()
    f1  = f1_score(y_test, y_pred, average='macro')

    return {
        'accuracy'        : float(acc),
        'f1_macro'        : float(f1),
        'confusion_matrix': cm,
    }


def get_cv_scores(model,
                  X_train: np.ndarray,
                  y_train: np.ndarray,
                  cv: int = 10) -> np.ndarray:
    """Return cross-validation accuracy scores.

    Parameters
    ----------
    model
        Fitted or unfitted sklearn classifier.
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training labels.
    cv : int
        Number of folds. Default 10.

    Returns
    -------
    np.ndarray of float, shape (cv,)
        Accuracy for each fold.
    """
    scores = cross_val_score(model, X_train, y_train,
                             cv=cv, scoring='accuracy', n_jobs=-1)
    return scores
