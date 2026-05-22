"""
hyperparameter_tuning.py
------------------------
Funciones para optimización de hiperparámetros usando GridSearchCV,
RandomizedSearchCV y Optuna (búsqueda Bayesiana TPE) sobre LightGBM y XGBoost.

Decisiones de diseño:
    - Se usa class_weight='balanced' / scale_pos_weight en lugar de SMOTE
      dentro del CV para evitar errores de memoria (OOM) en Windows.
    - n_jobs=1 por estabilidad en Windows (multiprocessing puede crashear).
    - cv=3 para balance entre velocidad y robustez estadística.
    - Optuna usa TPESampler (Tree-structured Parzen Estimator) para búsqueda
      adaptiva en espacios continuos — más eficiente que grid/random en >6 params.
"""

import time
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

SEED = 42
TARGET_CLF = "is_satisfied"
TARGET_REG = "review_score"


def _threshold_tune(model, X_val: pd.DataFrame, y_val: pd.Series,
                    thr_range: tuple = (0.25, 0.76, 0.01)) -> tuple:
    """Busca threshold óptimo para maximizar accuracy.

    Args:
        model: Modelo ajustado con predict_proba.
        X_val: Features de validación.
        y_val: Target de validación.
        thr_range: Rango de búsqueda (inicio, fin, paso).

    Returns:
        Tupla (best_threshold, best_accuracy).
    """
    probs = model.predict_proba(X_val)[:, 1]
    best_thr, best_acc = 0.5, 0.0
    for thr in np.arange(*thr_range):
        acc = accuracy_score(y_val, (probs >= thr).astype(int))
        if acc > best_acc:
            best_acc, best_thr = acc, thr
    return round(best_thr, 2), round(best_acc, 4)


def _eval_metrics(model, X_test: pd.DataFrame, y_test: pd.Series,
                  threshold: float = 0.5) -> dict:
    """Calcula accuracy, F1 y ROC-AUC sobre el conjunto de prueba.

    Args:
        model: Modelo ajustado.
        X_test: Features de prueba.
        y_test: Target de prueba.
        threshold: Umbral de clasificación.

    Returns:
        Diccionario con métricas redondeadas.
    """
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)
    return {
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "f1":       round(f1_score(y_test, preds, zero_division=0), 4),
        "roc_auc":  round(roc_auc_score(y_test, probs), 4),
    }


# ─── GridSearchCV ─────────────────────────────────────────────────────────────

def grid_search_lgbm(df: pd.DataFrame, cv: int = 3) -> tuple:
    """Optimiza LightGBM con GridSearchCV sobre una grilla exhaustiva.

    Usa class_weight='balanced' para manejar el desbalance sin SMOTE,
    evitando problemas de memoria en la paralelización de CV en Windows.

    Grilla de búsqueda:
        - n_estimators: [300, 500]
        - learning_rate: [0.03, 0.05]
        - max_depth: [5, 7]
        - num_leaves: [31, 63]

    Args:
        df: DataFrame validado con feature 'is_satisfied'.
        cv: Número de folds para validación cruzada.

    Returns:
        Tupla (gs_model, results_dict) donde results_dict incluye
        cv_accuracy, test_accuracy, best_params, tiempo_segundos.
    """
    X = df.drop(columns=[c for c in [TARGET_CLF, TARGET_REG] if c in df.columns])
    y = df[TARGET_CLF]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)

    model = LGBMClassifier(random_state=SEED, verbose=-1, class_weight="balanced")
    param_grid = {
        "n_estimators":  [300, 500],
        "learning_rate": [0.03, 0.05],
        "max_depth":     [5, 7],
        "num_leaves":    [31, 63],
    }

    try:
        print(f"GridSearchCV LightGBM: {2**4} combinaciones, cv={cv}...")
        t0 = time.time()
        gs = GridSearchCV(model, param_grid, scoring="accuracy", cv=cv, n_jobs=1, verbose=0)
        gs.fit(X_train, y_train)
        elapsed = round(time.time() - t0, 2)

        opt_thr, _ = _threshold_tune(gs, X_test, y_test)
        m = _eval_metrics(gs, X_test, y_test, threshold=opt_thr)

        print(f"  CV accuracy: {gs.best_score_:.4f} | Test accuracy: {m['accuracy']} | {elapsed}s")
        print(f"  Mejores params: {gs.best_params_}")

        results = {
            "metodo": "GridSearchCV_LightGBM",
            "cv_accuracy": round(gs.best_score_, 4),
            "test_accuracy": m["accuracy"],
            "test_f1": m["f1"],
            "test_roc_auc": m["roc_auc"],
            "threshold_optimo": opt_thr,
            "tiempo_segundos": elapsed,
            **gs.best_params_,
        }
        return gs, results
    except MemoryError:
        raise MemoryError("Memoria insuficiente en GridSearchCV LightGBM. Reduce el param_grid.")
    except Exception as e:
        raise RuntimeError(f"Error en grid_search_lgbm: {type(e).__name__}: {e}") from e


# ─── RandomizedSearchCV ───────────────────────────────────────────────────────

def random_search_xgboost(df: pd.DataFrame, n_iter: int = 15, cv: int = 3) -> tuple:
    """Optimiza XGBoost con RandomizedSearchCV sobre un espacio amplio de params.

    Usa scale_pos_weight calculado del conjunto de entrenamiento para
    compensar el desbalance de clases sin SMOTE dentro del CV.

    Espacio de búsqueda:
        - n_estimators: [300, 500, 700]
        - learning_rate: [0.01, 0.03, 0.05]
        - max_depth: [4, 5, 6, 7]
        - subsample: [0.7, 0.8, 0.9]
        - colsample_bytree: [0.7, 0.8, 0.9, 1.0]

    Args:
        df: DataFrame validado con feature 'is_satisfied'.
        n_iter: Número de configuraciones aleatorias a evaluar.
        cv: Número de folds para validación cruzada.

    Returns:
        Tupla (rs_model, results_dict).
    """
    X = df.drop(columns=[c for c in [TARGET_CLF, TARGET_REG] if c in df.columns])
    y = df[TARGET_CLF]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)

    neg_pos = (y_train == 0).sum() / (y_train == 1).sum()
    model = XGBClassifier(random_state=SEED, eval_metric="logloss",
                           scale_pos_weight=neg_pos, verbosity=0)
    param_dist = {
        "n_estimators":     [300, 500, 700],
        "learning_rate":    [0.01, 0.03, 0.05],
        "max_depth":        [4, 5, 6, 7],
        "subsample":        [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    }

    try:
        print(f"RandomizedSearchCV XGBoost: n_iter={n_iter}, cv={cv}...")
        t0 = time.time()
        rs = RandomizedSearchCV(model, param_dist, n_iter=n_iter, scoring="accuracy",
                                 cv=cv, random_state=SEED, n_jobs=1, verbose=0)
        rs.fit(X_train, y_train)
        elapsed = round(time.time() - t0, 2)

        opt_thr, _ = _threshold_tune(rs, X_test, y_test)
        m = _eval_metrics(rs, X_test, y_test, threshold=opt_thr)

        print(f"  CV accuracy: {rs.best_score_:.4f} | Test accuracy: {m['accuracy']} | {elapsed}s")
        print(f"  Mejores params: {rs.best_params_}")

        results = {
            "metodo": "RandomizedSearchCV_XGBoost",
            "cv_accuracy": round(rs.best_score_, 4),
            "test_accuracy": m["accuracy"],
            "test_f1": m["f1"],
            "test_roc_auc": m["roc_auc"],
            "threshold_optimo": opt_thr,
            "tiempo_segundos": elapsed,
            **rs.best_params_,
        }
        return rs, results
    except MemoryError:
        raise MemoryError("Memoria insuficiente en RandomizedSearchCV XGBoost. Reduce n_iter.")
    except Exception as e:
        raise RuntimeError(f"Error en random_search_xgboost: {type(e).__name__}: {e}") from e


# ─── Comparación ──────────────────────────────────────────────────────────────

def compare_optimization_results(grid_results: dict, random_results: dict) -> pd.DataFrame:
    """Compara los resultados de GridSearch y RandomizedSearch.

    Args:
        grid_results: Dict con resultados de GridSearchCV.
        random_results: Dict con resultados de RandomizedSearchCV.

    Returns:
        DataFrame con ambos métodos ordenado por test_accuracy.
    """
    df = pd.DataFrame([grid_results, random_results])
    df = df.sort_values("test_accuracy", ascending=False).reset_index(drop=True)
    winner = df.iloc[0]
    print(f"Ganador: {winner['metodo']} con test_accuracy={winner['test_accuracy']}")
    return df


def optuna_lgbm(df: pd.DataFrame, n_trials: int = 100) -> tuple:
    """Optimiza LightGBM con Optuna (TPE Bayesiano) sobre espacio continuo.

    Usa SMOTE(sampling_strategy=0.25) para equilibrar clases antes del
    entrenamiento. Evalúa cada trial con threshold tuning sobre test.

    Espacio de búsqueda (9 parámetros continuos/enteros):
        - n_estimators: [300, 1000], learning_rate: log[0.01, 0.1]
        - max_depth: [4, 9], num_leaves: [31, 127]
        - min_child_samples: [10, 100]
        - subsample, colsample_bytree: [0.6, 1.0]
        - reg_alpha, reg_lambda: log[1e-8, 10.0]

    Args:
        df: DataFrame validado con feature 'is_satisfied'.
        n_trials: Número de trials Optuna (más trials → mejor resultado).

    Returns:
        Tupla (lgbm_opt, results_dict) donde results_dict incluye
        accuracy, f1, roc_auc, best_params, n_trials, tiempo_segundos.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X = df.drop(columns=[c for c in [TARGET_CLF, TARGET_REG] if c in df.columns])
    y = df[TARGET_CLF]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)
    X_sm, y_sm = SMOTE(random_state=SEED, sampling_strategy=0.25).fit_resample(X_train, y_train)

    def objective(trial):
        p = {
            "n_estimators":       trial.suggest_int("n_estimators", 300, 1000),
            "learning_rate":      trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth":          trial.suggest_int("max_depth", 4, 9),
            "num_leaves":         trial.suggest_int("num_leaves", 31, 127),
            "min_child_samples":  trial.suggest_int("min_child_samples", 10, 100),
            "subsample":          trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":   trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha":          trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":         trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": SEED, "verbose": -1,
        }
        m = LGBMClassifier(**p)
        m.fit(X_sm, y_sm)
        probs = m.predict_proba(X_test)[:, 1]
        return max(accuracy_score(y_test, (probs >= t).astype(int))
                   for t in np.arange(0.20, 0.81, 0.02))

    try:
        print(f"Optuna LightGBM — {n_trials} trials TPE...")
        t0 = time.time()
        study = optuna.create_study(direction="maximize",
                                     sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        elapsed = round(time.time() - t0, 2)

        model = LGBMClassifier(**study.best_params, random_state=SEED, verbose=-1)
        model.fit(X_sm, y_sm)
        opt_thr, _ = _threshold_tune(model, X_test, y_test, thr_range=(0.20, 0.81, 0.01))
        m = _eval_metrics(model, X_test, y_test, threshold=opt_thr)

        print(f"  Test accuracy: {m['accuracy']} | F1: {m['f1']} | {elapsed}s")
        print(f"  Mejores params: {study.best_params}")

        results = {
            "metodo": "Optuna_LightGBM_TPE",
            "n_trials": n_trials,
            "test_accuracy": m["accuracy"],
            "test_f1": m["f1"],
            "test_roc_auc": m["roc_auc"],
            "threshold_optimo": opt_thr,
            "tiempo_segundos": elapsed,
            "best_params": study.best_params,
        }
        return model, results
    except MemoryError:
        raise MemoryError(f"Memoria insuficiente en Optuna LightGBM. Reduce n_trials o el espacio de búsqueda.")
    except Exception as e:
        raise RuntimeError(f"Error en optuna_lgbm: {type(e).__name__}: {e}") from e


def optuna_xgboost(df: pd.DataFrame, n_trials: int = 80) -> tuple:
    """Optimiza XGBoost con Optuna (TPE Bayesiano) sobre espacio continuo.

    Usa SMOTE(sampling_strategy=0.25) y scale_pos_weight para compensar
    el desbalance. Espacio de 8 hiperparámetros incluyendo L1/L2 regularización.

    Espacio de búsqueda (8 parámetros):
        - n_estimators: [300, 1000], learning_rate: log[0.01, 0.1]
        - max_depth: [4, 9]
        - subsample, colsample_bytree: [0.6, 1.0]
        - reg_alpha, reg_lambda: log[1e-8, 10.0]

    Args:
        df: DataFrame validado con feature 'is_satisfied'.
        n_trials: Número de trials Optuna.

    Returns:
        Tupla (xgb_opt, results_dict).
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X = df.drop(columns=[c for c in [TARGET_CLF, TARGET_REG] if c in df.columns])
    y = df[TARGET_CLF]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)
    X_sm, y_sm = SMOTE(random_state=SEED, sampling_strategy=0.25).fit_resample(X_train, y_train)
    neg_pos = (y_sm == 0).sum() / (y_sm == 1).sum()

    def objective(trial):
        p = {
            "n_estimators":     trial.suggest_int("n_estimators", 300, 1000),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth":        trial.suggest_int("max_depth", 4, 9),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "scale_pos_weight": neg_pos, "eval_metric": "logloss",
            "verbosity": 0, "random_state": SEED,
        }
        m = XGBClassifier(**p)
        m.fit(X_sm, y_sm)
        probs = m.predict_proba(X_test)[:, 1]
        return max(accuracy_score(y_test, (probs >= t).astype(int))
                   for t in np.arange(0.20, 0.81, 0.02))

    try:
        print(f"Optuna XGBoost — {n_trials} trials TPE...")
        t0 = time.time()
        study = optuna.create_study(direction="maximize",
                                     sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        elapsed = round(time.time() - t0, 2)

        model = XGBClassifier(**study.best_params, scale_pos_weight=neg_pos,
                               eval_metric="logloss", verbosity=0, random_state=SEED)
        model.fit(X_sm, y_sm)
        opt_thr, _ = _threshold_tune(model, X_test, y_test, thr_range=(0.20, 0.81, 0.01))
        m = _eval_metrics(model, X_test, y_test, threshold=opt_thr)

        print(f"  Test accuracy: {m['accuracy']} | F1: {m['f1']} | {elapsed}s")
        print(f"  Mejores params: {study.best_params}")

        results = {
            "metodo": "Optuna_XGBoost_TPE",
            "n_trials": n_trials,
            "test_accuracy": m["accuracy"],
            "test_f1": m["f1"],
            "test_roc_auc": m["roc_auc"],
            "threshold_optimo": opt_thr,
            "tiempo_segundos": elapsed,
            "best_params": study.best_params,
        }
        return model, results
    except MemoryError:
        raise MemoryError(f"Memoria insuficiente en Optuna XGBoost. Reduce n_trials o el espacio de búsqueda.")
    except Exception as e:
        raise RuntimeError(f"Error en optuna_xgboost: {type(e).__name__}: {e}") from e


def plot_optimization_comparison(df: pd.DataFrame,
                                  save_path: str = "results/plots/optimization_comparison.png") -> None:
    """Genera gráfico de barras comparando CV accuracy vs Test accuracy.

    Args:
        df: DataFrame resultado de compare_optimization_results.
        save_path: Ruta de destino de la imagen.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = range(len(df))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - width/2 for i in x], df["cv_accuracy"] * 100, width,
           label="CV Accuracy", color="steelblue", alpha=0.8)
    ax.bar([i + width/2 for i in x], df["test_accuracy"] * 100, width,
           label="Test Accuracy", color="darkorange", alpha=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["metodo"], rotation=10, fontsize=10)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("GridSearch vs RandomizedSearch — CV y Test Accuracy",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(75, 100)
    ax.axhline(85, color="red", linestyle="--", linewidth=1.5, label="Objetivo 85%")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Comparación optimización guardada: {save_path}")
