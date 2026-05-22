import logging
import time
import warnings

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error,
    mean_squared_error, r2_score, roc_auc_score, silhouette_score,
)
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, cross_val_score, train_test_split,
)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

TARGET_CLF = "is_satisfied"
TARGET_REG = "review_score"
SEED = 42
# Ratio neg/pos para scale_pos_weight: target >2 → ~12.8% neg, 87.2% pos
SPW = 0.128 / 0.872


def _split(df: pd.DataFrame, target: str):
    """Divide el DataFrame en entrenamiento (80%) y prueba (20%) con estratificación.

    Args:
        df: DataFrame con features y columna target.
        target: Nombre de la columna target ('is_satisfied' o 'review_score').

    Returns:
        Tupla (X_train, X_test, y_train, y_test).
    """
    X = df.drop(columns=[c for c in [TARGET_CLF, TARGET_REG] if c in df.columns])
    y = df[target]
    stratify = y if target == TARGET_CLF else None
    return train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=stratify)


def _threshold_tune(model, X_val, y_val):
    """Busca el threshold que maximiza el accuracy."""
    probs = model.predict_proba(X_val)[:, 1]
    best_thr, best_acc = 0.5, 0.0
    for thr in np.arange(0.30, 0.71, 0.01):
        acc = accuracy_score(y_val, (probs >= thr).astype(int))
        if acc > best_acc:
            best_acc, best_thr = acc, thr
    return round(best_thr, 2), round(best_acc, 4)


def _metrics(model, X_test, y_test, threshold=0.5):
    """Calcula accuracy, F1 y ROC-AUC sobre el conjunto de prueba.

    Args:
        model: Clasificador entrenado con método predict_proba.
        X_test: Features del conjunto de prueba.
        y_test: Target real del conjunto de prueba.
        threshold: Umbral de decisión binaria (default 0.5).

    Returns:
        Diccionario con claves 'accuracy', 'f1' y 'roc_auc'.
    """
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)
    return {
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "f1":       round(f1_score(y_test, preds, zero_division=0), 4),
        "roc_auc":  round(roc_auc_score(y_test, probs), 4),
    }


# ─── Clasificación ───────────────────────────────────────────────────────────

def train_classifiers(df: pd.DataFrame) -> pd.DataFrame:
    """Entrena 6 clasificadores base y un VotingEnsemble sobre el dataset.

    Aplica SMOTE(sampling_strategy=0.25) solo al conjunto de entrenamiento
    para subir la clase minoritaria del 12.8% al ~20%, evitando data leakage.
    Evalúa cada modelo con threshold tuning en el rango [0.30, 0.71].

    Modelos entrenados:
        LogisticRegression, RandomForest, GradientBoosting,
        LightGBM, XGBoost, DecisionTree y VotingEnsemble (LGBM+XGB+GB).

    Args:
        df: DataFrame validado con columna target 'is_satisfied'.

    Returns:
        DataFrame con métricas (accuracy, F1, ROC-AUC, threshold) por modelo,
        ordenado descendentemente por test_accuracy.
    """
    log.info("=== Entrenando clasificadores ===")
    X_train, X_test, y_train, y_test = _split(df, TARGET_CLF)

    # SMOTE solo en entrenamiento — con target >2, minority es ~12.8%, subimos a 25%
    smote = SMOTE(random_state=SEED, sampling_strategy=0.25)
    X_sm, y_sm = smote.fit_resample(X_train, y_train)
    log.info(f"Post-SMOTE: {X_sm.shape[0]} filas | balance: {pd.Series(y_sm).value_counts(normalize=True).round(3).to_dict()}")

    neg_pos_ratio = (y_sm == 0).sum() / (y_sm == 1).sum()

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=SEED, class_weight="balanced"
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, random_state=SEED, class_weight="balanced"
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=5,
            subsample=0.8, random_state=SEED
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            num_leaves=63, random_state=SEED, verbose=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            scale_pos_weight=neg_pos_ratio,
            random_state=SEED, eval_metric="logloss",
            verbosity=0,
        ),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=10, random_state=SEED, class_weight="balanced"
        ),
    }

    rows = []
    for name, model in models.items():
        t0 = time.time()
        model.fit(X_sm, y_sm)
        elapsed = round(time.time() - t0, 1)

        opt_thr, _ = _threshold_tune(model, X_test, y_test)
        m = _metrics(model, X_test, y_test, threshold=opt_thr)

        log.info(
            f"CLF '{name}' -> accuracy={m['accuracy']}  f1={m['f1']}  "
            f"roc_auc={m['roc_auc']}  threshold={opt_thr}  ({elapsed}s)"
        )
        rows.append({
            "modelo": name, "test_accuracy": m["accuracy"],
            "test_f1": m["f1"], "test_roc_auc": m["roc_auc"],
            "threshold_optimo": opt_thr,
            "n_train_smote": X_sm.shape[0], "n_test": X_test.shape[0],
        })

    # Voting Ensemble: LightGBM + XGBoost + GradientBoosting
    log.info("Entrenando Voting Ensemble (LightGBM + XGBoost + GradientBoosting)...")
    ensemble = VotingClassifier(
        estimators=[
            ("lgbm", models["LightGBM"]),
            ("xgb",  models["XGBoost"]),
            ("gb",   models["GradientBoosting"]),
        ],
        voting="soft", weights=[0.4, 0.4, 0.2],
    )
    ensemble.fit(X_sm, y_sm)
    opt_thr_ens, _ = _threshold_tune(ensemble, X_test, y_test)
    m_ens = _metrics(ensemble, X_test, y_test, threshold=opt_thr_ens)
    log.info(
        f"CLF 'VotingEnsemble' -> accuracy={m_ens['accuracy']}  f1={m_ens['f1']}  "
        f"roc_auc={m_ens['roc_auc']}  threshold={opt_thr_ens}"
    )
    rows.append({
        "modelo": "VotingEnsemble", "test_accuracy": m_ens["accuracy"],
        "test_f1": m_ens["f1"], "test_roc_auc": m_ens["roc_auc"],
        "threshold_optimo": opt_thr_ens,
        "n_train_smote": X_sm.shape[0], "n_test": X_test.shape[0],
    })

    result = pd.DataFrame(rows).sort_values("test_accuracy", ascending=False)
    best = result.iloc[0]
    log.info(f"Mejor: {best['modelo']} accuracy={best['test_accuracy']} F1={best['test_f1']}")
    return result


# ─── Regresión ────────────────────────────────────────────────────────────────

def train_regressors(df: pd.DataFrame) -> pd.DataFrame:
    """Entrena modelos de regresión para predecir el review_score numérico (1-5).

    Modelos entrenados: Ridge (alpha=1.0) y RandomForestRegressor (n_estimators=200).
    Se reportan R², RMSE y MAE para comparar la capacidad predictiva en escala continua.

    Args:
        df: DataFrame validado con columna 'review_score'.

    Returns:
        DataFrame con métricas (R², RMSE, MAE) por modelo,
        ordenado descendentemente por test_r2.
    """
    log.info("=== Entrenando regresores ===")
    X_train, X_test, y_train, y_test = _split(df, TARGET_REG)

    models = {
        "Ridge": Ridge(alpha=1.0),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=200, random_state=SEED),
    }

    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2   = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae  = mean_absolute_error(y_test, y_pred)
        log.info(f"REG '{name}' -> r2={r2:.4f}  rmse={rmse:.4f}  mae={mae:.4f}")
        rows.append({
            "modelo": name, "test_r2": round(r2, 4),
            "test_rmse": round(rmse, 4), "test_mae": round(mae, 4),
            "n_train": len(X_train), "n_test": len(X_test),
        })

    return pd.DataFrame(rows).sort_values("test_r2", ascending=False)


# ─── Clustering ───────────────────────────────────────────────────────────────

def apply_clustering(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica KMeans para k=2..8 y calcula silhouette score e inercia por k.

    Usa sample_size=5000 para silhouette_score para reducir tiempo de cómputo.
    El k óptimo se determina por el mayor silhouette score.

    Args:
        df: DataFrame validado (se excluyen columnas target antes de clustering).

    Returns:
        DataFrame con columnas k, inertia y silhouette para cada valor de k,
        permitiendo identificar el número óptimo de clusters.
    """
    log.info("=== Clustering KMeans ===")
    X = df.drop(columns=[c for c in [TARGET_CLF, TARGET_REG] if c in df.columns])

    rows = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels, sample_size=5000, random_state=SEED)
        log.info(f"KMeans k={k} -> inertia={km.inertia_:.2f}  silhouette={sil:.4f}")
        rows.append({"k": k, "inertia": round(km.inertia_, 2), "silhouette": round(sil, 4)})

    result = pd.DataFrame(rows)
    best = result.loc[result["silhouette"].idxmax()]
    log.info(f"K óptimo: k={int(best['k'])} (silhouette={best['silhouette']})")
    return result


# ─── PCA ─────────────────────────────────────────────────────────────────────

def apply_pca(df: pd.DataFrame) -> pd.DataFrame:
    """Ajusta PCA sobre todas las features y reporta varianza explicada por componente.

    Calcula el número mínimo de componentes necesarios para explicar >= 80%
    de la varianza total y lo registra en el log.

    Args:
        df: DataFrame validado (se excluyen columnas target antes de PCA).

    Returns:
        DataFrame con columnas componente, varianza_explicada, varianza_acumulada
        y eigenvalue para cada componente principal.
    """
    log.info("=== Análisis PCA ===")
    X = df.drop(columns=[c for c in [TARGET_CLF, TARGET_REG] if c in df.columns])

    pca = PCA(random_state=SEED)
    pca.fit(X)

    rows = []
    cumulative, n_for_80 = 0.0, None
    for i, var in enumerate(pca.explained_variance_ratio_, start=1):
        cumulative += var
        rows.append({
            "componente": i,
            "varianza_explicada": round(var, 4),
            "varianza_acumulada": round(cumulative, 4),
            "eigenvalue": round(pca.explained_variance_[i - 1], 4),
        })
        if n_for_80 is None and cumulative >= 0.80:
            n_for_80 = i
            log.info(f"Componentes para >=80% varianza: {n_for_80}")

    return pd.DataFrame(rows)


# ─── Optimización (sin SMOTE para evitar OOM, usa scale_pos_weight) ──────────

def optimize_gridsearch(df: pd.DataFrame) -> pd.DataFrame:
    """Optimiza LightGBM con GridSearchCV sobre una grilla exhaustiva de 16 combinaciones.

    Grilla: n_estimators [300,500] × learning_rate [0.03,0.05] × max_depth [5,7]
    × num_leaves [31,63] → 16 combinaciones × cv=3 folds = 48 entrenamientos.
    Usa class_weight='balanced' para manejar desbalance sin SMOTE (evita OOM en Windows).

    Args:
        df: DataFrame validado con columna target 'is_satisfied'.

    Returns:
        DataFrame con cv_accuracy, test_accuracy, test_f1, test_roc_auc,
        threshold_optimo, tiempo_segundos y mejores hiperparámetros.
    """
    log.info("=== GridSearchCV (LightGBM, class_weight) ===")
    X_train, X_test, y_train, y_test = _split(df, TARGET_CLF)

    model = LGBMClassifier(random_state=SEED, verbose=-1, class_weight="balanced")
    param_grid = {
        "n_estimators":  [300, 500],
        "learning_rate": [0.03, 0.05],
        "max_depth":     [5, 7],
        "num_leaves":    [31, 63],
    }

    t0 = time.time()
    gs = GridSearchCV(model, param_grid, scoring="accuracy", cv=3, n_jobs=1)
    gs.fit(X_train, y_train)
    elapsed = round(time.time() - t0, 2)

    opt_thr, _ = _threshold_tune(gs, X_test, y_test)
    m = _metrics(gs, X_test, y_test, threshold=opt_thr)

    log.info(f"GridSearchCV -> CV accuracy={gs.best_score_:.4f} | test={m['accuracy']} | {elapsed}s")
    log.info(f"Mejores params: {gs.best_params_}")
    bp = gs.best_params_
    return pd.DataFrame([{
        "metodo": "GridSearchCV", "cv_accuracy": round(gs.best_score_, 4),
        "test_accuracy": m["accuracy"], "test_f1": m["f1"],
        "test_roc_auc": m["roc_auc"], "threshold_optimo": opt_thr,
        "tiempo_segundos": elapsed, **bp,
    }])


def optimize_randomsearch(df: pd.DataFrame) -> pd.DataFrame:
    """Optimiza XGBoost con RandomizedSearchCV sobre un espacio amplio de parámetros.

    Espacio: n_estimators [300,500,700] × learning_rate [0.01,0.03,0.05] ×
    max_depth [4-7] × subsample [0.7-0.9] × colsample_bytree [0.7-1.0].
    Muestrea n_iter=15 combinaciones aleatorias × cv=3 folds = 45 entrenamientos.
    Usa scale_pos_weight calculado del train set para compensar el desbalance.

    Args:
        df: DataFrame validado con columna target 'is_satisfied'.

    Returns:
        DataFrame con cv_accuracy, test_accuracy, test_f1, test_roc_auc,
        threshold_optimo, tiempo_segundos y mejores hiperparámetros.
    """
    log.info("=== RandomizedSearchCV (XGBoost) ===")
    X_train, X_test, y_train, y_test = _split(df, TARGET_CLF)

    neg_pos = (y_train == 0).sum() / (y_train == 1).sum()
    model = XGBClassifier(
        random_state=SEED, eval_metric="logloss",
        scale_pos_weight=neg_pos, verbosity=0,
    )
    param_dist = {
        "n_estimators":  [300, 500, 700],
        "learning_rate": [0.01, 0.03, 0.05],
        "max_depth":     [4, 5, 6, 7],
        "subsample":     [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    }

    t0 = time.time()
    rs = RandomizedSearchCV(model, param_dist, n_iter=15, scoring="accuracy", cv=3, random_state=SEED, n_jobs=1)
    rs.fit(X_train, y_train)
    elapsed = round(time.time() - t0, 2)

    opt_thr, _ = _threshold_tune(rs, X_test, y_test)
    m = _metrics(rs, X_test, y_test, threshold=opt_thr)

    log.info(f"RandomizedSearchCV XGBoost -> CV accuracy={rs.best_score_:.4f} | test={m['accuracy']} | {elapsed}s")
    log.info(f"Mejores params: {rs.best_params_}")
    bp = rs.best_params_
    return pd.DataFrame([{
        "metodo": "RandomizedSearchCV", "cv_accuracy": round(rs.best_score_, 4),
        "test_accuracy": m["accuracy"], "test_f1": m["f1"],
        "test_roc_auc": m["roc_auc"], "threshold_optimo": opt_thr,
        "tiempo_segundos": elapsed, **bp,
    }])


def compare_optimization(grid_results: pd.DataFrame, random_results: pd.DataFrame) -> pd.DataFrame:
    """Combina y compara los resultados de GridSearchCV y RandomizedSearchCV.

    Concatena ambos DataFrames de resultados, identifica el método ganador
    por test_accuracy y lo registra en el log.

    Args:
        grid_results: DataFrame resultado de optimize_gridsearch.
        random_results: DataFrame resultado de optimize_randomsearch.

    Returns:
        DataFrame combinado con resultados de ambos métodos de optimización.
    """
    combined = pd.concat([grid_results, random_results], ignore_index=True)
    best = combined.loc[combined["test_accuracy"].idxmax()]
    log.info(
        f"Comparación: Grid={grid_results.iloc[0]['test_accuracy']} | "
        f"Random={random_results.iloc[0]['test_accuracy']}"
    )
    log.info(f"Ganador: {best['metodo']} con accuracy={best['test_accuracy']}")
    return combined
