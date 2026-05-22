"""
model_training.py
-----------------
Definición y entrenamiento de modelos supervisados y no supervisados para
predecir la satisfacción del cliente en el dataset Brazilian E-Commerce (Olist).

Modelos supervisados:
    - Clasificación: LogisticRegression, RandomForest, GradientBoosting,
      LightGBM, XGBoost, DecisionTree, VotingEnsemble.
    - Regresión: Ridge, RandomForestRegressor.

Modelos no supervisados:
    - KMeans (k=2..8), PCA.

Técnicas de balanceo:
    - SMOTE (Synthetic Minority Over-sampling Technique) en train set.
    - class_weight='balanced' para modelos que lo soportan.
    - scale_pos_weight en XGBoost/LightGBM.
"""

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import (GradientBoostingClassifier, RandomForestClassifier,
                               RandomForestRegressor, VotingClassifier)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

SEED = 42
TARGET_CLF = "is_satisfied"
TARGET_REG = "review_score"


def split_data(df: pd.DataFrame, target: str, test_size: float = 0.2):
    """Divide el DataFrame en conjuntos de entrenamiento y prueba.

    Args:
        df: DataFrame con features y target.
        target: Nombre de la columna target.
        test_size: Proporción del conjunto de prueba (default 0.2).

    Returns:
        Tupla (X_train, X_test, y_train, y_test).
    """
    X = df.drop(columns=[c for c in [TARGET_CLF, TARGET_REG] if c in df.columns])
    y = df[target]
    stratify = y if target == TARGET_CLF else None
    return train_test_split(X, y, test_size=test_size, random_state=SEED, stratify=stratify)


def apply_smote(X_train: pd.DataFrame, y_train: pd.Series,
                sampling_strategy: float = 0.25) -> tuple:
    """Aplica SMOTE al conjunto de entrenamiento para balancear clases.

    Solo se aplica sobre el conjunto de entrenamiento para evitar data leakage.
    Con el target actual (review_score > 2), la clase minoritaria es ~12.8%
    y se sube a ~20% (sampling_strategy=0.25).

    Args:
        X_train: Features de entrenamiento.
        y_train: Target de entrenamiento.
        sampling_strategy: Ratio minoritaria/mayoritaria deseado post-SMOTE.

    Returns:
        Tupla (X_resampled, y_resampled) con datos balanceados.
    """
    smote = SMOTE(random_state=SEED, sampling_strategy=sampling_strategy)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    balance = pd.Series(y_res).value_counts(normalize=True).round(3).to_dict()
    print(f"Post-SMOTE: {X_res.shape[0]} filas | balance: {balance}")
    return X_res, y_res


def train_classifiers(df: pd.DataFrame) -> dict:
    """Entrena múltiples clasificadores y un ensemble sobre el dataset.

    Utiliza SMOTE en el conjunto de entrenamiento para manejar el desbalance
    de clases (87.2% satisfechos vs 12.8% insatisfechos).

    Args:
        df: DataFrame validado con features y columna 'is_satisfied'.

    Returns:
        Diccionario {nombre_modelo: modelo_entrenado}.
    """
    X_train, X_test, y_train, y_test = split_data(df, TARGET_CLF)
    X_sm, y_sm = apply_smote(X_train, y_train)
    neg_pos = (y_sm == 0).sum() / (y_sm == 1).sum()

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=SEED, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, random_state=SEED, class_weight="balanced"),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=5,
            subsample=0.8, random_state=SEED),
        "LightGBM": LGBMClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            num_leaves=63, random_state=SEED, verbose=-1),
        "XGBoost": XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            scale_pos_weight=neg_pos, random_state=SEED,
            eval_metric="logloss", verbosity=0),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=10, random_state=SEED, class_weight="balanced"),
    }

    trained = {}
    for name, model in models.items():
        try:
            print(f"Entrenando {name}...")
            model.fit(X_sm, y_sm)
            trained[name] = model
        except MemoryError:
            raise MemoryError(f"Memoria insuficiente al entrenar {name}. Reduce n_estimators.")
        except Exception as e:
            raise RuntimeError(f"Error al entrenar {name}: {type(e).__name__}: {e}") from e

    # VotingEnsemble: combina los 3 mejores clasificadores
    try:
        ensemble = VotingClassifier(
            estimators=[("lgbm", trained["LightGBM"]),
                        ("xgb",  trained["XGBoost"]),
                        ("gb",   trained["GradientBoosting"])],
            voting="soft", weights=[0.4, 0.4, 0.2])
        ensemble.fit(X_sm, y_sm)
        trained["VotingEnsemble"] = ensemble
    except Exception as e:
        raise RuntimeError(f"Error al entrenar VotingEnsemble: {type(e).__name__}: {e}") from e

    print(f"Entrenamiento completado: {len(trained)} modelos.")
    return trained, X_test, y_test


def train_regressors(df: pd.DataFrame) -> dict:
    """Entrena modelos de regresión para predecir el review_score numérico.

    Args:
        df: DataFrame validado con columna 'review_score'.

    Returns:
        Diccionario {nombre_modelo: modelo_entrenado}.
    """
    X_train, X_test, y_train, y_test = split_data(df, TARGET_REG)

    models = {
        "Ridge": Ridge(alpha=1.0),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=200, random_state=SEED),
    }
    trained = {}
    for name, model in models.items():
        try:
            print(f"Entrenando regressor {name}...")
            model.fit(X_train, y_train)
            trained[name] = model
        except Exception as e:
            raise RuntimeError(f"Error al entrenar regresor {name}: {type(e).__name__}: {e}") from e

    return trained, X_test, y_test


def train_kmeans(df: pd.DataFrame, k_range: range = range(2, 9)) -> dict:
    """Aplica KMeans para distintos valores de k y calcula silhouette score.

    Args:
        df: DataFrame sin columnas target.
        k_range: Rango de valores de k a probar.

    Returns:
        Diccionario {k: {"model": km, "inertia": ..., "silhouette": ...}}.
    """
    X = df.drop(columns=[c for c in [TARGET_CLF, TARGET_REG] if c in df.columns])
    results = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels, sample_size=5000, random_state=SEED)
        results[k] = {"model": km, "inertia": km.inertia_, "silhouette": sil}
        print(f"KMeans k={k} -> silhouette={sil:.4f}  inertia={km.inertia_:.2f}")
    return results, X


def apply_pca(df: pd.DataFrame, n_components: int = None) -> tuple:
    """Ajusta PCA y retorna el modelo y la varianza explicada acumulada.

    Args:
        df: DataFrame sin columnas target.
        n_components: Número de componentes. None = todos.

    Returns:
        Tupla (pca_model, X_transformed, explained_variance_ratio).
    """
    X = df.drop(columns=[c for c in [TARGET_CLF, TARGET_REG] if c in df.columns])
    pca = PCA(n_components=n_components, random_state=SEED)
    X_pca = pca.fit_transform(X)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n80 = next((i + 1 for i, v in enumerate(cumvar) if v >= 0.80), len(cumvar))
    print(f"PCA: {n80} componentes explican >= 80% de la varianza.")
    return pca, X_pca, pca.explained_variance_ratio_


def save_model(model, name: str, output_dir: str = "models/trained_models") -> str:
    """Serializa un modelo entrenado en formato joblib.

    Args:
        model: Modelo scikit-learn / compatible entrenado.
        name: Nombre del archivo (sin extensión).
        output_dir: Carpeta de destino.

    Returns:
        Ruta completa del archivo guardado.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    path = f"{output_dir}/{name}.pkl"
    joblib.dump(model, path)
    print(f"Modelo guardado: {path}")
    return path


def load_model(name: str, model_dir: str = "models/trained_models"):
    """Carga un modelo serializado desde disco.

    Args:
        name: Nombre del archivo (sin extensión).
        model_dir: Carpeta donde está el modelo.

    Returns:
        Modelo deserializado.

    Raises:
        FileNotFoundError: Si el archivo no existe.
    """
    path = f"{model_dir}/{name}.pkl"
    return joblib.load(path)
