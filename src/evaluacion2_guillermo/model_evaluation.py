"""
model_evaluation.py
-------------------
Funciones de evaluación y comparación de modelos supervisados y no supervisados.

Incluye:
    - Métricas de clasificación: accuracy, F1, ROC-AUC con threshold tuning.
    - Métricas de regresión: R², RMSE, MAE.
    - Validación cruzada estratificada.
    - Generación de visualizaciones: curvas ROC, matrices de confusión,
      importancia de features, comparación de modelos.
"""

import matplotlib
matplotlib.use("Agg")  # Backend sin pantalla para guardar figuras

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, f1_score,
                              mean_absolute_error, mean_squared_error, r2_score,
                              roc_auc_score, roc_curve)
from sklearn.model_selection import cross_val_score, StratifiedKFold

PALETTE = "viridis"


# ─── Threshold Tuning ─────────────────────────────────────────────────────────

def find_best_threshold(model, X_val: pd.DataFrame, y_val: pd.Series,
                         thr_range: tuple = (0.25, 0.76, 0.01)) -> tuple:
    """Busca el threshold de clasificación que maximiza el accuracy.

    Args:
        model: Modelo con método predict_proba.
        X_val: Features de validación.
        y_val: Target real de validación.
        thr_range: Tupla (inicio, fin, paso) para np.arange.

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


# ─── Métricas de Clasificación ────────────────────────────────────────────────

def evaluate_classifier(model, X_test: pd.DataFrame, y_test: pd.Series,
                         threshold: float = 0.5) -> dict:
    """Calcula métricas de clasificación para un modelo dado.

    Args:
        model: Clasificador entrenado con predict_proba.
        X_test: Features de prueba.
        y_test: Target real de prueba.
        threshold: Umbral de decisión para convertir probabilidades en clases.

    Returns:
        Diccionario con accuracy, f1, roc_auc, precision, recall.

    Raises:
        ValueError: Si el modelo no tiene método predict_proba.
    """
    try:
        from sklearn.metrics import precision_score, recall_score
        if not hasattr(model, "predict_proba"):
            raise ValueError(f"El modelo {type(model).__name__} no tiene predict_proba.")
        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs >= threshold).astype(int)
        return {
            "accuracy":  round(accuracy_score(y_test, preds), 4),
            "f1":        round(f1_score(y_test, preds, zero_division=0), 4),
            "roc_auc":   round(roc_auc_score(y_test, probs), 4),
            "precision": round(precision_score(y_test, preds, zero_division=0), 4),
            "recall":    round(recall_score(y_test, preds, zero_division=0), 4),
        }
    except (ValueError, AttributeError):
        raise
    except Exception as e:
        raise RuntimeError(f"Error al evaluar clasificador: {type(e).__name__}: {e}") from e


def cross_validate_model(model, X: pd.DataFrame, y: pd.Series,
                          cv: int = 5, scoring: str = "accuracy") -> dict:
    """Realiza validación cruzada estratificada.

    Args:
        model: Modelo scikit-learn compatible.
        X: Features completas.
        y: Target completo.
        cv: Número de folds (default 5).
        scoring: Métrica a usar (default 'accuracy').

    Returns:
        Diccionario con mean y std de la métrica.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring=scoring)
    return {
        f"cv_{scoring}_mean": round(scores.mean(), 4),
        f"cv_{scoring}_std":  round(scores.std(), 4),
    }


# ─── Métricas de Regresión ────────────────────────────────────────────────────

def evaluate_regressor(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Calcula métricas de regresión para un modelo dado.

    Args:
        model: Regresor entrenado.
        X_test: Features de prueba.
        y_test: Target real de prueba.

    Returns:
        Diccionario con r2, rmse, mae.
    """
    y_pred = model.predict(X_test)
    return {
        "r2":   round(r2_score(y_test, y_pred), 4),
        "rmse": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
        "mae":  round(mean_absolute_error(y_test, y_pred), 4),
    }


# ─── Comparación de Modelos ───────────────────────────────────────────────────

def compare_classifiers(results: list[dict]) -> pd.DataFrame:
    """Construye una tabla comparativa ordenada por accuracy descendente.

    Args:
        results: Lista de dicts con keys 'modelo' y métricas.

    Returns:
        DataFrame ordenado por test_accuracy.
    """
    df = pd.DataFrame(results)
    return df.sort_values("accuracy", ascending=False).reset_index(drop=True)


# ─── Visualizaciones ──────────────────────────────────────────────────────────

def plot_roc_curves(models: dict, X_test: pd.DataFrame, y_test: pd.Series,
                    save_path: str = "results/plots/roc_curves.png") -> None:
    """Genera curvas ROC para múltiples modelos en una sola figura.

    Args:
        models: Diccionario {nombre: modelo_entrenado}.
        X_test: Features de prueba.
        y_test: Target real de prueba.
        save_path: Ruta donde guardar la imagen.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    for (name, model), color in zip(models.items(), colors):
        probs = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})", color=color, lw=1.8)

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Baseline (AUC=0.5)")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Curvas ROC — Comparación de Clasificadores", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"ROC curves guardadas: {save_path}")


def plot_confusion_matrix(model, X_test: pd.DataFrame, y_test: pd.Series,
                           threshold: float, name: str,
                           save_path: str = "results/plots/confusion_matrix.png") -> None:
    """Genera y guarda la matriz de confusión de un clasificador.

    Args:
        model: Clasificador entrenado.
        X_test: Features de prueba.
        y_test: Target real.
        threshold: Umbral de decisión.
        name: Nombre del modelo (para el título).
        save_path: Ruta de destino de la imagen.
    """
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(y_test, preds)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Insatisfecho", "Satisfecho"],
                yticklabels=["Insatisfecho", "Satisfecho"])
    ax.set_ylabel("Real", fontsize=11)
    ax.set_xlabel("Predicho", fontsize=11)
    ax.set_title(f"Matriz de Confusión — {name}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix guardada: {save_path}")


def plot_model_comparison(df_metrics: pd.DataFrame,
                           save_path: str = "results/plots/model_comparison.png") -> None:
    """Genera gráfico de barras comparando accuracy de todos los modelos.

    Args:
        df_metrics: DataFrame con columnas 'modelo' y 'accuracy'.
        save_path: Ruta de destino.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(df_metrics["modelo"], df_metrics["accuracy"] * 100,
                   color=plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(df_metrics))))
    ax.axvline(85, color="red", linestyle="--", linewidth=1.5, label="Objetivo 85%")
    ax.set_xlabel("Accuracy (%)", fontsize=12)
    ax.set_title("Comparación de Clasificadores — Accuracy en Test", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    for bar, val in zip(bars, df_metrics["accuracy"]):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                f"{val*100:.2f}%", va="center", fontsize=9)
    ax.set_xlim(50, 100)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Comparación guardada: {save_path}")


def plot_feature_importance(model, feature_names: list, top_n: int = 15,
                             save_path: str = "results/plots/feature_importance.png") -> None:
    """Genera gráfico de importancia de features para modelos basados en árboles.

    Args:
        model: Modelo con atributo feature_importances_.
        feature_names: Lista de nombres de features.
        top_n: Número de features a mostrar.
        save_path: Ruta de destino.
    """
    importances = model.feature_importances_
    df_imp = pd.DataFrame({"feature": feature_names, "importance": importances})
    df_imp = df_imp.nlargest(top_n, "importance")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=df_imp, x="importance", y="feature", palette="Blues_r", ax=ax)
    ax.set_title(f"Top {top_n} Features más Importantes", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importancia", fontsize=11)
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Feature importance guardada: {save_path}")


def plot_silhouette(kmeans_results: dict,
                    save_path: str = "results/plots/silhouette_kmeans.png") -> None:
    """Grafica el silhouette score vs k para resultados de KMeans.

    Args:
        kmeans_results: Dict {k: {"silhouette": ..., "inertia": ...}}.
        save_path: Ruta de destino.
    """
    ks = list(kmeans_results.keys())
    sils = [kmeans_results[k]["silhouette"] for k in ks]
    inertias = [kmeans_results[k]["inertia"] for k in ks]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(ks, sils, "bo-", linewidth=2, markersize=7)
    ax1.set_xlabel("Número de clusters (k)", fontsize=11)
    ax1.set_ylabel("Silhouette Score", fontsize=11)
    ax1.set_title("Silhouette Score por k", fontsize=12, fontweight="bold")
    ax1.grid(alpha=0.3)
    best_k = max(kmeans_results, key=lambda k: kmeans_results[k]["silhouette"])
    ax1.axvline(best_k, color="red", linestyle="--", label=f"k óptimo={best_k}")
    ax1.legend()

    ax2.plot(ks, inertias, "rs-", linewidth=2, markersize=7)
    ax2.set_xlabel("Número de clusters (k)", fontsize=11)
    ax2.set_ylabel("Inercia", fontsize=11)
    ax2.set_title("Elbow Method — Inercia por k", fontsize=12, fontweight="bold")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Silhouette/Elbow guardado: {save_path}")


def plot_pca_variance(explained_variance_ratio: np.ndarray,
                       save_path: str = "results/plots/pca_variance.png") -> None:
    """Grafica la varianza explicada y acumulada por componente PCA.

    Args:
        explained_variance_ratio: Array de varianza por componente.
        save_path: Ruta de destino.
    """
    cumulative = np.cumsum(explained_variance_ratio)
    n_comp = len(explained_variance_ratio)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(1, n_comp + 1), explained_variance_ratio * 100,
           alpha=0.6, color="steelblue", label="Varianza por componente")
    ax.plot(range(1, n_comp + 1), cumulative * 100, "ro-",
            linewidth=2, markersize=4, label="Varianza acumulada")
    ax.axhline(80, color="green", linestyle="--", linewidth=1.5, label="80% umbral")
    ax.set_xlabel("Componente Principal", fontsize=11)
    ax.set_ylabel("Varianza Explicada (%)", fontsize=11)
    ax.set_title("PCA — Varianza Explicada por Componente", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"PCA variance guardada: {save_path}")
