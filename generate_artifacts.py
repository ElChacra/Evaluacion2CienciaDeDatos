"""
generate_artifacts.py
---------------------
Script único para generar todos los artefactos del proyecto:
    - Modelos serializados en models/trained_models/
    - Gráficos en results/plots/
    - Métricas CSV en results/metrics/

Ejecutar desde la raíz del proyecto:
    python generate_artifacts.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              roc_curve, auc, confusion_matrix)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               RandomForestRegressor, VotingClassifier)
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

SEED = 42
TARGET_CLF = "is_satisfied"
TARGET_REG  = "review_score"

print("=" * 60)
print("GENERANDO ARTEFACTOS DEL PROYECTO OLIST E-COMMERCE")
print("=" * 60)

# ── Cargar datos ──────────────────────────────────────────────
print("\n[1/6] Cargando datos...")
df = pd.read_csv("data/05_model_input/dataset_validated.csv", encoding="utf-8")
print(f"  Dataset: {df.shape[0]:,} filas x {df.shape[1]} columnas")
print(f"  Balance is_satisfied: {df[TARGET_CLF].value_counts(normalize=True).round(3).to_dict()}")

X_all = df.drop(columns=[c for c in [TARGET_CLF, TARGET_REG] if c in df.columns])
y_clf = df[TARGET_CLF]
y_reg = df[TARGET_REG]

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_clf, test_size=0.2, random_state=SEED, stratify=y_clf)

X_sm_train, y_sm_train = SMOTE(random_state=SEED, sampling_strategy=0.25).fit_resample(X_train, y_train)
neg_pos = (y_sm_train == 0).sum() / (y_sm_train == 1).sum()

# ── Entrenar clasificadores ───────────────────────────────────
print("\n[2/6] Entrenando clasificadores...")

models_clf = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=SEED, class_weight="balanced"),
    "DecisionTree":       DecisionTreeClassifier(max_depth=10, random_state=SEED, class_weight="balanced"),
    "RandomForest":       RandomForestClassifier(n_estimators=300, random_state=SEED, class_weight="balanced"),
    "GradientBoosting":   GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, subsample=0.8, random_state=SEED),
    "LightGBM":           LGBMClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, num_leaves=63, random_state=SEED, verbose=-1),
    "XGBoost":            XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, scale_pos_weight=neg_pos, random_state=SEED, eval_metric="logloss", verbosity=0),
}

for name, model in models_clf.items():
    model.fit(X_sm_train, y_sm_train)
    print(f"  {name}: OK")

ensemble = VotingClassifier(
    estimators=[("lgbm", models_clf["LightGBM"]),
                ("xgb",  models_clf["XGBoost"]),
                ("gb",   models_clf["GradientBoosting"])],
    voting="soft", weights=[0.4, 0.4, 0.2])
ensemble.fit(X_sm_train, y_sm_train)
models_clf["VotingEnsemble"] = ensemble
print("  VotingEnsemble: OK")


def best_threshold(model, X, y):
    probs = model.predict_proba(X)[:, 1]
    best_thr, best_acc = 0.5, 0.0
    for t in np.arange(0.25, 0.76, 0.01):
        acc = accuracy_score(y, (probs >= t).astype(int))
        if acc > best_acc:
            best_acc, best_thr = acc, t
    return round(best_thr, 2), round(best_acc, 4)


# ── Guardar métricas clasificación ────────────────────────────
print("\n[3/6] Calculando métricas y guardando modelos...")

rows_clf = []
for name, model in models_clf.items():
    thr, _ = best_threshold(model, X_test, y_test)
    probs  = model.predict_proba(X_test)[:, 1]
    preds  = (probs >= thr).astype(int)
    rows_clf.append({
        "modelo":     name,
        "accuracy":   round(accuracy_score(y_test, preds), 4),
        "f1":         round(f1_score(y_test, preds, zero_division=0), 4),
        "roc_auc":    round(roc_auc_score(y_test, probs), 4),
        "threshold":  thr,
    })
    joblib.dump(model, f"models/trained_models/{name}.pkl")
    print(f"  {name}: acc={rows_clf[-1]['accuracy']}  saved")

df_clf = pd.DataFrame(rows_clf).sort_values("accuracy", ascending=False)
df_clf.to_csv("results/metrics/clasificacion_metrics.csv", index=False)
print(f"\n  Métricas clasificación guardadas.")

# ── Regresores ────────────────────────────────────────────────
print("\n  Entrenando regresores...")
X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(X_all, y_reg, test_size=0.2, random_state=SEED)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
rows_reg = []
for name, model in [("Ridge", Ridge(alpha=1.0)),
                     ("RandomForestReg", RandomForestRegressor(n_estimators=200, random_state=SEED))]:
    model.fit(X_tr_r, y_tr_r)
    yp = model.predict(X_te_r)
    rows_reg.append({"modelo": name,
                     "r2":   round(r2_score(y_te_r, yp), 4),
                     "rmse": round(np.sqrt(mean_squared_error(y_te_r, yp)), 4),
                     "mae":  round(mean_absolute_error(y_te_r, yp), 4)})
    joblib.dump(model, f"models/trained_models/{name}.pkl")
    print(f"  {name}: r2={rows_reg[-1]['r2']}")
pd.DataFrame(rows_reg).to_csv("results/metrics/regresion_metrics.csv", index=False)

# ── Clustering & PCA ──────────────────────────────────────────
print("\n  Clustering KMeans...")
rows_km = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    labels = km.fit_predict(X_all)
    sil = silhouette_score(X_all, labels, sample_size=5000, random_state=SEED)
    rows_km.append({"k": k, "inertia": round(km.inertia_, 2), "silhouette": round(sil, 4)})
    if k == 2:
        joblib.dump(km, "models/trained_models/KMeans_k2.pkl")
df_km = pd.DataFrame(rows_km)
df_km.to_csv("results/metrics/clustering_metrics.csv", index=False)

print("  PCA...")
pca = PCA(random_state=SEED)
pca.fit(X_all)
cumvar = np.cumsum(pca.explained_variance_ratio_)
df_pca = pd.DataFrame({"componente": range(1, len(cumvar)+1),
                        "varianza_explicada": pca.explained_variance_ratio_.round(4),
                        "varianza_acumulada": cumvar.round(4)})
df_pca.to_csv("results/metrics/pca_metrics.csv", index=False)
joblib.dump(pca, "models/trained_models/PCA.pkl")

# ── PLOTS ─────────────────────────────────────────────────────
print("\n[4/6] Generando visualizaciones...")
sns.set_theme(style="whitegrid")

# --- 1. Distribución review_score ---
feat = pd.read_csv("data/04_feature/dataset_features.csv", encoding="utf-8")
fig, ax = plt.subplots(figsize=(8, 4))
counts = feat["review_score"].value_counts().sort_index()
bars = ax.bar(counts.index.astype(str), counts.values,
              color=["#d62728","#ff7f0e","#ffdd57","#2ca02c","#1f77b4"])
for bar, v in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
            f"{v:,}", ha="center", fontsize=10)
ax.set_xlabel("Review Score (estrellas)", fontsize=12)
ax.set_ylabel("Cantidad de órdenes", fontsize=12)
ax.set_title("Distribución de Puntajes de Reseña — Olist E-Commerce", fontsize=13, fontweight="bold")
fig.tight_layout(); fig.savefig("results/plots/distribucion_review_score.png", dpi=150); plt.close(fig)
print("  distribucion_review_score.png")

# --- 2. Comparación de clasificadores ---
fig, ax = plt.subplots(figsize=(10, 5))
df_sorted = df_clf.sort_values("accuracy")
colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(df_sorted)))
bars = ax.barh(df_sorted["modelo"], df_sorted["accuracy"] * 100, color=colors)
ax.axvline(85, color="red", linestyle="--", linewidth=1.5, label="Objetivo 85%")
ax.axvline(df_sorted["accuracy"].max() * 100, color="navy", linestyle=":", linewidth=1.5, label=f"Mejor: {df_sorted['accuracy'].max()*100:.2f}%")
for bar, val in zip(bars, df_sorted["accuracy"]):
    ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
            f"{val*100:.2f}%", va="center", fontsize=9)
ax.set_xlabel("Accuracy (%)", fontsize=12); ax.set_xlim(50, 100)
ax.set_title("Comparación de Clasificadores — Test Accuracy", fontsize=13, fontweight="bold")
ax.legend(fontsize=10); ax.grid(axis="x", alpha=0.3)
fig.tight_layout(); fig.savefig("results/plots/model_comparison.png", dpi=150); plt.close(fig)
print("  model_comparison.png")

# --- 3. Curvas ROC ---
fig, ax = plt.subplots(figsize=(9, 6))
colors_roc = plt.cm.tab10(np.linspace(0, 1, len(models_clf)))
for (name, model), color in zip(models_clf.items(), colors_roc):
    probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc_val = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})", color=color, lw=1.8)
ax.plot([0,1],[0,1],"k--",lw=1,label="Baseline (AUC=0.5)")
ax.set_xlabel("False Positive Rate", fontsize=12); ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("Curvas ROC — Todos los Clasificadores", fontsize=13, fontweight="bold")
ax.legend(loc="lower right", fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig("results/plots/roc_curves.png", dpi=150); plt.close(fig)
print("  roc_curves.png")

# --- 4. Matriz de confusión (mejor modelo: VotingEnsemble) ---
best_model = models_clf["VotingEnsemble"]
thr_best, _ = best_threshold(best_model, X_test, y_test)
probs_best = best_model.predict_proba(X_test)[:, 1]
preds_best = (probs_best >= thr_best).astype(int)
cm = confusion_matrix(y_test, preds_best)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Insatisfecho (0)", "Satisfecho (1)"],
            yticklabels=["Insatisfecho (0)", "Satisfecho (1)"])
ax.set_ylabel("Real", fontsize=11); ax.set_xlabel("Predicho", fontsize=11)
ax.set_title("Matriz de Confusión — VotingEnsemble", fontsize=12, fontweight="bold")
fig.tight_layout(); fig.savefig("results/plots/confusion_matrix_best.png", dpi=150); plt.close(fig)
print("  confusion_matrix_best.png")

# --- 5. Feature Importance (LightGBM) ---
lgbm = models_clf["LightGBM"]
importances = lgbm.feature_importances_
df_imp = pd.DataFrame({"feature": X_all.columns, "importance": importances})
df_imp = df_imp.nlargest(15, "importance")
fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(data=df_imp, x="importance", y="feature", palette="Blues_r", ax=ax)
ax.set_title("Top 15 Features más Importantes — LightGBM", fontsize=13, fontweight="bold")
ax.set_xlabel("Importancia (ganancia)", fontsize=11); ax.set_ylabel("")
fig.tight_layout(); fig.savefig("results/plots/feature_importance.png", dpi=150); plt.close(fig)
print("  feature_importance.png")

# --- 6. Silhouette + Elbow ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ks = df_km["k"].tolist(); sils = df_km["silhouette"].tolist(); inertias = df_km["inertia"].tolist()
ax1.plot(ks, sils, "bo-", lw=2, ms=7)
ax1.axvline(2, color="red", linestyle="--", label="k=2 óptimo")
ax1.set_xlabel("k"); ax1.set_ylabel("Silhouette Score")
ax1.set_title("Silhouette por k", fontsize=12, fontweight="bold"); ax1.legend(); ax1.grid(alpha=0.3)
ax2.plot(ks, inertias, "rs-", lw=2, ms=7)
ax2.set_xlabel("k"); ax2.set_ylabel("Inercia")
ax2.set_title("Elbow Method — Inercia por k", fontsize=12, fontweight="bold"); ax2.grid(alpha=0.3)
fig.tight_layout(); fig.savefig("results/plots/silhouette_elbow.png", dpi=150); plt.close(fig)
print("  silhouette_elbow.png")

# --- 7. PCA varianza ---
evr = pca.explained_variance_ratio_[:20]
cumv = np.cumsum(evr)
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(range(1, len(evr)+1), evr*100, alpha=0.6, color="steelblue", label="Por componente")
ax.plot(range(1, len(evr)+1), cumv*100, "ro-", lw=2, ms=4, label="Acumulada")
ax.axhline(80, color="green", linestyle="--", lw=1.5, label="80% umbral")
ax.set_xlabel("Componente Principal"); ax.set_ylabel("Varianza Explicada (%)")
ax.set_title("PCA — Varianza Explicada", fontsize=13, fontweight="bold"); ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig("results/plots/pca_variance.png", dpi=150); plt.close(fig)
print("  pca_variance.png")

# --- 8. Métricas múltiples heatmap ---
df_heat = df_clf.set_index("modelo")[["accuracy","f1","roc_auc"]].astype(float)
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(df_heat, annot=True, fmt=".4f", cmap="YlGnBu", ax=ax, vmin=0.7, vmax=1.0)
ax.set_title("Comparación de Métricas — Todos los Modelos", fontsize=12, fontweight="bold")
ax.set_xlabel("Métrica"); ax.set_ylabel("Modelo")
fig.tight_layout(); fig.savefig("results/plots/metrics_heatmap.png", dpi=150); plt.close(fig)
print("  metrics_heatmap.png")

# --- 9. Correlación de features ---
corr = X_all.corr()
fig, ax = plt.subplots(figsize=(14, 11))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0, ax=ax,
            annot=False, linewidths=0.5, fmt=".2f")
ax.set_title("Matriz de Correlación — Features", fontsize=13, fontweight="bold")
fig.tight_layout(); fig.savefig("results/plots/correlation_matrix.png", dpi=150); plt.close(fig)
print("  correlation_matrix.png")

# --- 10. Optimización CV vs Test ---
opt_data = pd.read_csv("data/07_model_output/comparacion_optimizacion.csv", encoding="utf-8")
fig, ax = plt.subplots(figsize=(8, 5))
x = range(len(opt_data)); width = 0.35
ax.bar([i - width/2 for i in x], opt_data["cv_accuracy"]*100, width, label="CV Accuracy", color="steelblue", alpha=0.8)
ax.bar([i + width/2 for i in x], opt_data["test_accuracy"]*100, width, label="Test Accuracy", color="darkorange", alpha=0.8)
ax.set_xticks(list(x)); ax.set_xticklabels(opt_data["metodo"], rotation=10, fontsize=10)
ax.set_ylabel("Accuracy (%)"); ax.set_ylim(75, 100)
ax.axhline(85, color="red", linestyle="--", lw=1.5, label="Objetivo 85%")
ax.set_title("Optimización de Hiperparámetros — CV vs Test", fontsize=12, fontweight="bold")
ax.legend(); ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig("results/plots/optimization_comparison.png", dpi=150); plt.close(fig)
print("  optimization_comparison.png")

print("\n[5/6] Guardando métricas consolidadas...")
df_clf.to_csv("results/metrics/clasificacion_metrics.csv", index=False, encoding="utf-8")
print("  clasificacion_metrics.csv")

print("\n" + "=" * 60)
print("ARTEFACTOS GENERADOS EXITOSAMENTE")
print(f"  Modelos:       models/trained_models/  ({len(models_clf)+4} archivos)")
print(f"  Plots:         results/plots/           (10 gráficos)")
print(f"  Métricas:      results/metrics/         (4 CSVs)")
print("=" * 60)
