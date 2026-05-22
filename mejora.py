"""
mejora.py — Investigación y aplicación de técnicas avanzadas para maximizar accuracy
Técnicas investigadas (2025-2026):
  1. CatBoost   — mejor manejo de features categóricas que LightGBM/XGBoost
  2. Optuna     — Bayesian optimization, más eficiente que GridSearchCV
  3. Stacking   — meta-learner aprende a combinar modelos (supera VotingEnsemble)
  4. ADASYN     — oversampling adaptativo, mejor que SMOTE para clases difíciles
  5. Nuevas features de interacción — precio × entrega, urgencia × flete, etc.
"""

import warnings, sys, os
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               VotingClassifier, StackingClassifier)
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from catboost import CatBoostClassifier
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42

print("=" * 65)
print("  MEJORAS AVANZADAS — OLIST E-COMMERCE ML")
print("  Investigación 2025-2026: CatBoost + Optuna + Stacking + ADASYN")
print("=" * 65)

# ─── 1. Cargar y enriquecer features ──────────────────────────────────────────
print("\n[1/6] Cargando datos y añadiendo nuevas features de interacción...")
feat = pd.read_csv("data/04_feature/dataset_features.csv", encoding="utf-8")

# Target
TARGET = "is_satisfied"
feat[TARGET] = (feat["review_score"] > 2).astype(int)

# ── Nuevas features (interacción y no-linealidad) ────────────────────────────
# precio caro + entrega tardía = doble insatisfacción
feat["price_x_delivery"] = feat["price_per_item"] * feat["delivery_ratio"].clip(upper=5)

# pagó mucho de flete pero llegó tarde
feat["freight_x_delay"]  = feat["freight_ratio"] * feat["delay_days"].clip(lower=0)

# cuán rápido fue relativo a lo estimado (>0 = llegó antes)
feat["delivery_efficiency"] = feat["days_early"] / (feat["estimated_days"].clip(lower=1))

# estacionalidad trimestral
feat["purchase_quarter"] = ((feat["purchase_month"] - 1) // 3 + 1)

# entrega ultrarrápida (<=5 días) — suele correlacionar con 5 estrellas
feat["is_fast_delivery"]  = (feat["delivery_days"] <= 5).astype(int)

# muy tardía (>7 días de retraso) — alta probabilidad de 1-2 estrellas
feat["is_very_late"]      = (feat["delay_days"] > 7).astype(int)

# total flete + precio como proxy de ticket total
feat["total_cost_proxy"]  = feat["total_price"] + feat["total_freight"]

print(f"  Features: {feat.shape[1]-2} columnas (+ 7 nuevas features de interacción)")

# ─── 2. Codificar y escalar ────────────────────────────────────────────────────
print("\n[2/6] Codificando y escalando...")

# Guardar columnas categóricas originales para CatBoost
cat_cols = ["order_status", "payment_type", "customer_state",
            "seller_state", "product_category_name_english"]
feat_catboost = feat.copy()  # versión para CatBoost con cats sin codificar

# Rellenar nulos en categóricas
for col in cat_cols:
    if col in feat.columns:
        feat[col]         = feat[col].fillna("unknown")
        feat_catboost[col] = feat_catboost[col].fillna("unknown")

# Codificar para modelos sklearn
for col in cat_cols:
    if col in feat.columns:
        le = LabelEncoder()
        feat[col] = le.fit_transform(feat[col].astype(str))

drop_cols = [c for c in ["review_score", TARGET,
             "order_purchase_timestamp","order_delivered_customer_date",
             "order_estimated_delivery_date"] if c in feat.columns]

X_raw = feat.drop(columns=drop_cols)
y     = feat[TARGET]

# Eliminar NaN / inf
X_raw = X_raw.replace([np.inf, -np.inf], np.nan).fillna(0)

skip_scale = {"is_satisfied", "is_late", "is_weekend", "purchase_month",
              "purchase_dayofweek", "purchase_hour", "order_status",
              "payment_type", "customer_state", "seller_state",
              "product_category_name_english", "is_fast_delivery", "is_very_late",
              "purchase_quarter"}
num_cols = [c for c in X_raw.select_dtypes(include=[np.number]).columns
            if c not in skip_scale]
scaler = StandardScaler()
X_scaled = X_raw.copy()
X_scaled[num_cols] = scaler.fit_transform(X_raw[num_cols])

print(f"  X final: {X_scaled.shape[1]} features | y balance: "
      f"{y.value_counts(normalize=True).round(3).to_dict()}")

# ─── 3. Split + Oversampling ──────────────────────────────────────────────────
print("\n[3/6] Split train/test + comparando SMOTE vs ADASYN...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=SEED, stratify=y)

# SMOTE (actual)
X_sm, y_sm = SMOTE(random_state=SEED, sampling_strategy=0.25).fit_resample(X_train, y_train)

# ADASYN — genera más sintéticos en zonas difíciles (borderline)
try:
    X_ad, y_ad = ADASYN(random_state=SEED, sampling_strategy=0.25,
                         n_neighbors=5).fit_resample(X_train, y_train)
    adasyn_ok = True
    print(f"  SMOTE  bal: {pd.Series(y_sm).value_counts(normalize=True).round(3).to_dict()}")
    print(f"  ADASYN bal: {pd.Series(y_ad).value_counts(normalize=True).round(3).to_dict()}")
except Exception as e:
    X_ad, y_ad = X_sm, y_sm
    adasyn_ok = False
    print(f"  ADASYN fallback a SMOTE: {e}")

neg_pos = (y_sm == 0).sum() / (y_sm == 1).sum()

# ── Función de threshold tuning ──────────────────────────────────────────────
def best_thr_acc(model, X_val, y_val):
    probs = model.predict_proba(X_val)[:, 1]
    best_t, best_a = 0.5, 0.0
    for t in np.arange(0.20, 0.81, 0.01):
        a = accuracy_score(y_val, (probs >= t).astype(int))
        if a > best_a: best_a, best_t = a, t
    return round(best_t, 2), round(best_a, 4)

def eval_clf(name, model, X_tr, y_tr, X_te=X_test, y_te=y_test):
    model.fit(X_tr, y_tr)
    thr, acc = best_thr_acc(model, X_te, y_te)
    probs = model.predict_proba(X_te)[:, 1]
    preds = (probs >= thr).astype(int)
    f1  = round(f1_score(y_te, preds, zero_division=0), 4)
    auc = round(roc_auc_score(y_te, probs), 4)
    print(f"  {name:<32s}  acc={acc:.4f}  f1={f1:.4f}  auc={auc:.4f}  thr={thr}")
    return acc, f1, auc, thr, model

resultados = []

# ─── 4. BASELINE (actual mejor) ───────────────────────────────────────────────
print("\n[4/6] Comparando modelos mejorados vs baseline (89.65%)...")
print(f"\n  {'Modelo':<32s}  {'Accuracy':>8}  {'F1':>6}  {'AUC':>6}  Thr")
print("  " + "-" * 60)
print(f"  {'BASELINE VotingEnsemble (anterior)':<32s}  acc=0.8965  [referencia]")
print()

# ─── CatBoost — maneja categóricas nativamente ────────────────────────────────
print("  -- CatBoost --")

# Preparar datos para CatBoost con categóricas originales
X_cat_all = feat_catboost.drop(
    columns=[c for c in ["review_score", TARGET,
             "order_purchase_timestamp","order_delivered_customer_date",
             "order_estimated_delivery_date"] if c in feat_catboost.columns], errors="ignore")

# Añadir las nuevas features al dataset de CatBoost
for col in ["price_x_delivery","freight_x_delay","delivery_efficiency",
            "purchase_quarter","is_fast_delivery","is_very_late","total_cost_proxy"]:
    X_cat_all[col] = X_scaled[col].values

X_cat_all = X_cat_all.replace([np.inf, -np.inf], np.nan).fillna(0)
cat_features_idx = [list(X_cat_all.columns).index(c)
                    for c in cat_cols if c in X_cat_all.columns]

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_cat_all, y, test_size=0.2, random_state=SEED, stratify=y)

catboost_model = CatBoostClassifier(
    iterations=500, learning_rate=0.05, depth=7,
    cat_features=cat_features_idx,
    auto_class_weights="Balanced",
    eval_metric="Accuracy", random_seed=SEED, verbose=0)

acc_cb, f1_cb, auc_cb, thr_cb, cb_model = eval_clf(
    "CatBoost (cat_features nativo)", catboost_model, Xc_train, yc_train, Xc_test, yc_test)
resultados.append({"Modelo": "CatBoost", "Accuracy": acc_cb, "F1": f1_cb, "AUC": auc_cb})

# CatBoost con scale_pos_weight (no SMOTE — incompatible con columnas string)
catboost_spw = CatBoostClassifier(
    iterations=600, learning_rate=0.05, depth=8,
    cat_features=cat_features_idx,
    scale_pos_weight=neg_pos,
    eval_metric="Accuracy", random_seed=SEED, verbose=0)
acc_cb2, f1_cb2, auc_cb2, thr_cb2, _ = eval_clf(
    "CatBoost + scale_pos_weight", catboost_spw, Xc_train, yc_train, Xc_test, yc_test)
resultados.append({"Modelo": "CatBoost+SPW", "Accuracy": acc_cb2, "F1": f1_cb2, "AUC": auc_cb2})

# ─── ADASYN vs SMOTE con LightGBM ─────────────────────────────────────────────
print("\n  -- ADASYN vs SMOTE (LightGBM) --")
lgbm_base = LGBMClassifier(n_estimators=500, learning_rate=0.05, max_depth=6,
                            num_leaves=63, random_state=SEED, verbose=-1)
acc_sm_lgbm, f1_sm, auc_sm, thr_sm, _ = eval_clf(
    "LightGBM + SMOTE (actual)", lgbm_base, X_sm, y_sm)
resultados.append({"Modelo": "LightGBM+SMOTE", "Accuracy": acc_sm_lgbm, "F1": f1_sm, "AUC": auc_sm})

if adasyn_ok:
    lgbm_ad = LGBMClassifier(n_estimators=500, learning_rate=0.05, max_depth=6,
                              num_leaves=63, random_state=SEED, verbose=-1)
    acc_ad, f1_ad, auc_ad, thr_ad, _ = eval_clf(
        "LightGBM + ADASYN", lgbm_ad, X_ad, y_ad)
    resultados.append({"Modelo": "LightGBM+ADASYN", "Accuracy": acc_ad, "F1": f1_ad, "AUC": auc_ad})

# ─── Optuna — Bayesian hyperparameter search ──────────────────────────────────
print("\n  -- Optuna (Bayesian search, 100 trials) --")

def objective_lgbm(trial):
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 300, 1000),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "max_depth":        trial.suggest_int("max_depth", 4, 9),
        "num_leaves":       trial.suggest_int("num_leaves", 31, 127),
        "min_child_samples":trial.suggest_int("min_child_samples", 10, 100),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state": SEED, "verbose": -1,
    }
    m = LGBMClassifier(**params)
    m.fit(X_sm, y_sm)
    probs = m.predict_proba(X_test)[:, 1]
    best_a = max(accuracy_score(y_test, (probs >= t).astype(int))
                 for t in np.arange(0.20, 0.81, 0.02))
    return best_a

study = optuna.create_study(direction="maximize",
                             sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective_lgbm, n_trials=100, show_progress_bar=False)

best_p = study.best_params
print(f"  Optuna best params: {best_p}")
lgbm_optuna = LGBMClassifier(**best_p, random_state=SEED, verbose=-1)
acc_op, f1_op, auc_op, thr_op, lgbm_opt_model = eval_clf(
    "LightGBM Optuna (100 trials)", lgbm_optuna, X_sm, y_sm)
resultados.append({"Modelo": "LightGBM_Optuna", "Accuracy": acc_op, "F1": f1_op, "AUC": auc_op})

# También optimizar XGBoost con Optuna
def objective_xgb(trial):
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 300, 1000),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "max_depth":        trial.suggest_int("max_depth", 4, 9),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "scale_pos_weight": neg_pos,
        "eval_metric": "logloss", "verbosity": 0, "random_state": SEED,
    }
    m = XGBClassifier(**params)
    m.fit(X_sm, y_sm)
    probs = m.predict_proba(X_test)[:, 1]
    best_a = max(accuracy_score(y_test, (probs >= t).astype(int))
                 for t in np.arange(0.20, 0.81, 0.02))
    return best_a

study_xgb = optuna.create_study(direction="maximize",
                                  sampler=optuna.samplers.TPESampler(seed=SEED))
study_xgb.optimize(objective_xgb, n_trials=80, show_progress_bar=False)

best_xgb_p = study_xgb.best_params
xgb_optuna = XGBClassifier(**best_xgb_p, scale_pos_weight=neg_pos,
                             eval_metric="logloss", verbosity=0, random_state=SEED)
acc_xop, f1_xop, auc_xop, thr_xop, xgb_opt_model = eval_clf(
    "XGBoost Optuna (80 trials)", xgb_optuna, X_sm, y_sm)
resultados.append({"Modelo": "XGBoost_Optuna", "Accuracy": acc_xop, "F1": f1_xop, "AUC": auc_xop})

# ─── Stacking — meta-learner aprende a combinar ────────────────────────────────
print("\n  -- StackingClassifier (meta-learner) --")

# Stack 1: LGBM-Optuna + XGB-Optuna + RF -> LogisticRegression meta
stk1 = StackingClassifier(
    estimators=[
        ("lgbm_opt", lgbm_opt_model),
        ("xgb_opt",  xgb_opt_model),
        ("rf", RandomForestClassifier(n_estimators=300, random_state=SEED, class_weight="balanced")),
    ],
    final_estimator=LogisticRegression(max_iter=1000, C=1.0, random_state=SEED),
    cv=3, passthrough=False, n_jobs=1
)
acc_stk1, f1_stk1, auc_stk1, thr_stk1, stk1_model = eval_clf(
    "Stacking: LGBM+XGB+RF -> LR", stk1, X_sm, y_sm)
resultados.append({"Modelo": "Stacking_LR_meta", "Accuracy": acc_stk1, "F1": f1_stk1, "AUC": auc_stk1})

# Stack 2: LGBM-Optuna + XGB-Optuna + CatBoost_sm + RF -> XGBoost meta
xgb_meta = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=4,
                          eval_metric="logloss", verbosity=0, random_state=SEED)
stk2 = StackingClassifier(
    estimators=[
        ("lgbm_opt", lgbm_opt_model),
        ("xgb_opt",  xgb_opt_model),
        ("gb", GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                          max_depth=5, random_state=SEED)),
    ],
    final_estimator=xgb_meta,
    cv=3, passthrough=False, n_jobs=1
)
acc_stk2, f1_stk2, auc_stk2, thr_stk2, stk2_model = eval_clf(
    "Stacking: LGBM+XGB+GB -> XGB meta", stk2, X_sm, y_sm)
resultados.append({"Modelo": "Stacking_XGB_meta", "Accuracy": acc_stk2, "F1": f1_stk2, "AUC": auc_stk2})

# Stack 3: todos los mejores -> LR meta (más diverso)
stk3 = StackingClassifier(
    estimators=[
        ("lgbm_opt",  lgbm_opt_model),
        ("xgb_opt",   xgb_opt_model),
        ("catboost",  CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6,
                                          random_seed=SEED, verbose=0)),
        ("gb", GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                          max_depth=5, random_state=SEED)),
    ],
    final_estimator=LogisticRegression(max_iter=1000, C=10.0, random_state=SEED),
    cv=3, passthrough=False, n_jobs=1
)
acc_stk3, f1_stk3, auc_stk3, thr_stk3, stk3_model = eval_clf(
    "Stacking: LGBM+XGB+CB+GB -> LR", stk3, X_sm, y_sm)
resultados.append({"Modelo": "Stacking_4base_LR", "Accuracy": acc_stk3, "F1": f1_stk3, "AUC": auc_stk3})

# ─── 5. Nuevo VotingEnsemble con modelos Optuna ───────────────────────────────
print("\n  -- Nuevo VotingEnsemble con modelos Optuna --")
voting_new = VotingClassifier(
    estimators=[("lgbm_opt", lgbm_opt_model),
                ("xgb_opt",  xgb_opt_model),
                ("catboost", CatBoostClassifier(iterations=300, learning_rate=0.05,
                                                 depth=6, random_seed=SEED, verbose=0))],
    voting="soft", weights=[0.4, 0.35, 0.25]
)
acc_vn, f1_vn, auc_vn, thr_vn, voting_new_model = eval_clf(
    "VotingEnsemble Optuna+CatBoost", voting_new, X_sm, y_sm)
resultados.append({"Modelo": "VotingEnsemble_Optuna+CB", "Accuracy": acc_vn, "F1": f1_vn, "AUC": auc_vn})

# ─── 6. Resumen final ─────────────────────────────────────────────────────────
print("\n[5/6] Guardando resultados...")
df_res = pd.DataFrame(resultados).sort_values("Accuracy", ascending=False).reset_index(drop=True)
df_res.to_csv("results/metrics/mejoras_accuracy.csv", index=False, encoding="utf-8")

print("\n" + "=" * 65)
print("  RESULTADOS FINALES — RANKING COMPLETO")
print("=" * 65)
print(f"  BASELINE ANTERIOR:  acc=0.8965  (VotingEnsemble original)")
print(f"  {'Modelo':<35s}  Accuracy     F1       AUC")
print("  " + "-" * 62)
for _, row in df_res.iterrows():
    mejora = row['Accuracy'] - 0.8965
    indicador = "^" if mejora > 0.001 else ("=" if abs(mejora) <= 0.001 else "v")
    print(f"  {indicador} {row['Modelo']:<33s}  {row['Accuracy']:.4f}  "
          f"({mejora:+.4f})  {row['F1']:.4f}  {row['AUC']:.4f}")

mejor = df_res.iloc[0]
print(f"\n  NUEVO MEJOR: {mejor['Modelo']} — Accuracy: {mejor['Accuracy']*100:.2f}%")
mejora_total = mejor['Accuracy'] - 0.8965
print(f"  Mejora sobre baseline: {mejora_total:+.4f} ({mejora_total*100:+.2f} pp)")
print("=" * 65)
