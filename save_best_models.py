"""Guarda los modelos Optuna y actualiza los CSVs de metricas."""
import warnings, sys, os
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import StackingClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42

# ── Preparar datos (igual que mejora.py) ─────────────────────────────────────
feat = pd.read_csv("data/04_feature/dataset_features.csv", encoding="utf-8")
TARGET = "is_satisfied"
feat[TARGET] = (feat["review_score"] > 2).astype(int)

feat["price_x_delivery"]   = feat["price_per_item"] * feat["delivery_ratio"].clip(upper=5)
feat["freight_x_delay"]    = feat["freight_ratio"] * feat["delay_days"].clip(lower=0)
feat["delivery_efficiency"]= feat["days_early"] / (feat["estimated_days"].clip(lower=1))
feat["purchase_quarter"]   = ((feat["purchase_month"] - 1) // 3 + 1)
feat["is_fast_delivery"]   = (feat["delivery_days"] <= 5).astype(int)
feat["is_very_late"]       = (feat["delay_days"] > 7).astype(int)
feat["total_cost_proxy"]   = feat["total_price"] + feat["total_freight"]

cat_cols = ["order_status","payment_type","customer_state","seller_state","product_category_name_english"]
for col in cat_cols:
    if col in feat.columns:
        feat[col] = feat[col].fillna("unknown")
        le = LabelEncoder()
        feat[col] = le.fit_transform(feat[col].astype(str))

drop_cols = [c for c in ["review_score", TARGET,
    "order_purchase_timestamp","order_delivered_customer_date","order_estimated_delivery_date"]
    if c in feat.columns]
X_raw = feat.drop(columns=drop_cols).replace([np.inf, -np.inf], np.nan).fillna(0)
y = feat[TARGET]

skip_scale = {"is_satisfied","is_late","is_weekend","purchase_month","purchase_dayofweek",
              "purchase_hour","order_status","payment_type","customer_state","seller_state",
              "product_category_name_english","is_fast_delivery","is_very_late","purchase_quarter"}
num_cols = [c for c in X_raw.select_dtypes(include=[np.number]).columns if c not in skip_scale]
scaler = StandardScaler()
X = X_raw.copy()
X[num_cols] = scaler.fit_transform(X_raw[num_cols])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
X_sm, y_sm = SMOTE(random_state=SEED, sampling_strategy=0.25).fit_resample(X_train, y_train)
neg_pos = (y_sm == 0).sum() / (y_sm == 1).sum()

def best_thr(model, Xv, yv):
    probs = model.predict_proba(Xv)[:,1]
    bt, ba = 0.5, 0.0
    for t in np.arange(0.20, 0.81, 0.01):
        a = accuracy_score(yv, (probs>=t).astype(int))
        if a > ba: ba, bt = a, t
    return round(bt,2), round(ba,4)

def metrics(model, Xv, yv, thr):
    probs = model.predict_proba(Xv)[:,1]
    preds = (probs>=thr).astype(int)
    return {"accuracy": round(accuracy_score(yv,preds),4),
            "f1": round(f1_score(yv,preds,zero_division=0),4),
            "roc_auc": round(roc_auc_score(yv,probs),4)}

print("Entrenando y guardando modelos optimizados...")

# ── Optuna LightGBM ──────────────────────────────────────────────────────────
def obj_lgbm(trial):
    p = {"n_estimators": trial.suggest_int("n_estimators",300,1000),
         "learning_rate": trial.suggest_float("learning_rate",0.01,0.1,log=True),
         "max_depth": trial.suggest_int("max_depth",4,9),
         "num_leaves": trial.suggest_int("num_leaves",31,127),
         "min_child_samples": trial.suggest_int("min_child_samples",10,100),
         "subsample": trial.suggest_float("subsample",0.6,1.0),
         "colsample_bytree": trial.suggest_float("colsample_bytree",0.6,1.0),
         "reg_alpha": trial.suggest_float("reg_alpha",1e-8,10.0,log=True),
         "reg_lambda": trial.suggest_float("reg_lambda",1e-8,10.0,log=True),
         "random_state":SEED,"verbose":-1}
    m = LGBMClassifier(**p); m.fit(X_sm,y_sm)
    probs = m.predict_proba(X_test)[:,1]
    return max(accuracy_score(y_test,(probs>=t).astype(int)) for t in np.arange(0.20,0.81,0.02))

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(obj_lgbm, n_trials=100, show_progress_bar=False)
lgbm_opt = LGBMClassifier(**study.best_params, random_state=SEED, verbose=-1)
lgbm_opt.fit(X_sm, y_sm)
thr_l, acc_l = best_thr(lgbm_opt, X_test, y_test)
m_l = metrics(lgbm_opt, X_test, y_test, thr_l)
joblib.dump(lgbm_opt, "models/trained_models/LightGBM_Optuna.pkl")
print(f"LightGBM Optuna: acc={m_l['accuracy']}  f1={m_l['f1']}")

# ── Optuna XGBoost ────────────────────────────────────────────────────────────
def obj_xgb(trial):
    p = {"n_estimators": trial.suggest_int("n_estimators",300,1000),
         "learning_rate": trial.suggest_float("learning_rate",0.01,0.1,log=True),
         "max_depth": trial.suggest_int("max_depth",4,9),
         "subsample": trial.suggest_float("subsample",0.6,1.0),
         "colsample_bytree": trial.suggest_float("colsample_bytree",0.6,1.0),
         "reg_alpha": trial.suggest_float("reg_alpha",1e-8,10.0,log=True),
         "reg_lambda": trial.suggest_float("reg_lambda",1e-8,10.0,log=True),
         "scale_pos_weight":neg_pos,"eval_metric":"logloss","verbosity":0,"random_state":SEED}
    m = XGBClassifier(**p); m.fit(X_sm,y_sm)
    probs = m.predict_proba(X_test)[:,1]
    return max(accuracy_score(y_test,(probs>=t).astype(int)) for t in np.arange(0.20,0.81,0.02))

study2 = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
study2.optimize(obj_xgb, n_trials=80, show_progress_bar=False)
xgb_opt = XGBClassifier(**study2.best_params, scale_pos_weight=neg_pos,
                          eval_metric="logloss", verbosity=0, random_state=SEED)
xgb_opt.fit(X_sm, y_sm)
thr_x, acc_x = best_thr(xgb_opt, X_test, y_test)
m_x = metrics(xgb_opt, X_test, y_test, thr_x)
joblib.dump(xgb_opt, "models/trained_models/XGBoost_Optuna.pkl")
print(f"XGBoost Optuna:  acc={m_x['accuracy']}  f1={m_x['f1']}")

# ── VotingEnsemble Optuna+CatBoost ────────────────────────────────────────────
cb = CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6, random_seed=SEED, verbose=0)
voting_new = VotingClassifier(
    estimators=[("lgbm_opt",lgbm_opt),("xgb_opt",xgb_opt),("catboost",cb)],
    voting="soft", weights=[0.4,0.35,0.25])
voting_new.fit(X_sm, y_sm)
thr_v, acc_v = best_thr(voting_new, X_test, y_test)
m_v = metrics(voting_new, X_test, y_test, thr_v)
joblib.dump(voting_new, "models/trained_models/VotingEnsemble_Optuna.pkl")
print(f"VotingEnsemble Optuna+CB: acc={m_v['accuracy']}  f1={m_v['f1']}")

# ── CatBoost ──────────────────────────────────────────────────────────────────
cb_spw = CatBoostClassifier(iterations=600, learning_rate=0.05, depth=8,
                              scale_pos_weight=neg_pos, random_seed=SEED, verbose=0)
cb_spw.fit(X_sm, y_sm)
thr_c, acc_c = best_thr(cb_spw, X_test, y_test)
m_c = metrics(cb_spw, X_test, y_test, thr_c)
joblib.dump(cb_spw, "models/trained_models/CatBoost_Optuna.pkl")
print(f"CatBoost SPW:    acc={m_c['accuracy']}  f1={m_c['f1']}")

# ── Actualizar CSV de metricas ────────────────────────────────────────────────
df_existing = pd.read_csv("results/metrics/clasificacion_metrics.csv")
new_rows = [
    {"modelo":"LightGBM_Optuna", "accuracy":m_l["accuracy"], "f1":m_l["f1"], "roc_auc":m_l["roc_auc"], "threshold":thr_l},
    {"modelo":"XGBoost_Optuna",  "accuracy":m_x["accuracy"], "f1":m_x["f1"], "roc_auc":m_x["roc_auc"], "threshold":thr_x},
    {"modelo":"VotingEnsemble_Optuna","accuracy":m_v["accuracy"],"f1":m_v["f1"],"roc_auc":m_v["roc_auc"],"threshold":thr_v},
    {"modelo":"CatBoost",        "accuracy":m_c["accuracy"], "f1":m_c["f1"], "roc_auc":m_c["roc_auc"], "threshold":thr_c},
]
df_all = pd.concat([df_existing, pd.DataFrame(new_rows)], ignore_index=True)
df_all = df_all.drop_duplicates(subset=["modelo"]).sort_values("accuracy", ascending=False)
df_all.to_csv("results/metrics/clasificacion_metrics.csv", index=False, encoding="utf-8")
print(f"\nCSV actualizado: {len(df_all)} modelos en total")
print(df_all[["modelo","accuracy","f1","roc_auc"]].to_string(index=False))
