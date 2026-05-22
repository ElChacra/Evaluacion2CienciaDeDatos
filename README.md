# Olist E-Commerce — Predicción de Satisfacción del Cliente

**Asignatura:** SCY1101 — Programación para la Ciencia de Datos  
**Evaluación Parcial N°2** | Duoc UC  
[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

## Descripción del Proyecto

Sistema de machine learning end-to-end para predecir si un cliente quedará satisfecho con su compra en el marketplace brasileño Olist, usando datos logísticos, de precio y de producto de ~100K órdenes reales.

**Target:** `is_satisfied = (review_score > 2)` — clasifica órdenes como "satisfactorias" (3-5 estrellas) o "insatisfactorias" (1-2 estrellas).

**Resultado obtenido:** 89.71% accuracy (XGBoost_Optuna) — supera el objetivo del 85% en +4.71 pp.

---

## Estructura del Proyecto

```
olist-ecommerce/
├── notebooks/                          # Análisis y modelado paso a paso
│   ├── 01_exploratory_analysis.ipynb   # EDA, distribuciones, correlaciones
│   ├── 02_supervised_modeling.ipynb    # Clasificación y regresión
│   ├── 03_model_evaluation.ipynb       # Métricas, ROC, validación cruzada
│   ├── 04_hyperparameter_optimization.ipynb  # GridSearch + RandomizedSearch
│   └── 05_final_analysis.ipynb         # Integración y conclusiones
│
├── src/                                # Módulos Python reutilizables
│   ├── data_preprocessing.py           # Limpieza, features, encoding
│   ├── model_training.py               # Entrenamiento y serialización
│   ├── model_evaluation.py             # Métricas y visualizaciones
│   └── hyperparameter_tuning.py        # GridSearch y RandomizedSearch
│
├── src/olist_ecommerce/pipelines/      # Pipelines Kedro (producción)
│   ├── data_ingestion/                 # Merge de 9 CSVs
│   ├── data_cleaning/                  # Limpieza (Winsorización, imputación)
│   ├── data_transform/                 # Feature engineering
│   ├── data_validation/                # Validaciones automáticas
│   └── modeling/                       # Entrenamiento completo
│
├── data/
│   ├── 01_raw/                         # CSVs originales de Olist (Kaggle)
│   ├── 03_primary/                     # Dataset fusionado (9 tablas → 1)
│   ├── 04_feature/                     # Dataset con features creadas
│   ├── 05_model_input/                 # Dataset validado para ML
│   └── 07_model_output/                # Métricas y resultados de modelos
│
├── models/
│   └── trained_models/                 # Modelos serializados (.pkl)
│       ├── LightGBM.pkl, XGBoost.pkl, VotingEnsemble.pkl
│       ├── LightGBM_Optuna.pkl, XGBoost_Optuna.pkl        # Mejores modelos
│       ├── VotingEnsemble_Optuna.pkl, CatBoost_Optuna.pkl
│       ├── RandomForest.pkl, GradientBoosting.pkl
│       ├── LogisticRegression.pkl, DecisionTree.pkl
│       ├── Ridge.pkl, RandomForestReg.pkl
│       ├── KMeans_k2.pkl, PCA.pkl
│
├── results/
│   ├── metrics/                        # CSVs con métricas de cada modelo
│   │   ├── clasificacion_metrics.csv
│   │   ├── regresion_metrics.csv
│   │   ├── clustering_metrics.csv
│   │   └── pca_metrics.csv
│   └── plots/                          # 10 visualizaciones generadas
│
├── conf/base/catalog.yml               # Catálogo de datasets Kedro
├── generate_artifacts.py               # Script para regenerar plots y modelos
├── requirements.txt                    # Dependencias del proyecto
└── README.md                           # Este archivo
```

---

## Dataset

**Fuente:** [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) — Kaggle

**Tablas utilizadas (9 CSVs):**

| Archivo | Descripción |
|---------|-------------|
| `orders.csv` | Órdenes con timestamps y estado |
| `customers.csv` | Datos de clientes y estado |
| `order_items.csv` | Ítems por orden (precio, flete) |
| `payments.csv` | Métodos y montos de pago |
| `reviews.csv` | Reseñas y puntajes (target) |
| `products.csv` | Dimensiones y peso de productos |
| `sellers.csv` | Estado del vendedor |
| `category_translation.csv` | Traducción de categorías |

**Estadísticas del dataset limpio:**
- 95,811 órdenes entregadas
- 39 features finales (34 base + 7 features de interacción)
- 87.2% satisfechos (review > 2) / 12.8% insatisfechos

---

## Instalación y Uso

### Requisitos previos
- Python 3.11+

### 1. Crear entorno virtual e instalar dependencias

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Ejecutar el pipeline completo (Kedro)

```bash
# Pipeline completo end-to-end
.venv\Scripts\kedro.exe run

# O por etapas individuales
.venv\Scripts\kedro.exe run --pipeline=ingestion
.venv\Scripts\kedro.exe run --pipeline=cleaning
.venv\Scripts\kedro.exe run --pipeline=transform
.venv\Scripts\kedro.exe run --pipeline=validation
.venv\Scripts\kedro.exe run --pipeline=modeling
```

### 3. Regenerar plots y modelos serializados

```bash
.venv\Scripts\python.exe generate_artifacts.py
```

### 4. Abrir los notebooks

```bash
.venv\Scripts\jupyter.exe notebook notebooks/
```

Ejecutar en orden: `01_` → `02_` → `03_` → `04_` → `05_`

---

## Resultados Principales

### Clasificación (target: review_score > 2)

| Modelo | Accuracy | F1-Score | ROC-AUC | Método |
|--------|----------|----------|---------|--------|
| **XGBoost_Optuna** | **89.71%** | **0.9436** | 0.7717 | Optuna TPE 80 trials |
| LightGBM_Optuna | 89.70% | 0.9434 | 0.7714 | Optuna TPE 100 trials |
| VotingEnsemble_Optuna | 89.68% | 0.9433 | 0.7735 | LGBM+XGB+CatBoost |
| CatBoost | 89.66% | 0.9432 | 0.7717 | scale_pos_weight |
| VotingEnsemble | 89.65% | 0.9431 | 0.7756 | LGBM+XGB+GB |
| LightGBM | 89.62% | 0.9430 | 0.7707 | base |
| XGBoost | 89.60% | 0.9428 | 0.7737 | base |
| GradientBoosting | 89.59% | 0.9428 | 0.7743 | base |
| RandomForest | 89.52% | 0.9426 | 0.7586 | base |
| DecisionTree | 89.17% | 0.9403 | 0.7282 | base |
| LogisticRegression | 88.54% | 0.9361 | 0.7535 | base |

### Optimización de Hiperparámetros

| Método | CV Accuracy | Test Accuracy |
|--------|-------------|---------------|
| GridSearchCV (LightGBM) | 84.17% | 88.97% |
| RandomizedSearchCV (XGBoost) | 84.09% | 89.07% |
| **Optuna TPE LightGBM (100 trials)** | — | **89.70%** |
| **Optuna TPE XGBoost (80 trials)** | — | **89.71%** |

### No Supervisados

- **KMeans:** k=2 óptimo (silhouette=0.51)
- **PCA:** 2 componentes explican >= 80% de varianza

---

## Dependencias Principales

```
kedro~=1.3.0          # Framework de pipelines reproducibles
kedro-datasets>=9.3.0
pandas>=2.0.0         # Manipulación de datos
numpy>=1.24.0         # Operaciones numéricas
scikit-learn>=1.3.0   # Modelos base, GridSearch, métricas
lightgbm>=4.6.0       # Clasificador LightGBM
xgboost>=3.2.0        # Clasificador XGBoost
catboost==1.2.10      # Clasificador CatBoost
optuna==4.8.0         # Optimización Bayesiana (TPE)
imbalanced-learn>=0.14.0  # SMOTE / ADASYN
matplotlib>=3.7.0     # Visualizaciones
seaborn>=0.12.0       # Visualizaciones estadísticas
joblib                # Serialización de modelos
jupyter               # Notebooks interactivos
```

Ver [requirements.txt](requirements.txt) para versiones exactas.

---

## Reproducibilidad

- Todos los modelos usan `random_state=42` / `random_seed=42`
- El pipeline Kedro es determinista y completamente reproducible
- Los datos de entrenamiento/test se separan con `train_test_split(..., random_state=42, stratify=y)`
- SMOTE usa `random_state=42`, `sampling_strategy=0.25`
- Optuna usa `TPESampler(seed=42)` para reproducibilidad
- El script `generate_artifacts.py` regenera todos los artefactos desde cero en ~10 minutos
- El script `save_best_models.py` entrena y guarda los modelos Optuna (~30-60 minutos)
