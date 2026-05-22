"""
data_preprocessing.py
---------------------
Funciones de limpieza, transformación y preparación de datos para el dataset
Brazilian E-Commerce (Olist). Aplica principios de 'garbage in, garbage out':
valida, limpia y enriquece los datos antes de entrenar cualquier modelo.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

log = logging.getLogger(__name__)


# ─── Limpieza ─────────────────────────────────────────────────────────────────

def cap_iqr(df: pd.DataFrame, col: str, factor: float = 1.5) -> pd.DataFrame:
    """Winsoriza una columna reemplazando outliers con los límites del rango IQR.

    Args:
        df: DataFrame de entrada.
        col: Nombre de la columna a winsorizar.
        factor: Multiplicador del IQR para definir los límites (default 1.5).

    Returns:
        DataFrame con la columna capada.
    """
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    antes = ((df[col] < lower) | (df[col] > upper)).sum()
    df[col] = df[col].clip(lower=lower, upper=upper)
    log.info(f"  Outliers {col}: {antes} valores capados a [{lower:.2f}, {upper:.2f}]")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia el dataset crudo aplicando filtros, imputación y winsorización.

    Pasos:
        1. Filtra solo órdenes con estado 'delivered'.
        2. Elimina filas sin review_score (target).
        3. Parsea y valida timestamps.
        4. Descarta entregas con tiempos imposibles (< 0 o > 180 días).
        5. Imputa nulos de dimensiones de producto con la mediana.
        6. Imputa nulos de pago y categoría con valores por defecto.
        7. Winsorizacion IQR en columnas numéricas con outliers extremos.
        8. Elimina duplicados.
        9. Descarta columnas de identificación no útiles para modelado.

    Args:
        df: DataFrame crudo con todas las tablas del dataset Olist fusionadas.

    Returns:
        DataFrame limpio listo para ingeniería de features.

    Raises:
        ValueError: Si el DataFrame resultante tiene menos de 1000 filas.
        KeyError: Si faltan columnas requeridas en el DataFrame de entrada.
    """
    try:
        log.info(f"Filas iniciales: {df.shape[0]}")

        # 1. Solo órdenes entregadas
        df = df[df["order_status"] == "delivered"].copy()
        log.info(f"[1] Tras filtro 'delivered': {df.shape[0]}")

        # 2. Sin target
        n = df.shape[0]
        df = df.dropna(subset=["review_score"])
        log.info(f"[2] Eliminadas {n - df.shape[0]} filas sin review_score. Quedan: {df.shape[0]}")

        # 3. Timestamps
        ts_cols = ["order_purchase_timestamp", "order_delivered_customer_date",
                   "order_estimated_delivery_date"]
        for col in ts_cols:
            df[col] = pd.to_datetime(df[col], errors="coerce")
        n = df.shape[0]
        df = df.dropna(subset=ts_cols)
        log.info(f"[3] Eliminadas {n - df.shape[0]} filas con timestamps nulos.")

        # 4. Tiempos de entrega imposibles
        delivery_days = (df["order_delivered_customer_date"] - df["order_purchase_timestamp"]).dt.days
        mask = (delivery_days >= 0) & (delivery_days <= 180)
        n = df.shape[0]
        df = df[mask]
        log.info(f"[4] Eliminadas {n - df.shape[0]} filas con delivery_days fuera de [0,180].")

        # 5. Dimensiones de producto
        dim_cols = ["product_weight_g", "product_length_cm", "product_height_cm",
                    "product_width_cm", "product_photos_qty"]
        for col in dim_cols:
            if col in df.columns:
                nulos = df[col].isnull().sum()
                if nulos > 0:
                    df[col] = df[col].fillna(df[col].median())
                    log.info(f"[5] {col}: {nulos} nulos imputados con mediana.")

        # 6. Pago y categoría
        df["total_payment"] = df["total_payment"].fillna(df["total_price"])
        df["max_installments"] = df["max_installments"].fillna(1)
        df["payment_type"] = df["payment_type"].fillna("unknown")
        df["product_category_name_english"] = df["product_category_name_english"].fillna("unknown")
        df["customer_state"] = df["customer_state"].fillna("unknown")
        df["seller_state"] = df["seller_state"].fillna("unknown")

        # 7. Winsorización IQR
        cols_outlier = ["total_price", "total_freight", "total_payment",
                        "product_weight_g", "product_length_cm", "product_height_cm",
                        "product_width_cm", "product_photos_qty", "max_installments"]
        for col in cols_outlier:
            if col in df.columns:
                df = cap_iqr(df, col)

        if "n_items" in df.columns:
            p99 = df["n_items"].quantile(0.99)
            df["n_items"] = df["n_items"].clip(upper=p99)

        n = df.shape[0]
        df = df[df["total_price"] > 0]
        log.info(f"[7b] Eliminadas {n - df.shape[0]} filas con precio <= 0.")

        # 8. Duplicados
        n = df.shape[0]
        df = df.drop_duplicates()
        log.info(f"[8] Eliminados {n - df.shape[0]} duplicados. Quedan: {df.shape[0]}")

        # 9. Columnas de ID
        cols_drop = ["order_id", "customer_id", "customer_unique_id", "product_id",
                     "seller_id", "order_approved_at", "order_delivered_carrier_date",
                     "customer_zip_code_prefix", "seller_zip_code_prefix",
                     "product_name_lenght", "product_description_lenght"]
        df = df.drop(columns=[c for c in cols_drop if c in df.columns])

        if df.shape[0] < 1000:
            raise ValueError(f"Dataset muy pequeño después de limpieza: {df.shape[0]} filas.")

        log.info(f"Limpieza completa: {df.shape[0]} filas x {df.shape[1]} columnas.")
        return df
    except (ValueError, KeyError):
        raise
    except Exception as e:
        raise RuntimeError(f"Error inesperado en clean_data: {type(e).__name__}: {e}") from e


# ─── Feature Engineering ──────────────────────────────────────────────────────

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crea features derivadas a partir de los datos limpios.

    Features generadas:
        - Temporales: delivery_days, estimated_days, delay_days, is_late,
          purchase_month, purchase_dayofweek, purchase_hour, is_weekend.
        - Delivery avanzadas: delivery_ratio, days_early, delivery_urgency.
        - Precio: price_per_item, freight_ratio, freight_per_item, payment_premium.
        - Producto: volume_cm3, freight_per_kg.
        - Target: is_satisfied = (review_score > 2).

    Args:
        df: DataFrame limpio.

    Returns:
        DataFrame con todas las features creadas y timestamps eliminados.

    Raises:
        KeyError: Si faltan columnas requeridas para calcular las features.
    """
    try:
        ts_cols = ["order_purchase_timestamp", "order_delivered_customer_date",
                   "order_estimated_delivery_date"]
        for col in ts_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        df["delivery_days"] = (df["order_delivered_customer_date"] - df["order_purchase_timestamp"]).dt.days
        df["estimated_days"] = (df["order_estimated_delivery_date"] - df["order_purchase_timestamp"]).dt.days
        df["delay_days"] = df["delivery_days"] - df["estimated_days"]
        df["is_late"] = (df["delay_days"] > 0).astype(int)
        df["purchase_month"] = df["order_purchase_timestamp"].dt.month
        df["purchase_dayofweek"] = df["order_purchase_timestamp"].dt.dayofweek
        df["purchase_hour"] = df["order_purchase_timestamp"].dt.hour
        df["is_weekend"] = (df["purchase_dayofweek"] >= 5).astype(int)

        df["delivery_ratio"] = df["delivery_days"] / (df["estimated_days"].clip(lower=1))
        df["days_early"] = (-df["delay_days"]).clip(lower=0)
        df["delivery_urgency"] = 1 / (df["estimated_days"].clip(lower=1))

        df["price_per_item"] = df["total_price"] / df["n_items"].clip(lower=1)
        df["freight_ratio"] = df["total_freight"] / (df["total_price"] + df["total_freight"] + 1e-6)
        df["freight_per_item"] = df["total_freight"] / df["n_items"].clip(lower=1)
        df["payment_premium"] = df["total_payment"] - df["total_price"] - df["total_freight"]

        vol_cols = ["product_length_cm", "product_height_cm", "product_width_cm"]
        if all(c in df.columns for c in vol_cols):
            df["volume_cm3"] = df["product_length_cm"] * df["product_height_cm"] * df["product_width_cm"]
            df["freight_per_kg"] = df["total_freight"] / (df["product_weight_g"].clip(lower=1) / 1000)

        # Target: >2 separa reseñas malas (1-2 estrellas) de buenas (3-5)
        df["is_satisfied"] = (df["review_score"] > 2).astype(int)

        df = df.drop(columns=[c for c in ts_cols if c in df.columns])
        log.info(f"Features creadas: {df.shape[1]} columnas. Balance: "
                 f"{df['is_satisfied'].value_counts(normalize=True).round(3).to_dict()}")
        return df
    except KeyError as e:
        raise KeyError(f"Columna requerida no encontrada en create_features: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Error inesperado en create_features: {type(e).__name__}: {e}") from e


# ─── Codificación y Escalado ──────────────────────────────────────────────────

def encode_and_scale(df: pd.DataFrame) -> pd.DataFrame:
    """Codifica variables categóricas y escala variables numéricas continuas.

    Aplica LabelEncoder a columnas categóricas y StandardScaler a numéricas,
    excluyendo columnas binarias y targets para evitar data leakage.

    Args:
        df: DataFrame con features creadas.

    Returns:
        DataFrame listo para modelado con todas las columnas numéricas.

    Raises:
        ValueError: Si el DataFrame no contiene columnas numéricas válidas.
    """
    try:
        skip_scale = {"is_satisfied", "review_score", "is_late", "is_weekend",
                      "purchase_month", "purchase_dayofweek", "purchase_hour",
                      "order_status", "payment_type", "customer_state",
                      "seller_state", "product_category_name_english"}

        cat_cols = ["order_status", "payment_type", "customer_state",
                    "seller_state", "product_category_name_english"]
        for col in cat_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))

        num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in skip_scale]
        if not num_cols:
            raise ValueError("No se encontraron columnas numéricas para escalar.")
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])

        log.info(f"Dataset final: {df.shape[0]} filas x {df.shape[1]} columnas.")
        return df
    except (ValueError, KeyError):
        raise
    except Exception as e:
        raise RuntimeError(f"Error inesperado en encode_and_scale: {type(e).__name__}: {e}") from e
