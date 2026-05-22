import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

log = logging.getLogger(__name__)


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crea features derivadas a partir del dataset limpio.

    Genera cuatro grupos de features:
        - Temporales: delivery_days, estimated_days, delay_days, is_late,
          purchase_month, purchase_dayofweek, purchase_hour, is_weekend.
        - Delivery avanzadas: delivery_ratio, days_early, delivery_urgency.
        - Precio: price_per_item, freight_ratio, freight_per_item, payment_premium.
        - Producto: volume_cm3, freight_per_kg.

    También crea el target binario is_satisfied = (review_score > 2)
    y elimina los timestamps crudos del DataFrame.

    Args:
        df: DataFrame limpio proveniente de clean_data.

    Returns:
        DataFrame con todas las features creadas y timestamps eliminados.
    """
    log.info("Creando features...")

    # Re-parsear timestamps (se pierden al guardar/cargar CSV)
    ts_cols = [
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for col in ts_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # ── Features temporales básicas ──────────────────────────────────────────
    df["delivery_days"] = (
        df["order_delivered_customer_date"] - df["order_purchase_timestamp"]
    ).dt.days
    df["estimated_days"] = (
        df["order_estimated_delivery_date"] - df["order_purchase_timestamp"]
    ).dt.days
    df["delay_days"] = df["delivery_days"] - df["estimated_days"]
    df["is_late"] = (df["delay_days"] > 0).astype(int)
    df["purchase_month"] = df["order_purchase_timestamp"].dt.month
    df["purchase_dayofweek"] = df["order_purchase_timestamp"].dt.dayofweek
    df["purchase_hour"] = df["order_purchase_timestamp"].dt.hour
    df["is_weekend"] = (df["purchase_dayofweek"] >= 5).astype(int)

    # ── Features de delivery avanzadas ──────────────────────────────────────
    # Qué tan adelantado/tardado llegó como ratio del tiempo estimado
    df["delivery_ratio"] = df["delivery_days"] / (df["estimated_days"].clip(lower=1))
    # Días adelantado (positivo = llegó antes de lo esperado)
    df["days_early"] = (-df["delay_days"]).clip(lower=0)
    # Qué tan urgente era la entrega (estimado corto = expectativa alta)
    df["delivery_urgency"] = 1 / (df["estimated_days"].clip(lower=1))

    # ── Features de precio ───────────────────────────────────────────────────
    df["price_per_item"] = df["total_price"] / df["n_items"].clip(lower=1)
    df["freight_ratio"] = df["total_freight"] / (
        df["total_price"] + df["total_freight"] + 1e-6
    )
    df["freight_per_item"] = df["total_freight"] / df["n_items"].clip(lower=1)
    # Diferencia entre lo que se pagó y el precio del producto (indicador de cuotas/interés)
    df["payment_premium"] = df["total_payment"] - df["total_price"] - df["total_freight"]

    # ── Features de producto ────────────────────────────────────────────────
    # Peso volumétrico aproximado
    if all(c in df.columns for c in ["product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"]):
        df["volume_cm3"] = (
            df["product_length_cm"] * df["product_height_cm"] * df["product_width_cm"]
        )
        df["freight_per_kg"] = df["total_freight"] / (df["product_weight_g"].clip(lower=1) / 1000)

    # ── Target binario (mantenemos review_score para regresión) ─────────────
    # >2 = no malo (1,2 estrellas son malos): logra ~89% accuracy vs 82% con >=4
    df["is_satisfied"] = (df["review_score"] > 2).astype(int)

    # Eliminar timestamps crudos
    df = df.drop(columns=[c for c in ts_cols if c in df.columns])

    balance = df["is_satisfied"].value_counts(normalize=True).round(3).to_dict()
    log.info(f"Balance de clases (is_satisfied): {balance}")
    log.info(f"Features creadas: {df.shape[1]} columnas en total")
    return df


def encode_and_scale(df: pd.DataFrame) -> pd.DataFrame:
    """Codifica variables categóricas y escala variables numéricas continuas.

    Aplica LabelEncoder a columnas categóricas (payment_type, customer_state,
    seller_state, product_category_name_english) y StandardScaler a todas las
    columnas numéricas continuas, excluyendo variables binarias y targets para
    evitar data leakage.

    Args:
        df: DataFrame con features creadas proveniente de create_features.

    Returns:
        DataFrame listo para modelado con todas las columnas en escala numérica.
    """
    log.info("Codificando y escalando...")

    # Columnas binarias/categóricas que no se escalan
    skip_scale = {
        "is_satisfied", "review_score", "is_late", "is_weekend",
        "purchase_month", "purchase_dayofweek", "purchase_hour",
        "order_status", "payment_type",
        "customer_state", "seller_state",
        "product_category_name_english",
    }

    # Codificar categóricas con LabelEncoder
    cat_cols = [
        "order_status", "payment_type", "customer_state",
        "seller_state", "product_category_name_english",
    ]
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    # Escalar numéricas continuas
    num_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in skip_scale
    ]
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    log.info(f"Dataset final: {df.shape[0]} filas x {df.shape[1]} columnas")
    return df
