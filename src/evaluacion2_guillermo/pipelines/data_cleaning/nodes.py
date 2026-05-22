import logging
import pandas as pd
import numpy as np

log = logging.getLogger(__name__)


def _cap_iqr(df: pd.DataFrame, col: str, factor: float = 1.5) -> pd.DataFrame:
    """Winsoriza una columna: reemplaza outliers con los límites IQR en vez de eliminarlos."""
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    antes = ((df[col] < lower) | (df[col] > upper)).sum()
    df[col] = df[col].clip(lower=lower, upper=upper)
    log.info(f"  Outliers {col}: {antes} valores capados al rango [{lower:.2f}, {upper:.2f}]")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia el dataset fusionado aplicando 9 pasos secuenciales.

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
        df: DataFrame fusionado resultante de merge_datasets.

    Returns:
        DataFrame limpio con solo órdenes válidas y columnas relevantes.
    """
    log.info(f"=== LIMPIEZA DE DATOS ===")
    log.info(f"Filas iniciales: {df.shape[0]}")

    # ── 1. Filtrar solo órdenes entregadas ──────────────────────────────────
    df = df[df["order_status"] == "delivered"].copy()
    log.info(f"[1] Tras filtro 'delivered': {df.shape[0]} filas")

    # ── 2. Eliminar filas sin target (review_score) ──────────────────────────
    n_antes = df.shape[0]
    df = df.dropna(subset=["review_score"])
    log.info(f"[2] Eliminadas {n_antes - df.shape[0]} filas sin review_score. Quedan: {df.shape[0]}")

    # ── 3. Parsear y validar timestamps ─────────────────────────────────────
    ts_cols = [
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for col in ts_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    n_antes = df.shape[0]
    df = df.dropna(subset=ts_cols)
    log.info(f"[3] Eliminadas {n_antes - df.shape[0]} filas con timestamps nulos. Quedan: {df.shape[0]}")

    # ── 4. Eliminar entregas con tiempos imposibles ──────────────────────────
    # delivery_days negativos o > 180 días son datos erróneos
    delivery_days = (df["order_delivered_customer_date"] - df["order_purchase_timestamp"]).dt.days
    mask_valida = (delivery_days >= 0) & (delivery_days <= 180)
    n_antes = df.shape[0]
    df = df[mask_valida]
    log.info(f"[4] Eliminadas {n_antes - df.shape[0]} filas con delivery_days fuera de [0, 180]. Quedan: {df.shape[0]}")

    # ── 5. Rellenar nulos de dimensiones de producto con mediana ────────────
    dim_cols = [
        "product_weight_g", "product_length_cm",
        "product_height_cm", "product_width_cm", "product_photos_qty",
    ]
    for col in dim_cols:
        if col in df.columns:
            nulos = df[col].isnull().sum()
            if nulos > 0:
                df[col] = df[col].fillna(df[col].median())
                log.info(f"[5] {col}: {nulos} nulos rellenados con mediana ({df[col].median():.2f})")

    # ── 6. Rellenar nulos de pago ───────────────────────────────────────────
    df["total_payment"] = df["total_payment"].fillna(df["total_price"])
    df["max_installments"] = df["max_installments"].fillna(1)
    df["payment_type"] = df["payment_type"].fillna("unknown")
    df["product_category_name_english"] = df["product_category_name_english"].fillna("unknown")
    df["customer_state"] = df["customer_state"].fillna("unknown")
    df["seller_state"] = df["seller_state"].fillna("unknown")
    log.info(f"[6] Nulos de pago y categoría rellenados.")

    # ── 7. Tratar outliers con Winsorización IQR ────────────────────────────
    log.info("[7] Tratamiento de outliers (Winsorización IQR 1.5):")
    cols_outlier = [
        "total_price", "total_freight", "total_payment",
        "product_weight_g", "product_length_cm",
        "product_height_cm", "product_width_cm",
        "product_photos_qty", "max_installments",
    ]
    for col in cols_outlier:
        if col in df.columns:
            df = _cap_iqr(df, col)

    # n_items: outlier extremo (21 items), cap a 99th percentile
    if "n_items" in df.columns:
        p99 = df["n_items"].quantile(0.99)
        antes = (df["n_items"] > p99).sum()
        df["n_items"] = df["n_items"].clip(upper=p99)
        log.info(f"  Outliers n_items: {antes} valores capados al p99 ({p99})")

    # Eliminar precios cero o negativos (imposibles)
    n_antes = df.shape[0]
    df = df[df["total_price"] > 0]
    log.info(f"[7b] Eliminadas {n_antes - df.shape[0]} filas con precio <= 0.")

    # ── 8. Eliminar duplicados ───────────────────────────────────────────────
    n_antes = df.shape[0]
    df = df.drop_duplicates()
    log.info(f"[8] Eliminados {n_antes - df.shape[0]} duplicados. Quedan: {df.shape[0]}")

    # ── 9. Eliminar columnas no útiles para modelado ─────────────────────────
    cols_to_drop = [
        "order_id", "customer_id", "customer_unique_id", "product_id",
        "seller_id", "order_approved_at", "order_delivered_carrier_date",
        "customer_zip_code_prefix", "seller_zip_code_prefix",
        "product_name_lenght", "product_description_lenght",
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    log.info(f"[9] Columnas irrelevantes eliminadas.")

    log.info(f"=== LIMPIEZA COMPLETA: {df.shape[0]} filas x {df.shape[1]} columnas ===")
    return df
