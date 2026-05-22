import logging
import pandas as pd

log = logging.getLogger(__name__)


def merge_datasets(
    orders: pd.DataFrame,
    customers: pd.DataFrame,
    order_items: pd.DataFrame,
    payments: pd.DataFrame,
    reviews: pd.DataFrame,
    products: pd.DataFrame,
    sellers: pd.DataFrame,
    category_translation: pd.DataFrame,
) -> pd.DataFrame:
    """Fusiona las 8 tablas del dataset Olist en un único DataFrame analítico.

    Agrega ítems y pagos por orden, toma la primera reseña cronológica,
    enriquece productos con la categoría en inglés y realiza el merge central
    usando order_id como clave principal.

    Args:
        orders: Órdenes con timestamps y estado.
        customers: Datos de clientes (estado, zip code).
        order_items: Ítems por orden (precio, flete, producto, vendedor).
        payments: Métodos y montos de pago por orden.
        reviews: Reseñas y puntajes de satisfacción.
        products: Dimensiones y peso de productos.
        sellers: Estado del vendedor.
        category_translation: Traducción de categorías PT → EN.

    Returns:
        DataFrame fusionado con todas las tablas, listo para limpieza.
    """
    log.info("Iniciando merge de datasets Olist...")

    # Agregar items por orden
    items_agg = (
        order_items.groupby("order_id")
        .agg(
            n_items=("order_item_id", "count"),
            total_price=("price", "sum"),
            total_freight=("freight_value", "sum"),
            product_id=("product_id", "first"),
            seller_id=("seller_id", "first"),
        )
        .reset_index()
    )

    # Agregar pagos por orden
    pay_agg = (
        payments.groupby("order_id")
        .agg(
            total_payment=("payment_value", "sum"),
            max_installments=("payment_installments", "max"),
            payment_type=("payment_type", "first"),
        )
        .reset_index()
    )

    # Una review por orden (la primera cronológicamente)
    reviews_sorted = reviews.sort_values("review_creation_date")
    reviews_first = (
        reviews_sorted.groupby("order_id")
        .first()
        .reset_index()
        [["order_id", "review_score"]]
    )

    # Productos con categoría en inglés
    products_with_cat = products.merge(category_translation, on="product_category_name", how="left")

    # Merge central
    df = (
        orders
        .merge(customers, on="customer_id", how="left")
        .merge(reviews_first, on="order_id", how="left")
        .merge(items_agg, on="order_id", how="left")
        .merge(pay_agg, on="order_id", how="left")
        .merge(products_with_cat, on="product_id", how="left")
        .merge(sellers, on="seller_id", how="left")
    )

    # Eliminar columnas con caracteres especiales no-ASCII que bloquean el guardado en Windows
    drop_early = [
        "customer_city", "seller_city",
        "product_category_name",        # se usa la versión en inglés
        "review_comment_message", "review_comment_title",
    ]
    df = df.drop(columns=[c for c in drop_early if c in df.columns])

    log.info(f"Dataset merged: {df.shape[0]} filas x {df.shape[1]} columnas")
    log.info(f"Columnas: {list(df.columns)}")
    return df
