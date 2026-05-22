import logging
import pandas as pd

log = logging.getLogger(__name__)


def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Valida el dataset transformado antes de pasarlo al pipeline de modelado.

    Realiza las siguientes validaciones:
        - Elimina filas con nulos residuales y registra advertencia si existen.
        - Verifica que la columna target 'is_satisfied' esté presente y sea binaria.
        - Verifica que 'review_score' esté presente para la tarea de regresión.
        - Comprueba que el dataset tenga al menos 1000 filas y 8 columnas.

    Args:
        df: DataFrame transformado proveniente de encode_and_scale.

    Returns:
        DataFrame validado listo para entrenamiento.

    Raises:
        AssertionError: Si alguna validación crítica falla (target ausente,
            target no binario, dataset demasiado pequeño o con pocas columnas).
    """
    log.info("Validando dataset final...")

    # Sin nulos
    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if len(cols_with_nulls) > 0:
        log.warning(f"Columnas con nulos: {cols_with_nulls.to_dict()}")
        df = df.dropna()
        log.info(f"Nulos eliminados. Filas restantes: {df.shape[0]}")
    else:
        log.info("Sin nulos.")

    # Target presente y binario
    assert "is_satisfied" in df.columns, "Columna target 'is_satisfied' no encontrada"
    assert df["is_satisfied"].nunique() == 2, "Target no es binario"

    # review_score presente para regresión
    assert "review_score" in df.columns, "Columna 'review_score' no encontrada"

    # Tamaño mínimo
    assert df.shape[0] >= 1000, f"Dataset muy pequeño: {df.shape[0]} filas"
    assert df.shape[1] >= 8, f"Muy pocas columnas: {df.shape[1]}"

    balance = df["is_satisfied"].value_counts(normalize=True).round(3).to_dict()
    log.info(f"Balance final: {balance}")
    log.info(f"Validación OK: {df.shape[0]} filas x {df.shape[1]} columnas")
    return df
