from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    apply_clustering,
    apply_pca,
    compare_optimization,
    optimize_gridsearch,
    optimize_randomsearch,
    train_classifiers,
    train_regressors,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=train_classifiers,
            inputs="dataset_validated",
            outputs="metricas_clasificacion",
            name="node_train_classifiers",
        ),
        node(
            func=train_regressors,
            inputs="dataset_validated",
            outputs="metricas_regresion",
            name="node_train_regressors",
        ),
        node(
            func=apply_clustering,
            inputs="dataset_validated",
            outputs="metricas_clustering",
            name="node_apply_clustering",
        ),
        node(
            func=apply_pca,
            inputs="dataset_validated",
            outputs="metricas_pca",
            name="node_apply_pca",
        ),
        node(
            func=optimize_gridsearch,
            inputs="dataset_validated",
            outputs="resultados_gridsearch",
            name="node_gridsearch",
        ),
        node(
            func=optimize_randomsearch,
            inputs="dataset_validated",
            outputs="resultados_randomsearch",
            name="node_randomsearch",
        ),
        node(
            func=compare_optimization,
            inputs=["resultados_gridsearch", "resultados_randomsearch"],
            outputs="comparacion_optimizacion",
            name="node_compare_optimization",
        ),
    ])
