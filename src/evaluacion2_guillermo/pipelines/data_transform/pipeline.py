from kedro.pipeline import Pipeline, node, pipeline
from .nodes import create_features, encode_and_scale


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=create_features,
            inputs="dataset_clean",
            outputs="dataset_features",
            name="node_create_features",
        ),
        node(
            func=encode_and_scale,
            inputs="dataset_features",
            outputs="dataset_final",
            name="node_encode_and_scale",
        ),
    ])
