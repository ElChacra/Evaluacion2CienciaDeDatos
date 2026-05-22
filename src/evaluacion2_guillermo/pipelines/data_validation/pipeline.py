from kedro.pipeline import Pipeline, node, pipeline
from .nodes import validate_dataset


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=validate_dataset,
            inputs="dataset_final",
            outputs="dataset_validated",
            name="node_validate_dataset",
        ),
    ])
