from kedro.pipeline import Pipeline, node, pipeline
from .nodes import clean_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=clean_data,
            inputs="dataset_merged",
            outputs="dataset_clean",
            name="node_clean_data",
        ),
    ])
