from kedro.pipeline import Pipeline, node, pipeline
from .nodes import merge_datasets


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=merge_datasets,
            inputs=[
                "olist_orders",
                "olist_customers",
                "olist_order_items",
                "olist_payments",
                "olist_reviews",
                "olist_products",
                "olist_sellers",
                "category_translation",
            ],
            outputs="dataset_merged",
            name="node_merge_datasets",
        ),
    ])
