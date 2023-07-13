"""
This is a boilerplate pipeline 'data_prep'
generated using Kedro 0.18.11
"""
from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    create_model_input_table,
    get_data,
    preprocess_companies,
    preprocess_shuttles,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=get_data,
                inputs=None,
                outputs=["companies_raw", "shuttles_raw", "review_raw"],
                name="get_data",
            ),
            node(
                func=preprocess_companies,
                inputs="companies_raw",
                outputs="companies_preprocessed",
                name="preprocess_companies",
            ),
            node(
                func=preprocess_shuttles,
                inputs="shuttles_raw",
                outputs="shuttles_preprocessed",
                name="preprocess_shuttles",
            ),
            node(
                func=create_model_input_table,
                inputs=dict(
                    shuttles="shuttles_preprocessed",
                    companies="companies_preprocessed",
                    reviews="review_raw",
                ),
                outputs="model_input_table",
                name="create_model_input_table",
            ),
        ]
    )
