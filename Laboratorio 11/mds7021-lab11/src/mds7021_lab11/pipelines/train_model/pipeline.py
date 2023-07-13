"""
This is a boilerplate pipeline 'train_model'
generated using Kedro 0.18.11
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluate_model, split_data, train_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs=["model_input_table", "params:split_params"],
                outputs=[
                    "X_train",
                    "X_valid",
                    "X_test",
                    "y_train",
                    "y_valid",
                    "y_test",
                ],
                name="split_data",
            ),
            node(
                func=train_model,
                inputs=dict(
                    X_train="X_train",
                    y_train="y_train",
                    X_valid="X_valid",
                    y_valid="y_valid",
                ),
                outputs="best_model",
                name="train_model",
            ),
            node(
                func=evaluate_model,
                inputs=dict(model="best_model", X_test="X_test", y_test="y_test"),
                outputs=None,
                name="evaluate_model",
            ),
        ]
    )
