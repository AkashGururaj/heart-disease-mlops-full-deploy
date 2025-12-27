import pytest
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier


def test_pipeline_creation():
    num_features = ["age", "trestbps", "chol"]
    cat_features = ["sex", "cp"]

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                ColumnTransformer(
                    transformers=[
                        ("num", StandardScaler(), num_features),
                        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
                    ]
                ),
            ),
            ("classifier", RandomForestClassifier(n_estimators=200, random_state=42)),
        ]
    )

    X_train = pd.DataFrame(
        {
            "age": [63, 37, 45],
            "trestbps": [145, 130, 120],
            "chol": [233, 250, 210],
            "sex": [1, 1, 0],
            "cp": [1, 2, 1],
        }
    )
    y_train = pd.Series([1, 0, 0])

    pipeline.fit(X_train, y_train)

    assert isinstance(pipeline, Pipeline)
    assert "preprocessor" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps
    assert pipeline.named_steps["classifier"].n_estimators == 200
