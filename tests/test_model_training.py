import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from model_training import preprocessor  # import your preprocessor from train_model.py if available


def test_data_cleaning():
    """Test that sample data can be cleaned correctly."""
    sample_data = pd.DataFrame(
        {
            "age": [63, 37],
            "sex": [1, 1],
            "cp": [1, 2],
            "trestbps": [145, 130],
            "chol": [233, 250],
            "fbs": [1, 0],
            "restecg": [0, 1],
            "thalach": [150, 187],
            "exang": [0, 0],
            "oldpeak": [2.3, 3.5],
            "slope": [0, 2],
            "ca": ["0", "?"],
            "thal": ["1", "?"],
            "target": [1, 0],
        }
    )

    # Replace "?" with median values similar to train_model.py
    sample_data.replace("?", np.nan, inplace=True)
    for col in ["ca", "thal"]:
        sample_data[col] = sample_data[col].astype(float)
        sample_data[col].fillna(sample_data[col].median(), inplace=True)

    for col in sample_data.columns:
        if col not in ["ca", "thal"]:
            sample_data[col] = sample_data[col].astype(float)

    # Ensure target column is 0/1
    sample_data["target"] = sample_data["target"].apply(lambda x: 1 if x > 0 else 0)

    # Assertions
    assert sample_data.isnull().sum().sum() == 0
    assert set(sample_data["target"].unique()).issubset({0, 1})
    assert sample_data.shape[0] == 2


def test_pipeline_creation():
    """Test that a pipeline can be created and trained on sample data."""
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

    # Minimal sample training data
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

    # Fit pipeline
    pipeline.fit(X_train, y_train)

    # Assertions
    assert isinstance(pipeline, Pipeline)
    assert "preprocessor" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps
    assert pipeline.named_steps["classifier"].n_estimators == 200
