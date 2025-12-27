import pytest
import pandas as pd
import numpy as np


def test_cleaning():
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
            "ca": ["0", "?"],  # will be converted to float and missing filled
            "thal": ["1", "?"],  # will be converted to float and missing filled
            "target": [1, 0],
        }
    )

    # -----------------------
    # Cleaning logic from train_model.py
    # -----------------------
    sample_data.replace("?", np.nan, inplace=True)
    for col in ["ca", "thal"]:
        sample_data[col] = sample_data[col].astype(float)
        sample_data[col].fillna(sample_data[col].median(), inplace=True)

    for col in sample_data.columns:
        if col not in ["ca", "thal"]:
            sample_data[col] = sample_data[col].astype(float)

    # Convert target to 0/1
    sample_data["target"] = sample_data["target"].apply(lambda x: 1 if x > 0 else 0)

    # -----------------------
    # Assertions
    # -----------------------
    assert sample_data.isnull().sum().sum() == _
