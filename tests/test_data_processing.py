import pytest
import pandas as pd
import numpy as np


def test_cleaning():
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

    sample_data.replace("?", np.nan, inplace=True)
    for col in ["ca", "thal"]:
        sample_data[col] = sample_data[col].astype(float)
        sample_data[col].fillna(sample_data[col].median(), inplace=True)
    for col in sample_data.columns:
        if col not in ["ca", "thal"]:
            sample_data[col] = sample_data[col].astype(float)
    sample_data["target"] = sample_data["target"].apply(lambda val: 1 if val > 0 else 0)

    assert sample_data.isnull().sum().sum() == 0
    assert set(sample_data["target"].unique()).issubset({0, 1})
    assert sample_data.shape[0] == 2
    assert sample_data.shape[1] == 14
