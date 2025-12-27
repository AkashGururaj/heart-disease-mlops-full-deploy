import os
import pickle
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

import mlflow
import mlflow.sklearn

# -----------------------
# Setup
# -----------------------
os.makedirs("outputs", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# -----------------------
# Load dataset
# -----------------------
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
columns = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","target"]
df = pd.read_csv(url, header=None, names=columns)

# -----------------------
# Clean dataset
# -----------------------
df.replace("?", np.nan, inplace=True)
for col in ["ca","thal"]:
    df[col] = df[col].astype(float)
    df[col].fillna(df[col].median(), inplace=True)
for col in df.columns:
    if col not in ["ca","thal"]:
        df[col] = df[col].astype(float)
df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)

# -----------------------
# EDA
# -----------------------
plt.figure(figsize=(8,6))
sns.countplot(x="target", data=df)
plt.savefig(f"outputs/class_balance_{timestamp}.png")
plt.close()

plt.figure(figsize=(12,10))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.savefig(f"outputs/corr_heatmap_{timestamp}.png")
plt.close()

for col in df.columns[:-1]:
    plt.figure()
    sns.histplot(df[col], kde=True)
    plt.savefig(f"outputs/hist_{col}_{timestamp}.png")
    plt.close()

# -----------------------
# Train-test split
# -----------------------
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------
# Feature engineering
# -----------------------
num_features = ["age","trestbps","chol","thalach","oldpeak"]
cat_features = [col for col in X.columns if col not in num_features]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ]
)

# -----------------------
# MLflow experiment tracking
# -----------------------
mlflow.set_experiment("Heart_Disease_Prediction")

models = {
    "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000)
}

results = {}

for name, clf in models.items():
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:,1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)
    
    results[name] = {"Accuracy": acc, "Precision": prec, "Recall": rec, "ROC-AUC": roc}
    
    # Log to MLflow
    with mlflow.start_run(run_name=name):
        mlflow.log_params(clf.get_params())
        mlflow.log_metrics({"Accuracy": acc, "Precision": prec, "Recall": rec, "ROC-AUC": roc})
        mlflow.sklearn.log_model(pipeline, f"{name}_model")

# -----------------------
# Create performance chart
# -----------------------
metrics = ["Accuracy", "Precision", "Recall", "ROC-AUC"]
model_names = list(models.keys())

fig, ax = plt.subplots(figsize=(10,6))
x = np.arange(len(model_names))
width = 0.2

for i, metric in enumerate(metrics):
    values = [results[m][metric] for m in model_names]
    bars = ax.bar(x + i*width, values, width=width, label=metric)
    
    # Annotate values on top
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0,3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

ax.set_xticks(x + width*1.5)
ax.set_xticklabels(model_names)
ax.set_ylim(0,1.1)
ax.set_ylabel("Score")
ax.set_title("Model Performance Metrics on Test Set")
ax.legend()
plt.tight_layout()

results_img_path = f"outputs/model_performance_{timestamp}.png"
plt.savefig(results_img_path)
plt.close()

# -----------------------
# Save the best model
# -----------------------
# Choose best model by Accuracy
best_model_name = max(results, key=lambda k: results[k]["Accuracy"])
best_model_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", models[best_model_name])])
best_model_pipeline.fit(X_train, y_train)

pickle_path = f"outputs/final_model_{timestamp}.pkl"
with open(pickle_path, "wb") as f:
    pickle.dump(best_model_pipeline, f)

preproc_path = f"outputs/preprocessing_pipeline_{timestamp}.pkl"
with open(preproc_path, "wb") as f:
    pickle.dump(best_model_pipeline.named_steps["preprocessor"], f)

# -----------------------
# Save requirements
# -----------------------
requirements = "numpy\npandas\nscikit-learn\nmatplotlib\nseaborn\nmlflow\npytest\nflake8"
with open("outputs/requirements.txt", "w") as f:
    f.write(requirements)

# -----------------------
# Done
# -----------------------
print(f"Training Complete. Outputs saved with timestamp: {timestamp}")
print(f"Best Model Saved: {best_model_name}")
print(f"performance Chart Saved at: {results_img_path}")
