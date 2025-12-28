from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pickle
import pandas as pd
import os
import logging
import json
from prometheus_fastapi_instrumentator import Instrumentator
import time
from collections import deque


app = FastAPI(title="Heart Disease Prediction API")

BASE_DIR = os.path.abspath(os.path.dirname(__file__))  # src/
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


log_file = os.path.join(OUTPUT_DIR, "api_requests.log")
logging.basicConfig(filename=log_file, level=logging.INFO, format="%(message)s")

# Load Latest Model
model_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith("final_model_") and f.endswith(".pkl")]

if not model_files:
    raise FileNotFoundError(f"No trained model found in {OUTPUT_DIR}")

latest_model_file = sorted(model_files)[-1]
model_path = os.path.join(OUTPUT_DIR, latest_model_file)
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Features
FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]


templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Metric Logging
instrumentator = Instrumentator(should_group_status_codes=False, should_ignore_untemplated=True)
instrumentator.instrument(app).expose(app)  # /metrics endpoint


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = round(time.time() - start_time, 3)

    log_entry = {
        "method": request.method,
        "url": str(request.url),
        "status_code": response.status_code,
        "process_time": process_time,
    }
    logging.info(json.dumps(log_entry))
    return response


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "features": FEATURES})


# Prediction API
@app.post("/predict_form", response_class=HTMLResponse)
async def predict_form(request: Request):
    form_data = await request.form()
    input_data = {k: float(v) for k, v in form_data.items() if k in FEATURES}

    df = pd.DataFrame([input_data])
    pred_class = int(model.predict(df)[0])
    pred_proba = float(model.predict_proba(df)[0][1])
    pred_label = "Heart Disease Likely" if pred_class == 1 else "No Heart Disease"

    # Log prediction
    log_entry = {
        "method": request.method,
        "url": str(request.url),
        "status_code": 200,
        "process_time": 0,
        "input": input_data,
        "prediction": pred_label,
        "confidence": pred_proba,
    }
    logging.info(json.dumps(log_entry))

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "features": FEATURES,
            "result": {"prediction": pred_label, "confidence": round(pred_proba, 2)},
        },
    )


# Log Page API
@app.get("/logs", response_class=HTMLResponse)
def view_logs(request: Request):
    logs_data = []
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            for line in deque(f, maxlen=50):
                try:
                    log_entry = json.loads(line)
                    logs_data.append(
                        {
                            "method": log_entry.get("method"),
                            "url": log_entry.get("url"),
                            "status_code": log_entry.get("status_code"),
                            "process_time": round(log_entry.get("process_time", 0), 3),
                            "prediction": log_entry.get("prediction", "-"),
                            "confidence": (
                                round(float(log_entry.get("confidence", 0)), 2) if "confidence" in log_entry else "-"
                            ),
                        }
                    )
                except (json.JSONDecodeError, ValueError):
                    continue

    return templates.TemplateResponse("logs.html", {"request": request, "logs": logs_data})
