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

# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI(title="Heart Disease Prediction API")

# ----------------------------
# Templates folder
# ----------------------------
templates = Jinja2Templates(directory="src/templates")

# ----------------------------
# Setup output folder & logging
# ----------------------------
os.makedirs("src/outputs", exist_ok=True)
logging.basicConfig(filename="src/outputs/api_requests.log", level=logging.INFO, format="%(message)s")

# ----------------------------
# Load latest trained model
# ----------------------------
model_files = [f for f in os.listdir("src/outputs") if f.startswith("final_model_") and f.endswith(".pkl")]
if not model_files:
    raise FileNotFoundError("No trained model found in src/outputs folder.")
latest_model_file = sorted(model_files)[-1]

with open(f"src/outputs/{latest_model_file}", "rb") as f:
    model = pickle.load(f)

# ----------------------------
# Features
# ----------------------------
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

# ----------------------------
# Prometheus metrics
# ----------------------------
instrumentator = Instrumentator(should_group_status_codes=False, should_ignore_untemplated=True)
instrumentator.instrument(app).expose(app)  # /metrics endpoint


# ----------------------------
# Middleware to log all requests
# ----------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = round(time.time() - start_time, 3)

    # Log basic request info
    log_entry = {
        "method": request.method,
        "url": str(request.url),
        "status_code": response.status_code,
        "process_time": process_time,
    }

    logging.info(json.dumps(log_entry))
    return response


# ----------------------------
# Home page - form
# ----------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "features": FEATURES})


# ----------------------------
# Prediction endpoint
# ----------------------------
@app.post("/predict_form", response_class=HTMLResponse)
async def predict_form(request: Request):
    form_data = await request.form()
    input_data = {key: float(value) for key, value in form_data.items() if key in FEATURES}

    df = pd.DataFrame([input_data])
    pred_class = int(model.predict(df)[0])
    pred_proba = float(model.predict_proba(df)[0][1])
    pred_label = "Heart Disease Likely" if pred_class == 1 else "No Heart Disease"

    # Log prediction immediately
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


# ----------------------------
# Logs page
# ----------------------------
@app.get("/logs", response_class=HTMLResponse)
def view_logs(request: Request):
    logs_file = "src/outputs/api_requests.log"
    logs_data = []

    if os.path.exists(logs_file):
        with open(logs_file, "r") as f:
            for line in deque(f, maxlen=50):
                try:
                    log_entry = json.loads(line)
                    logs_data.append(
                        {
                            "method": log_entry.get("method"),
                            "url": log_entry.get("url"),
                            "status_code": log_entry.get("status_code"),
                            "process_time": (
                                round(log_entry.get("process_time", 0), 3)
                                if isinstance(log_entry.get("process_time", 0), (int, float))
                                else "-"
                            ),
                            "prediction": log_entry.get("prediction", "-"),
                            "confidence": (
                                round(float(log_entry["confidence"]), 2)
                                if "confidence" in log_entry and isinstance(log_entry["confidence"], (int, float, str))
                                else "-"
                            ),
                        }
                    )
                except (json.JSONDecodeError, ValueError):
                    continue

    return templates.TemplateResponse("logs.html", {"request": request, "logs": logs_data})
