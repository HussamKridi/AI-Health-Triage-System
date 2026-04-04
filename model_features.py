import numpy as np
import pandas as pd


MODEL_INPUT_COLUMNS = [
    "Age",
    "Gender",
    "Weight (kg)",
    "Height (m)",
    "Oxygen Saturation",
    "Body Temperature",
    "Heart Rate",
]

INPUT_ALIASES = {
    "age": "Age",
    "gender": "Gender",
    "weight": "Weight (kg)",
    "height": "Height (m)",
    "spo2": "Oxygen Saturation",
    "temperature": "Body Temperature",
    "heart_rate": "Heart Rate",
}


def _normalize_columns(dataframe):
    renamed = dataframe.copy()
    rename_map = {}
    for column in renamed.columns:
        canonical = INPUT_ALIASES.get(column, column)
        rename_map[column] = canonical
    renamed = renamed.rename(columns=rename_map)
    missing = [column for column in MODEL_INPUT_COLUMNS if column not in renamed.columns]
    if missing:
        raise KeyError(f"Missing model input columns: {missing}")
    return renamed[MODEL_INPUT_COLUMNS].copy()


def build_feature_frame(data):
    if isinstance(data, pd.DataFrame):
        base = _normalize_columns(data)
    else:
        base = _normalize_columns(pd.DataFrame([data]))

    features = base.copy()
    features["Gender"] = features["Gender"].astype(str).str.strip().str.title()

    height = pd.to_numeric(features["Height (m)"], errors="coerce")
    weight = pd.to_numeric(features["Weight (kg)"], errors="coerce")
    safe_height = height.where(height > 0)
    bmi = weight / np.square(safe_height)

    features["Age"] = pd.to_numeric(features["Age"], errors="coerce")
    features["Weight (kg)"] = weight
    features["Height (m)"] = height
    features["Oxygen Saturation"] = pd.to_numeric(features["Oxygen Saturation"], errors="coerce")
    features["Body Temperature"] = pd.to_numeric(features["Body Temperature"], errors="coerce")
    features["Heart Rate"] = pd.to_numeric(features["Heart Rate"], errors="coerce")
    features["BMI"] = bmi
    features["Age X BMI"] = features["Age"] * features["BMI"]
    features["Spo2 Deficit From 97"] = (97.0 - features["Oxygen Saturation"]).clip(lower=0)
    features["Temperature Elevation"] = (features["Body Temperature"] - 37.0).clip(lower=0)
    features["Heart Rate Elevation"] = (features["Heart Rate"] - 100).clip(lower=0)
    features["Heart Rate Depression"] = (60 - features["Heart Rate"]).clip(lower=0)
    features["Low Spo2 Flag"] = (features["Oxygen Saturation"] < 95).astype(int)
    features["Fever Flag"] = (features["Body Temperature"] >= 38.0).astype(int)
    features["Tachycardia Flag"] = (features["Heart Rate"] >= 100).astype(int)
    features["Senior Flag"] = (features["Age"] >= 65).astype(int)
    features["Underweight Flag"] = (features["BMI"] < 18.5).astype(int)
    features["Obesity Flag"] = (features["BMI"] >= 30).astype(int)
    return features
