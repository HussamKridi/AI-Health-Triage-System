import pandas as pd


BASE_COLUMNS = ["Oxygen Saturation", "Body Temperature", "Heart Rate"]


def build_feature_frame(data):
    if isinstance(data, pd.DataFrame):
        base = data[BASE_COLUMNS].copy()
    else:
        base = pd.DataFrame([data], columns=BASE_COLUMNS)

    features = base.copy()

    features["Spo2 Deficit From 97"] = (97.0 - features["Oxygen Saturation"]).clip(lower=0)
    features["Spo2 Deficit From 95"] = (95.0 - features["Oxygen Saturation"]).clip(lower=0)
    features["Temperature Elevation"] = (features["Body Temperature"] - 37.0).clip(lower=0)
    features["Heart Rate Elevation"] = (features["Heart Rate"] - 100).clip(lower=0)
    features["Heart Rate Depression"] = (60 - features["Heart Rate"]).clip(lower=0)
    features["Heart Rate Distance From 80"] = (features["Heart Rate"] - 80).abs()
    features["Temperature Distance From 36.8"] = (features["Body Temperature"] - 36.8).abs()

    features["Low Spo2 Flag"] = (features["Oxygen Saturation"] < 95).astype(int)
    features["Borderline Spo2 Flag"] = (features["Oxygen Saturation"] < 97).astype(int)
    features["Fever Flag"] = (features["Body Temperature"] >= 38.0).astype(int)
    features["Borderline Fever Flag"] = (features["Body Temperature"] >= 37.5).astype(int)
    features["Tachycardia Flag"] = (features["Heart Rate"] >= 100).astype(int)
    features["Bradycardia Flag"] = (features["Heart Rate"] <= 60).astype(int)

    features["Spo2 X Temperature"] = features["Oxygen Saturation"] * features["Body Temperature"]
    features["Spo2 X Heart Rate"] = features["Oxygen Saturation"] * features["Heart Rate"]
    features["Temperature X Heart Rate"] = features["Body Temperature"] * features["Heart Rate"]

    return features
