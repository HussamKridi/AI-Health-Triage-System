import joblib
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from model_features import BASE_COLUMNS, build_feature_frame


DATASET_PATH = "vitals_data.csv"
MODEL_PATH = "triage_model.pkl"
RANDOM_STATE = 42


def main():
    print("Loading dataset...")
    df = pd.read_csv(DATASET_PATH)

    X = build_feature_frame(df[BASE_COLUMNS])
    y = df["Risk Category"]

    print("Splitting dataset with stratification...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("Training improved ExtraTrees model...")
    model = ExtraTreesClassifier(
        n_estimators=150,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Accuracy on test data: {accuracy * 100:.2f}%")
    print("\nClassification report:")
    print(classification_report(y_test, predictions))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nSaving model...")
    joblib.dump(model, MODEL_PATH)
    print(f"Complete! Saved improved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
