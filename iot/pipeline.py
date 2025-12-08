"""
Real-Time IoT Analytics Pipeline
Author: Prabhash S

Anomaly detection, predictive maintenance, and next-best-action
decisioning for industrial IoT sensor data streams.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score


class IoTPipeline:
    def __init__(self):
        self.data      = None
        self.scaler    = StandardScaler()
        self.model     = None
        self.anomalies = None

    def generate_sample_data(self, n: int = 1000):
        np.random.seed(42)
        timestamps   = pd.date_range("2025-01-01", periods=n, freq="1min")
        values       = np.sin(np.linspace(0, 20, n)) * 50 + 100
        anomaly_idx  = np.random.choice(n, size=20, replace=False)
        values[anomaly_idx] += np.random.uniform(80, 150, size=20)
        self.data = pd.DataFrame({
            "timestamp" : timestamps,
            "sensor_id" : "S001",
            "value"     : values,
            "label"     : [1 if i in anomaly_idx else 0 for i in range(n)],
        })
        print(f"Generated {n} sensor readings with {len(anomaly_idx)} anomalies.")
        return self

    def detect_anomalies(self, method: str = "zscore", threshold: float = 3.0):
        values = self.data["value"].values
        if method == "zscore":
            flags = np.abs((values - values.mean()) / values.std()) > threshold
        elif method == "iqr":
            q1, q3 = np.percentile(values, [25, 75])
            iqr    = q3 - q1
            flags  = (values < q1 - threshold * iqr) | (values > q3 + threshold * iqr)
        elif method == "rolling_mean":
            series  = pd.Series(values)
            rolling = series.rolling(window=20, center=True).mean().ffill().bfill()
            resid   = np.abs(series - rolling)
            flags   = resid > threshold * resid.std()
        else:
            raise ValueError(f"Unknown method: {method}")

        self.data["anomaly"] = flags.astype(int)
        self.anomalies       = self.data[self.data["anomaly"] == 1]
        print(f"Detected {flags.sum()} anomalies using {method}.")
        return self.anomalies

    def _engineer_features(self):
        df = self.data.copy()
        df["rolling_mean_5"] = df["value"].rolling(5).mean().fillna(df["value"].mean())
        df["rolling_std_5"]  = df["value"].rolling(5).std().fillna(0)
        df["rolling_max_5"]  = df["value"].rolling(5).max().fillna(df["value"].max())
        df["diff_1"]         = df["value"].diff(1).fillna(0)
        df["diff_2"]         = df["value"].diff(2).fillna(0)
        return df

    def predict_maintenance(self, model: str = "xgboost"):
        df       = self._engineer_features()
        features = ["value","rolling_mean_5","rolling_std_5",
                    "rolling_max_5","diff_1","diff_2"]
        X = df[features].values
        y = df["label"].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        X_train = self.scaler.fit_transform(X_train)
        X_test  = self.scaler.transform(X_test)

        self.model = (XGBClassifier(n_estimators=100, max_depth=4,
                                    learning_rate=0.1, random_state=42,
                                    eval_metric="logloss")
                      if model == "xgboost"
                      else RandomForestClassifier(n_estimators=100, random_state=42))

        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)
        f1    = f1_score(y_test, preds)
        print(f"Model: {model} | F1-Score: {f1:.4f}")
        return {"model": model, "f1_score": f1,
                "report": classification_report(y_test, preds, output_dict=True)}

    def recommend_actions(self, maintenance_results: dict) -> list:
        actions     = []
        n_anomalies = len(self.anomalies) if self.anomalies is not None else 0
        f1          = maintenance_results.get("f1_score", 0)

        if n_anomalies > 50:
            actions.append("🔴 CRITICAL: Schedule immediate inspection.")
        elif n_anomalies > 20:
            actions.append("🟡 WARNING: Schedule maintenance within 48 hours.")
        else:
            actions.append("🟢 NORMAL: Continue standard monitoring.")

        actions.append("✅ High model confidence — trust predictions."
                       if f1 > 0.85 else
                       "⚠️ Low confidence — recommend human review.")
        for a in actions:
            print(a)
        return actions


if __name__ == "__main__":
    pipeline = IoTPipeline()
    pipeline.generate_sample_data(500)
    pipeline.detect_anomalies(method="zscore")
    results = pipeline.predict_maintenance(model="xgboost")
    pipeline.recommend_actions(results)
