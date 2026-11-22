"""Features computed only from prediction-time inputs."""

from sklearn.base import BaseEstimator, TransformerMixin


class DemandFeatureEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        required = {"hr", "workingday", "temp", "hum"}

        missing = required - set(X.columns)

        if missing:
            raise ValueError(
                f"Missing feature inputs: {sorted(missing)}"
            )

        result = X.copy()

        result["commuter_peak"] = (
            result["hr"].isin([7, 8, 9, 16, 17, 18])
            & (result["workingday"] == 1)
        ).astype(int)

        result["temperature_humidity"] = (
            result["temp"] * result["hum"]
        )

        return result
