import lightgbm as lgb
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
import numpy as np


# Function to calculate the average rides over the last 4 weeks
def average_rides_last_4_weeks(X: pd.DataFrame) -> pd.DataFrame:
    last_4_weeks_columns = [
        f"rides_t-{7*24}",  # 1 week ago
        f"rides_t-{14*24}",  # 2 weeks ago
        f"rides_t-{21*24}",  # 3 weeks ago
        f"rides_t-{28*24}",  # 4 weeks ago
    ]

    # Ensure the required columns exist in the DataFrame
    for col in last_4_weeks_columns:
        if col not in X.columns:
            raise ValueError(f"Missing required column: {col}")

    # Calculate the average of the last 4 weeks
    X["average_rides_last_4_weeks"] = X[last_4_weeks_columns].mean(axis=1)

    return X


# FunctionTransformer to add the average rides feature
add_feature_average_rides_last_4_weeks = FunctionTransformer(
    average_rides_last_4_weeks, validate=False
)


# Custom transformer to add temporal features
class TemporalFeatureEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        X_["hour"] = X_["pickup_hour"].dt.hour
        X_["day_of_week"] = X_["pickup_hour"].dt.dayofweek

        return X_.drop(columns=["pickup_hour", "pickup_location_id"])


# Instantiate the temporal feature engineer
add_temporal_features = TemporalFeatureEngineer()


# Function to return the pipeline
def get_pipeline(**hyper_params):
    pipeline = make_pipeline(
        add_feature_average_rides_last_4_weeks,
        add_temporal_features,
        FFTFeatureEngineer(window_size=24, top_k=5, drop_original=False),
        lgb.LGBMRegressor(**hyper_params)
    )
    return pipeline



from sklearn.base import BaseEstimator, TransformerMixin

# ---- Our new FFT transformer ----
class FFTFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, window_size=24, top_k=5, drop_original=False):
        self.window_size = window_size
        self.top_k = top_k
        self.drop_original = drop_original
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_ = X.copy()
        required_cols = [f"rides_t-{i}" for i in range(1, self.window_size + 1)]
        missing = [c for c in required_cols if c not in X_.columns]
        if missing:
            raise ValueError(f"Missing columns for FFT: {missing}")
        
        def compute_fft_features(row):
            signal = row[required_cols].values
            fft_vals = np.fft.rfft(signal)
            fft_mags = np.abs(fft_vals)
            top_indices = np.argsort(fft_mags)[-self.top_k:]
            return fft_mags[top_indices]
        
        fft_features = X_[required_cols].apply(
            compute_fft_features, axis=1, result_type="expand"
        )
        fft_features.columns = [f"fft_feature_{i}" for i in range(self.top_k)]
        
        X_ = pd.concat([X_, fft_features], axis=1)
        if self.drop_original:
            X_.drop(columns=required_cols, inplace=True)
        return X_