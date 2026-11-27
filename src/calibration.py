"""Reusable probability calibration for the hotel's cancellation model.

Fit the base model on earlier bookings and the sigmoid calibrator on later
bookings. Neither component may see validation or test labels during fitting.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression


def logit_features(proba):
    p = np.clip(np.asarray(proba, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


@dataclass
class SigmoidCalibratedModel:
    base_model: object
    calibrator: LogisticRegression

    def predict_proba(self, X):
        raw = self.base_model.predict_proba(X)[:, 1]
        return self.calibrator.predict_proba(logit_features(raw))


def fit_calibrator(base_model, X_cal, y_cal):
    """Fit Platt sigmoid using held-out calibration rows only."""
    p = base_model.predict_proba(X_cal)[:, 1]
    calibrator = LogisticRegression(max_iter=1000)
    calibrator.fit(logit_features(p), y_cal)
    return SigmoidCalibratedModel(base_model, calibrator)
