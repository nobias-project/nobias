from nobias import ExplanationAudit

from xgboost import XGBClassifier, XGBRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.datasets import make_blobs
import pandas as pd
import numpy as np
from nobias import ExplanationAudit
import random

random.seed(0)


X, y = make_blobs(n_samples=2000, centers=2, n_features=5, random_state=0)
X = pd.DataFrame(X, columns=["a", "b", "c", "d", "e"])
# Protected att
X["a"] = np.where(X["a"] > X["a"].mean(), 1, 0)

# Train Val Holdout Split
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=0)
X_hold, X_te, y_hold, y_te = train_test_split(X_te, y_te, test_size=0.5, random_state=0)

a_tr = X_tr["a"]
a_te = X_te["a"]
a_hold = X_hold["a"]
X_tr = X_tr.drop("a", axis=1)
X_te = X_te.drop("a", axis=1)
X_hold = X_hold.drop("a", axis=1)
# Random
a_tr_ = np.random.randint(0, 2, size=X_tr.shape[0])
a_te_ = np.random.randint(0, 2, size=X_te.shape[0])
a_hold_ = np.random.randint(0, 2, size=X_hold.shape[0])


def test_explanation_audit():

    # Option 1: fit the detector when there is a trained model
    model = XGBClassifier().fit(X_tr, y_tr)

    detector = ExplanationAudit(model=model, gmodel=LogisticRegression())

    detector.fit_inspector(X_te, a_te)
    # pdb.set_trace()
    assert (
        np.round(
            roc_auc_score(a_hold, detector.predict_proba(X_hold)[:, 1]), decimals=1
        )
        == 0.9
    )


def test_explanation_audit_random():

    # Option 1: fit the detector when there is a trained model
    model = XGBClassifier().fit(X_tr, y_tr)

    detector = ExplanationAudit(model=model, gmodel=LogisticRegression())
    # On Random Data
    detector.fit_inspector(X_te, a_te_)
    # pdb.set_trace()
    assert (
        np.round(
            roc_auc_score(a_hold_, detector.predict_proba(X_hold)[:, 1]), decimals=1
        )
        == 0.5
    )


def test_fit_full_pipe():
    # Full pipe
    detector = ExplanationAudit(
        model=XGBClassifier(),
        gmodel=LogisticRegression(),
    )

    detector.fit(X_tr, y_tr, a_tr)

    # Partial Pipe
    m = XGBClassifier().fit(X_tr, y_tr)
    detector2 = ExplanationAudit(
        model=m,
        gmodel=LogisticRegression(),
    )
    detector2.fit_inspector(X_tr, a_tr)

    assert np.round(
        roc_auc_score(a_hold, detector.predict_proba(X_hold)[:, 1]), decimals=1
    ) == np.round(
        roc_auc_score(a_hold, detector2.predict_proba(X_hold)[:, 1]), decimals=1
    )


def test_masker():

    detector = ExplanationAudit(
        model=LogisticRegression(),
        gmodel=LogisticRegression(),
        masker=True,
        data_masker=X_tr,
    )

    detector.fit(X_tr, y_tr, a_tr)
    assert (
        np.round(
            roc_auc_score(a_hold, detector.predict_proba(X_hold)[:, 1]), decimals=1
        )
        > 0.6
    )


def test_mask_tree_explainer():

    detector = ExplanationAudit(
        model=XGBClassifier(),
        gmodel=LogisticRegression(),
        masker=True,
        data_masker=X_tr,
    )

    detector.fit(X_tr, y_tr, a_tr)
    assert (
        np.round(
            roc_auc_score(a_hold, detector.predict_proba(X_hold)[:, 1]), decimals=1
        )
        > 0.6
    )
