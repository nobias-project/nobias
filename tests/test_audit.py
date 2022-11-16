from nobias import ExplanationAudit

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
import pandas as pd
import numpy as np


X, y = make_blobs(n_samples=1000, centers=2, n_features=5, random_state=0)
X = pd.DataFrame(X, columns=["a", "b", "c", "d", "e"])
# Protected att
X["a"] = np.where(X["a"] > X["a"].mean(), 1, 0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=0)


def test_get_splits():
    detector = ExplanationAudit(
        model=GradientBoostingRegressor(), gmodel=LogisticRegression()
    )
    N = 1000
    X, y = make_blobs(n_samples=N, centers=2, n_features=5, random_state=0)
    X = pd.DataFrame(X, columns=["a", "b", "c", "d", "e"])
    # Binarize
    X["a"] = np.where(X["a"] > X["a"].mean(), 1, 0)

    detector.get_split_data(X, y, Z="a", n1=0.6, n2=0.5)
    assert detector.X_tr.shape == (N * 0.4, 5)
    assert detector.X_val.shape == (N * 0.6 * 0.5, 5)
    assert detector.X_te.shape == (N * 0.6 * 0.5, 5)

    assert len(set(detector.y_tr)) > 1
    assert len(set(detector.y_val)) > 1
    assert len(set(detector.y_te)) > 1


def test_return_shapDF():
    """
    If X is dataframe, return shap values as dataframe.
    """
    XX = pd.DataFrame(X, columns=["a", "b", "c", "d", "e"])

    esd = ExplanationAudit(model=LinearRegression(), gmodel=LogisticRegression())
    esd.fit_model(XX, y)
    ex = esd.get_explanations(XX)
    assert all([a == b for a, b in zip(ex.columns, XX.columns)])


def test_supported_models():
    """
    Check that models are supported.
    """
    for model in [GradientBoostingRegressor(), LogisticRegression()]:
        for gmodel in [GradientBoostingClassifier(), LogisticRegression()]:
            assert (
                type(ExplanationAudit(model=model, gmodel=gmodel)) is ExplanationAudit
            )


def test_not_supported_models():
    """
    Check that models are not supported.
    """

    from sklearn.neural_network import MLPClassifier
    import pytest

    with pytest.raises(ValueError):
        ExplanationAudit(model=MLPClassifier(), gmodel=LogisticRegression())
    with pytest.raises(ValueError):
        ExplanationAudit(model=LinearRegression(), gmodel=MLPClassifier())


def test_get_model_types():
    """
    Check that the model types are returned correctly.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    esd = ExplanationAudit(model=LinearRegression(), gmodel=LogisticRegression())
    assert esd.get_gmodel_type(), esd.get_model_type() == ("linear", "linear")
    # Case of pipeline
    esd = ExplanationAudit(
        model=Pipeline([("scaler", StandardScaler()), ("lr", LinearRegression())]),
        gmodel=Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression())]),
    )
    assert esd.get_gmodel_type(), esd.get_model_type() == ("linear", "linear")


def test_no_nan():
    """
    Check that no NaNs are present in the shap values.
    """
    esd = ExplanationAudit(model=LinearRegression(), gmodel=LogisticRegression())
    esd.fit_model(X, y)
    ex = esd.get_explanations(X)
    assert not np.any(np.isnan(ex))


def test_get_coefs_linear():
    """
    Check that the coefficients are returned correctly for the linear regression.
    """
    esd = ExplanationAudit(model=LinearRegression(), gmodel=LogisticRegression())
    esd.fit(X, y, Z="a")
    coefs = esd.get_linear_coefs()
    # Assert shape -1 from protected attribute
    assert len(coefs) == X.shape[1] - 1
    # Assert that there is non NaNs
    assert not np.any(np.isnan(coefs))
    # Check when we call the full methods
    coefs = esd.get_coefs()
    # Assert shape -1 from protected attribute
    assert len(coefs) == X.shape[1] - 1
    # Assert that there is non NaNs
    assert not np.any(np.isnan(coefs))


def test_get_coefs_pipeline():
    """
    Check that the coefficients are returned correctly for the linear regression pipeline.
    TODO : add a test for the case of a pipeline for F.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    esd = ExplanationAudit(
        model=LinearRegression(),
        gmodel=Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression())]),
    )
    esd.fit(X, y, Z="a")
    coefs = esd.get_coefs()
    # Assert shape -1 from protected attribute
    assert coefs.shape[1] == X.shape[1] - 1
    # Assert that there is non NaNs
    assert not np.any(np.isnan(coefs))


def test_predict_drop_prottected():
    """
    Check that the protected attribute is dropped when predicting.
    """
    audit = ExplanationAudit(
        model=GradientBoostingRegressor(), gmodel=LogisticRegression()
    )

    audit.fit(X, y, Z="a")
    assert np.isnan(audit.predict(X)).sum() == 0
    assert np.isnan(audit.predict_proba(X)).sum() == 0


'''
def test_doc_examples():
    """
    Check that doc examples work.
    WARNING: this test takes a long time to run. (1-5mins)
    """
    from folktables import ACSDataSource, ACSIncome

    data_source = ACSDataSource(survey_year="2014", horizon="1-Year", survey="person")
    try:
        acs_data = data_source.get_data(states=["CA"], download=False)
    except:
        acs_data = data_source.get_data(states=["CA"], download=True)

    X_, y_, _ = ACSIncome.df_to_numpy(acs_data)
    X_ = pd.DataFrame(X_, columns=ACSIncome.features)

    # White vs ALL
    X_["RAC1P"] = np.where(X_["RAC1P"] == 1, 1, 0)

    detector = ExplanationAudit(
        model=GradientBoostingRegressor(random_state=0), gmodel=LogisticRegression()
    )

    detector.fit(X_, y_, Z="RAC1P")
    # Check that the model prediction works
    assert np.round(detector.get_auc_val(), decimals=1) == 0.7
    # Check that the coefficients are returned correctly
    coefs = detector.get_coefs()
    assert len(coefs) == X_.shape[1] - 1  # -1 for the protected attribute
    assert np.isnan(coefs).sum() == 0
    # Hard coded results.
    # assert np.round(coefs,decimals=0) ==[ 1., -2., -1.,  1., -0., 17.,  2., -0.,  1.]
'''
