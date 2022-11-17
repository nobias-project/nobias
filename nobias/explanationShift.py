from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pandas as pd
import shap


class ExplanationShiftDetector(BaseEstimator, ClassifierMixin):
    """
    Given a model, and two datasets (source,test), we want to know if the behaviour of the model is different bt train and test.
    We can do this by computing the shap values of the model on the two datasets, and then train a classifier to distinguish between the two datasets.

    Example
    -------
    >>> import pandas as pd
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.datasets import make_blobs
    >>> from nobias import ExplanationShiftDetector
    >>> from xgboost import XGBRegressor
    >>> from sklearn.linear_model import LogisticRegression

    >>> X, y = make_blobs(n_samples=2000, centers=2, n_features=5, random_state=0)
    >>> X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=0)
    >>> X_ood,y_ood = make_blobs(n_samples=1000, centers=1, n_features=5, random_state=0)

    >>> detector = ExplanationShiftDetector(model=XGBRegressor(),gmodel=LogisticRegression())
    >>> detector.fit(X_tr, y_tr, X_ood)
    >>> detector.get_auc_val()
    # 0.76
    >>> detector.fit(X_tr, y_tr, X_te)
    >>> detector.get_auc_val()
    # 0.5
    """

    def __init__(self, model, gmodel):
        self.model = model
        self.gmodel = gmodel
        self.explainer = None

        # Supported F Models
        self.supported_tree_models = ["XGBClassifier", "XGBRegressor"]
        self.supported_linear_models = [
            "LogisticRegression",
            "LinearRegression",
            "Ridge",
            "Lasso",
        ]
        self.supported_models = (
            self.supported_tree_models + self.supported_linear_models
        )
        # Supported detectors
        self.supported_linear_detectors = [
            "LogisticRegression",
        ]
        self.supported_tree_detectors = ["XGBClassifier"]
        self.supported_detectors = (
            self.supported_linear_detectors + self.supported_tree_detectors
        )

        # Check if models are supported
        if self.get_model_type() not in self.supported_models:
            raise ValueError(
                "Model not supported. Supported models are: {} got {}".format(
                    self.supported_models, self.model.__class__.__name__
                )
            )
        if self.get_gmodel_type() not in self.supported_detectors:
            raise ValueError(
                "gmodel not supported. Supported models are: {} got {}".format(
                    self.supported_detectors, self.gmodel.__class__.__name__
                )
            )

    def get_gmodel_type(self):
        if self.gmodel.__class__.__name__ == "Pipeline":
            return self.gmodel.steps[-1][1].__class__.__name__
        else:
            return self.gmodel.__class__.__name__

    def get_model_type(self):
        if self.model.__class__.__name__ == "Pipeline":
            return self.model.steps[-1][1].__class__.__name__
        else:
            return self.model.__class__.__name__

    def fit(self, X_source, y_source, X_ood):

        # Check that X and y have correct shape
        check_X_y(X_source, y_source)
        self.X_ood = X_ood

        self.X_tr, self.X_val, self.y_tr, self.y_val = train_test_split(
            X_source, y_source, random_state=0, test_size=0.5
        )

        # Fit model F
        self.fit_model(self.X_tr, self.y_tr)

        # Get explanations
        self.S_val = self.get_explanations(self.X_val)
        self.S_ood = self.get_explanations(self.X_ood)

        # Create dataset for  explanation shift detector
        self.S_val["label"] = 1
        self.S_ood["label"] = 0

        self.S = pd.concat([self.S_val, self.S_ood])

        (
            self.X_shap_tr,
            self.X_shap_te,
            self.y_shap_tr,
            self.y_shap_te,
        ) = train_test_split(
            self.S.drop(columns="label"), self.S["label"], random_state=0, test_size=0.5
        )
        self.fit_explanation_shift(self.X_shap_tr, self.y_shap_tr)

        return self

    def predict(self, X):
        return self.gmodel.predict(self.get_explanations(X))

    def predict_proba(self, X):
        return self.gmodel.predict_proba(self.get_explanations(X))

    def explanation_predict(self, X):
        return self.gmodel.predict(X)

    def explanation_predict_proba(self, X):
        return self.gmodel.predict_proba(X)

    def fit_model(self, X, y):
        self.model.fit(X, y)

    def fit_explanation_shift(self, X, y):
        self.gmodel.fit(X, y)

    def get_explanations(self, X):
        # Determine the type of SHAP explainer to use
        if self.get_model_type() in self.supported_tree_models:
            self.explainer = shap.Explainer(self.model)
        elif self.get_model_type() in self.supported_linear_models:
            self.explainer = shap.LinearExplainer(
                self.model, X, feature_dependence="correlation_dependent"
            )
        else:
            raise ValueError(
                "Model not supported. Supported models are: {}, got {}".format(
                    self.supported_models, self.model.__class__.__name__
                )
            )

        shap_values = self.explainer(X)
        # Name columns
        if isinstance(X, pd.DataFrame):
            columns_name = X.columns
        else:
            columns_name = ["Shap%d" % (i + 1) for i in range(X.shape[1])]

        exp = pd.DataFrame(
            data=shap_values.values,
            columns=columns_name,
        )
        return exp

    def get_auc_val(self):
        """
        Returns the AUC of the explanation shift detector on the validation set of the explanation space
        Example
        -------
        from sklearn.model_selection import train_test_split
        from sklearn.datasets import make_blobs
        from tools.xaiUtils import ExplanationShiftDetector
        from xgboost import XGBRegressor
        from sklearn.linear_model import LogisticRegression

        # Create data
        X, y = make_blobs(n_samples=2000, centers=2, n_features=5, random_state=0)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=0)
        X_ood,y_ood = make_blobs(n_samples=1000, centers=1, n_features=5, random_state=0)

        detector = ExplanationShiftDetector(model=XGBRegressor(),gmodel=LogisticRegression())
        detector.fit(X_tr, y_tr, X_ood)
        detector.get_auc_val()
        # 0.76

        """
        return roc_auc_score(
            self.y_shap_te, self.explanation_predict_proba(self.X_shap_te)[:, 1]
        )

    def get_coefs(self):
        if self.gmodel.__class__.__name__ == "Pipeline":
            if (
                self.gmodel.steps[-1][1].__class__.__name__
                in self.supported_linear_models
            ):
                return self.gmodel.steps[-1][1].coef_
            else:
                raise ValueError(
                    "Pipeline model not supported. Supported models are: {}, got {}".format(
                        self.supported_linear_models,
                        self.gmodel.steps[-1][1].__class__.__name__,
                    )
                )
        else:
            return self.get_linear_coefs()

    def get_linear_coefs(self):
        if self.gmodel.__class__.__name__ in self.supported_linear_models:
            return self.gmodel.coef_
        else:
            raise ValueError(
                "Detector model not supported. Supported models ar linear: {}, got {}".format(
                    self.supported_linear_detector, self.model.__class__.__name__
                )
            )
