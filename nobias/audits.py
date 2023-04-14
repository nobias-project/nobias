from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
import numpy as np
import pandas as pd
import shap


class ExplanationAudit(BaseEstimator, ClassifierMixin):
    """
    Given a model, a dataset, and the protected attribute, we want to know if the model violates demographic parity or not and what are the features pushing for it.
    We do this by computing the shap values of the model, and then train a classifier to distinguish the protected attribute.

    Example
    -------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from sklearn.linear_model import LogisticRegression
    >>> from xgboost import XGBRegressor
    >>> from fairtools.detector import ExplanationAudit
    >>> N = 5_000
    >>> x1 = np.random.normal(1, 1, size=N)
    >>> x2 = np.random.normal(1, 1, size=N)
    >>> x34 = np.random.multivariate_normal([1, 1], [[1, 0.5], [0.5, 1]], size=N)
    >>> x3 = x34[:, 0]
    >>> x4 = x34[:, 1]
    >>> # Binarize protected attribute
    >>> x4 = np.where(x4 > np.mean(x4), 1, 0)
    >>> X = pd.DataFrame([x1, x2, x3, x4]).T
    >>> X.columns = ["var%d" % (i + 1) for i in range(X.shape[1])]
    >>> y = (x1 + x2 + x3) / 3
    >>> y = 1 / (1 + np.exp(-y))
    >>> detector = ExplanationAudit(model=XGBRegressor(), gmodel=LogisticRegression())
    >>> detector.fit(X, y, Z="var4")
    >>> detector.get_auc_val()
    """

    def __init__(
        self,
        model,
        gmodel,
        algorithm: str = "auto",
        masker: bool = False,
        data_masker: pd.DataFrame = None,
        verbose=False,
    ):
        self.model = model
        self.inspector = gmodel
        self.explainer = None
        self.algorithm = algorithm
        self.masker = masker
        self.data_masker = data_masker
        self.verbose = verbose

    def fit_inspector(self, X, z):
        """
        Fits the inspector model to the explanations

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        z : array-like of shape (n_samples,)
            The protected attribute values (class labels in classification, real numbers in regression).
        """
        try:
            check_is_fitted(self.model)
        except:
            raise ValueError(
                "Model is not fitted yet, to use this method the model must be fitted."
            )
        # Get explanations
        ## Filter by protected group z
        if self.verbose:
            print(f"Protected attribute values: {X[self.Z].unique()}")

        self.inspector.fit(self.get_explanations(X), z)

    def fit_pipeline(self, X, y, z):
        """
        1. Fits the model F to X and y
        2. Call fit_inspector to fit the Equal Treatment Inspector
        Careful this method trains F and G on the same data


        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels in classification, real numbers in regression).
        Z : array-like of shape (n_samples,)
            The protected attribute values (class labels in classification, real numbers in regression).
        """
        check_X_y(X, y)
        check_X_y(X, z)
        if self.verbose:
            print("Fitting the model")
            # Number of unique values in y
            print("Number of unique values in y:", len(np.unique(y)))

        self.model.fit(X, y)
        self.fit_inspector(X, z)

    def fit(self, X, y, Z):
        """
        Automatically fits the whole pipeline

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels in classification, real numbers in regression)
        Z : array-like of shape (n_samples,)
            The protected attribute values (class labels in classification, real numbers in regression)

        """
        self.fit_pipeline(X, y, Z)

    def get_explanations(self, X):
        """
        Returns the explanations of the model on the data X.
        Produces a dataframe with the explanations of the model on the data X.
        """
        if self.masker:
            self.explainer = shap.Explainer(
                self.model, algorithm=self.algorithm, masker=self.data_masker
            )
        else:
            self.explainer = shap.Explainer(self.model, algorithm=self.algorithm)

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

    def predict(self, X):
        """
        Returns the predictions (ID,OOD) of the detector on the data X.
        """
        return self.inspector.predict(self.get_explanations(X))

    def predict_proba(self, X):
        """
        Returns the soft predictions (ID,OOD) of the detector on the data X.
        """
        return self.inspector.predict_proba(self.get_explanations(X))
