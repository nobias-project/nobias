
Explanation Shift Detector
---------------------------

.. note::

   This project is under active development.

Importing libraries

.. code:: python

    import numpy as np
    from sklearn.datasets import make_blobs
    from xgboost import XGBRegressor
    from sklearn.linear_model import LogisticRegression
    from nobias import ExplanationShiftDetector


Generate synthetic ID and OOD data.

.. code:: python

    X, y = make_blobs(n_samples=2000, centers=2, n_features=5, random_state=0)
    X_ood, _ = make_blobs(n_samples=1000, centers=1, n_features=5, random_state=0)

Fit Explanation Shift Detector where the classifier is a Gradient Boosting Decision Tree and the Detector a logistic regression. Any other classifier or detector can be used.

.. code:: python

    detector = ExplanationShiftDetector(model=XGBRegressor(), gmodel=LogisticRegression())
    detector.fit(X, y,X_ood)
   
If the AUC is above 0.5 then we can expect and change on the model predictions.

.. code:: python

    detector.get_auc_val()
    # 0.70
