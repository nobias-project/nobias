# %%
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from nobias import ExplanationAudit
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
random.seed(0)


# %%
X, y = make_blobs(n_samples=2000, centers=2, n_features=5, random_state=0)
X = pd.DataFrame(X, columns=["a", "b", "c", "d", "e"])
# Protected att
X["a"] = np.where(X["a"] > X["a"].mean(), 1, 0)

# Train Val Holdout Split
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=0)
X_hold, X_te, y_hold, y_te = train_test_split(X_te, y_te, test_size=0.5, random_state=0)

z_tr = X_tr["a"]
z_te = X_te["a"]
z_hold = X_hold["a"]
X_tr = X_tr.drop("a", axis=1)
X_te = X_te.drop("a", axis=1)
X_hold = X_hold.drop("a", axis=1)
# Random
z_tr_ = np.random.randint(0, 2, size=X_tr.shape[0])
z_te_ = np.random.randint(0, 2, size=X_te.shape[0])
z_hold_ = np.random.randint(0, 2, size=X_hold.shape[0])

# %%
# Option 1: fit the auditor when there is a trained model
model = XGBClassifier().fit(X_tr, y_tr)

auditor = ExplanationAudit(model=model, gmodel=LogisticRegression())

auditor.fit_inspector(X_hold, z_hold)
print(roc_auc_score(z_te, auditor.predict_proba(X_te)[:, 1]))
# 0.84
# %%
# Option 2: fit the whole pipeline of model and auditor at once
auditor.fit_pipeline(X=X_tr, y=y_tr, z=z_tr)
print(roc_auc_score(z_te, auditor.predict_proba(X_te)[:, 1]))
# 0.83
# %%
# If we fit to random protected att, there is no performance
auditor.fit_pipeline(X=X_tr, y=y_tr, z=z_tr_)
print(roc_auc_score(z_te_, auditor.predict_proba(X_te)[:, 1]))
# 0.5
# %%
# Explaining the change of the model
import shap

explainer = shap.Explainer(auditor.inspector, masker=X_te)
shap_values = explainer(auditor.get_explanations(X_te))
# visualize the first prediction's explanation
shap.waterfall_plot(shap_values[0])
plt.close()

# %%
# Real World Example - Folktables
from folktables import ACSDataSource, ACSIncome
import pandas as pd

data_source = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person")
ca_data = data_source.get_data(states=["CA"], download=True)
ca_features, ca_labels, ca_group = ACSIncome.df_to_pandas(ca_data)
ca_features = ca_features.drop(columns="RAC1P")
ca_features["group"] = ca_group
ca_features["label"] = ca_labels
# Lets focus on groups 1 and 6
ca_features = ca_features[ca_features["group"].isin([1, 6])]
# %%
# Split train, test and holdout
X_tr, X_te, y_tr, y_te = train_test_split(
    ca_features.drop(columns="label"), ca_features.label, test_size=0.5, random_state=0
)
X_hold, X_te, y_hold, y_te = train_test_split(X_te, y_te, test_size=0.5, random_state=0)
# Prot att.
z_tr = np.where(X_tr["group"].astype(int) == 6, 0, 1)
z_te = np.where(X_te["group"].astype(int) == 6, 0, 1)
z_hold = np.where(X_hold["group"].astype(int) == 6, 0, 1)
X_tr = X_tr.drop("group", axis=1)
X_te = X_te.drop("group", axis=1)
X_hold = X_hold.drop("group", axis=1)

model = XGBClassifier().fit(X_tr, y_tr)

# %%
auditor = ExplanationAudit(model=model, gmodel=XGBClassifier())

auditor.fit_inspector(X_te, z_te)
print(roc_auc_score(z_hold, auditor.predict_proba(X_hold)[:, 1]))
# 0.96
# %%
explainer = shap.Explainer(auditor.inspector)

shap_values = explainer(auditor.get_explanations(X_hold))
# Local Explanations
fig = shap.waterfall_plot(shap_values[0], show=False)
plt.savefig("docs/source/images/folksShapLocal.png")
plt.close()
# Global Explanations
fig = shap.plots.bar(shap_values, show=False)
plt.savefig("docs/source/images/folkstShapGlobal.png")
plt.close()

# %%
# Now if we choose a differet another groups
ca_features, ca_labels, ca_group = ACSIncome.df_to_pandas(ca_data)
ca_features = ca_features.drop(columns="RAC1P")
ca_features["group"] = ca_group
ca_features["label"] = ca_labels
# Lets focus on groups 1 and 6
ca_features = ca_features[ca_features["group"].isin([8, 6])]
# %%
# Split train, test and holdout
X_tr, X_te, y_tr, y_te = train_test_split(
    ca_features.drop(columns="label"), ca_features.label, test_size=0.5, random_state=0
)
X_hold, X_te, y_hold, y_te = train_test_split(X_te, y_te, test_size=0.5, random_state=0)
# Prot att.
z_tr = np.where(X_tr["group"].astype(int) == 6, 0, 1)
z_te = np.where(X_te["group"].astype(int) == 6, 0, 1)
z_hold = np.where(X_hold["group"].astype(int) == 6, 0, 1)
X_tr = X_tr.drop("group", axis=1)
X_te = X_te.drop("group", axis=1)
X_hold = X_hold.drop("group", axis=1)

model = XGBClassifier().fit(X_tr, y_tr)

# %%
auditor = ExplanationAudit(model=model, gmodel=XGBClassifier())

auditor.fit_inspector(X_te, z_te)
print(roc_auc_score(z_hold, auditor.predict_proba(X_hold)[:, 1]))
# 0.96
# %%
explainer = shap.Explainer(auditor.inspector)

shap_values = explainer(auditor.get_explanations(X_hold))
# Local Explanations
import matplotlib.pyplot as plt

fig = shap.waterfall_plot(shap_values[0], show=False)
plt.savefig("docs/source/images/folksShapLocal2.png")
plt.close()
# Global Explanations
fig = shap.plots.bar(shap_values, show=False)
plt.savefig("docs/source/images/folkstShapGlobal2.png")
plt.close()
