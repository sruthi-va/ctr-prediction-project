# ============================================================
# Ad Recommendation & CTR Prediction System
# ============================================================
# Features:
# - Synthetic large-scale user/ad interaction dataset
# - Behavioral + contextual feature engineering
# - Logistic Regression baseline
# - Gradient Boosting model with tuning
# - Neural Network with TensorFlow
# - ROC-AUC / Log Loss / Precision-Recall evaluation
# - Ad ranking pipeline (Top-N recommendations per user)
# - Experiment tracking table
# - Model persistence
# ============================================================

import numpy as np
import pandas as pd
import random
import joblib
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    precision_recall_curve,
    auc
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV

# ============================================================
# Reproducibility
# ============================================================

np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# ============================================================
# Generate Synthetic Dataset
# ============================================================

N_SAMPLES = 250000

user_interests = [
    "sports",
    "technology",
    "fashion",
    "finance",
    "gaming"
]

ad_categories = [
    "sports",
    "technology",
    "fashion",
    "finance",
    "gaming"
]

traffic_sources = [
    "search",
    "social",
    "email",
    "direct"
]

devices = [
    "mobile",
    "desktop"
]

data = pd.DataFrame({
    "user_id": np.random.randint(1000, 50000, N_SAMPLES),
    "ad_id": np.random.randint(100, 1000, N_SAMPLES),

    # Behavioral features
    "session_frequency": np.random.poisson(5, N_SAMPLES),
    "weekly_sessions": np.random.poisson(10, N_SAMPLES),
    "pages_viewed": np.random.poisson(8, N_SAMPLES),
    "time_on_site": np.random.gamma(2, 5, N_SAMPLES),
    "past_click_rate": np.random.beta(2, 5, N_SAMPLES),
    "days_since_last_click": np.random.randint(0, 30, N_SAMPLES),
    "scroll_depth": np.random.uniform(0.2, 1.0, N_SAMPLES),
    "cart_additions": np.random.poisson(2, N_SAMPLES),

    # Contextual features
    "hour_of_day": np.random.randint(0, 24, N_SAMPLES),
    "weekday": np.random.randint(0, 7, N_SAMPLES),
    "ad_position": np.random.randint(1, 6, N_SAMPLES),
    "device_type": np.random.choice(devices, N_SAMPLES),
    "traffic_source": np.random.choice(traffic_sources, N_SAMPLES),

    # User/ad interests
    "user_interest": np.random.choice(user_interests, N_SAMPLES),
    "ad_category": np.random.choice(ad_categories, N_SAMPLES),

    # Engagement
    "engagement_score": np.random.normal(50, 15, N_SAMPLES)
})

# ============================================================
# Feature Engineering
# ============================================================

# Interest match feature
data["interest_match"] = (
    data["user_interest"] == data["ad_category"]
).astype(int)

# Interaction features
data["engagement_x_ctr"] = (
    data["engagement_score"] * data["past_click_rate"]
)

# Night activity feature
data["night_user"] = (
    (data["hour_of_day"] >= 22) |
    (data["hour_of_day"] <= 5)
).astype(int)

# ============================================================
# Simulate Click Probability
# ============================================================

logits = (
    0.30 * data["session_frequency"]
    + 0.50 * data["weekly_sessions"]
    + 1.50 * data["past_click_rate"]
    + 0.25 * data["pages_viewed"]
    + 0.02 * data["engagement_score"]
    + 2.00 * data["interest_match"]
    - 0.20 * data["ad_position"]
    - 0.05 * data["days_since_last_click"]
    + 0.30 * data["scroll_depth"]
)

# Add randomness/noise
logits += np.random.normal(0, 2, N_SAMPLES)

# Sigmoid transformation
probabilities = 1 / (1 + np.exp(-0.08 * logits))

# Binary click target
data["clicked"] = np.random.binomial(1, probabilities)

# ============================================================
# One-Hot Encoding
# ============================================================

categorical_cols = [
    "device_type",
    "traffic_source",
    "user_interest",
    "ad_category"
]

data = pd.get_dummies(
    data,
    columns=categorical_cols,
    drop_first=True
)

# ============================================================
# Train/Test Split
# ============================================================

X = data.drop(columns=["clicked"])
y = data["clicked"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ============================================================
# Feature Scaling
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# Logistic Regression
# ============================================================

print("\n==============================")
print("Training Logistic Regression")
print("==============================")

lr_model = LogisticRegression(
    max_iter=1000
)

lr_model.fit(X_train_scaled, y_train)

lr_preds = lr_model.predict_proba(X_test_scaled)[:, 1]

lr_auc = roc_auc_score(y_test, lr_preds)
lr_logloss = log_loss(y_test, lr_preds)

print("AUC:", round(lr_auc, 4))
print("Log Loss:", round(lr_logloss, 4))

# ============================================================
# Gradient Boosting + Hyperparameter Tuning
# ============================================================

print("\n==============================")
print("Training Gradient Boosting")
print("==============================")

param_grid = {
    "n_estimators": [50, 100],
    "learning_rate": [0.05, 0.1],
    "max_depth": [3, 5]
}

gb_search = RandomizedSearchCV(
    GradientBoostingClassifier(),
    param_distributions=param_grid,
    n_iter=4,
    scoring="roc_auc",
    cv=3,
    verbose=1,
    random_state=42
)

gb_search.fit(X_train, y_train)

gb_model = gb_search.best_estimator_

gb_preds = gb_model.predict_proba(X_test)[:, 1]

gb_auc = roc_auc_score(y_test, gb_preds)
gb_logloss = log_loss(y_test, gb_preds)

print("Best Params:", gb_search.best_params_)
print("AUC:", round(gb_auc, 4))
print("Log Loss:", round(gb_logloss, 4))

# ============================================================
# Neural Network
# ============================================================

print("\n==============================")
print("Training Neural Network")
print("==============================")

nn_model = tf.keras.Sequential([
    tf.keras.layers.Dense(
        128,
        activation="relu",
        input_shape=(X_train_scaled.shape[1],)
    ),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(
        64,
        activation="relu"
    ),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    )
])

nn_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["AUC"]
)

nn_model.fit(
    X_train_scaled,
    y_train,
    epochs=5,
    batch_size=256,
    validation_split=0.1,
    verbose=1
)

nn_preds = nn_model.predict(X_test_scaled).flatten()

nn_auc = roc_auc_score(y_test, nn_preds)
nn_logloss = log_loss(y_test, nn_preds)

print("AUC:", round(nn_auc, 4))
print("Log Loss:", round(nn_logloss, 4))

# ============================================================
# Precision-Recall Evaluation
# ============================================================

precision, recall, _ = precision_recall_curve(
    y_test,
    gb_preds
)

pr_auc = auc(recall, precision)

print("\n==============================")
print("Precision-Recall Evaluation")
print("==============================")
print("PR-AUC:", round(pr_auc, 4))

# ============================================================
# Experiment Tracking Table
# ============================================================

results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Gradient Boosting",
        "Neural Network"
    ],
    "ROC_AUC": [
        lr_auc,
        gb_auc,
        nn_auc
    ],
    "Log_Loss": [
        lr_logloss,
        gb_logloss,
        nn_logloss
    ]
})

print("\n==============================")
print("Experiment Results")
print("==============================")
print(results)

# ============================================================
# Save Best Model
# ============================================================

joblib.dump(gb_model, "best_ctr_model.pkl")
joblib.dump(scaler, "feature_scaler.pkl")

print("\nSaved best model and scaler.")

# ============================================================
# Recommendation Ranking Pipeline
# ============================================================

print("\n==============================")
print("Generating Ad Rankings")
print("==============================")

recommendation_df = X_test.copy()

recommendation_df["predicted_ctr"] = gb_preds
recommendation_df["user_id"] = X_test["user_id"]
recommendation_df["ad_id"] = X_test["ad_id"]

# Top 5 ads per user
top_ads = (
    recommendation_df
    .sort_values(
        ["user_id", "predicted_ctr"],
        ascending=False
    )
    .groupby("user_id")
    .head(5)
)

print(top_ads[
    [
        "user_id",
        "ad_id",
        "predicted_ctr"
    ]
].head(20))

print("\nPipeline Complete.")