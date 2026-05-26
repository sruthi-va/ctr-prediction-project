
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import tensorflow as tf

# -----------------------------
# Generate Synthetic Ad Dataset
# -----------------------------
np.random.seed(42)

n_samples = 50000

data = pd.DataFrame({
    "session_frequency": np.random.poisson(5, n_samples),
    "time_on_site": np.random.gamma(2, 5, n_samples),
    "pages_viewed": np.random.poisson(8, n_samples),
    "past_click_rate": np.random.beta(2, 5, n_samples),
    "device_mobile": np.random.randint(0, 2, n_samples),
    "hour_of_day": np.random.randint(0, 24, n_samples),
    "ad_position": np.random.randint(1, 6, n_samples),
    "engagement_score": np.random.normal(50, 15, n_samples)
})

# Simulated CTR target
logits = (
    0.25 * data["session_frequency"]
    + 0.8 * data["past_click_rate"] * 10
    + 0.3 * data["pages_viewed"]
    + 0.03 * data["engagement_score"]
    - 0.15 * data["ad_position"]
)

probabilities = 1 / (1 + np.exp(-0.1 * logits))
data["clicked"] = np.random.binomial(1, probabilities)

# -----------------------------
# Train/Test Split
# -----------------------------
X = data.drop("clicked", axis=1)
y = data["clicked"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Feature Scaling
# -----------------------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# Logistic Regression Model
# -----------------------------
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_scaled, y_train)

lr_preds = lr_model.predict_proba(X_test_scaled)[:, 1]

print("\n=== Logistic Regression ===")
print("AUC:", roc_auc_score(y_test, lr_preds))
print("Log Loss:", log_loss(y_test, lr_preds))

# -----------------------------
# Gradient Boosting Model
# -----------------------------
gb_model = GradientBoostingClassifier()
gb_model.fit(X_train, y_train)

gb_preds = gb_model.predict_proba(X_test)[:, 1]

print("\n=== Gradient Boosting ===")
print("AUC:", roc_auc_score(y_test, gb_preds))
print("Log Loss:", log_loss(y_test, gb_preds))

# -----------------------------
# Neural Network Model
# -----------------------------
nn_model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation="relu", input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
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
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

nn_preds = nn_model.predict(X_test_scaled).flatten()

print("\n=== Neural Network ===")
print("AUC:", roc_auc_score(y_test, nn_preds))
print("Log Loss:", log_loss(y_test, nn_preds))

# -----------------------------
# Recommendation Ranking Example
# -----------------------------
recommendation_df = X_test.copy()
recommendation_df["predicted_ctr"] = gb_preds

top_ads = recommendation_df.sort_values(
    by="predicted_ctr",
    ascending=False
).head(10)

print("\n=== Top Recommended Ad Opportunities ===")
print(top_ads[["predicted_ctr", "session_frequency", "past_click_rate"]])
