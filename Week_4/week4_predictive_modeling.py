import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("cleaned_logistics_data.csv")

rng = np.random.default_rng(7)
base_rate = {"Standard Class": 8, "Second Class": 12, "First Class": 18, "Same Day": 25}
market_surcharge = {"USCA": 3, "LATAM": 4, "Europe": 6, "Pacific Asia": 7, "Africa": 8}

df["transportation_cost"] = (
    df["shipping_mode"].map(base_rate)
    + df["order_item_quantity"] * 2.5
    + df["market"].map(market_surcharge)
    + rng.normal(0, 3, len(df))
).clip(lower=5).round(2)

target = "actual_shipping_days"
features = [
    "scheduled_shipping_days", "order_item_quantity", "sales_capped",
    "benefit_per_order", "transportation_cost", "shipping_mode",
    "market", "customer_segment", "category_name"
]
X = df[features]
y = df[target]

numeric_features = [
    "scheduled_shipping_days", "order_item_quantity", "sales_capped",
    "benefit_per_order", "transportation_cost"
]
categorical_features = [
    "shipping_mode", "market", "customer_segment", "category_name"
]

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric_features),
    ("cat", categorical_pipe, categorical_features)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

linear_model = Pipeline([
    ("prep", preprocessor),
    ("model", LinearRegression())
])

rf_model = Pipeline([
    ("prep", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        min_samples_leaf=2,
        n_jobs=-1
    ))
])

linear_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

def evaluate(model):
    pred = model.predict(X_test)
    return (
        mean_absolute_error(y_test, pred),
        np.sqrt(mean_squared_error(y_test, pred)),
        r2_score(y_test, pred)
    )

print("Linear Regression:", evaluate(linear_model))
print("Random Forest:", evaluate(rf_model))

cv = KFold(n_splits=5, shuffle=True, random_state=42)

cv_mae = -cross_val_score(
    rf_model, X_train, y_train,
    cv=cv, scoring="neg_mean_absolute_error"
)
cv_rmse = -cross_val_score(
    rf_model, X_train, y_train,
    cv=cv, scoring="neg_root_mean_squared_error"
)

print("5-fold CV MAE:", cv_mae.mean())
print("5-fold CV RMSE:", cv_rmse.mean())

tune_pipe = Pipeline([
    ("prep", preprocessor),
    ("model", RandomForestRegressor(random_state=42, n_jobs=-1))
])

param_grid = {
    "model__n_estimators": [200, 300],
    "model__max_depth": [None, 12],
    "model__min_samples_leaf": [1, 2]
}

grid = GridSearchCV(
    tune_pipe, param_grid,
    cv=3,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
best_pred = best_model.predict(X_test)

print("Best parameters:", grid.best_params_)
print("Tuned Random Forest:",
      mean_absolute_error(y_test, best_pred),
      np.sqrt(mean_squared_error(y_test, best_pred)),
      r2_score(y_test, best_pred))

results = X_test.copy()
results["actual_shipping_days"] = y_test.values
results["predicted_shipping_days"] = best_pred
results["predicted_delay_over_schedule"] = (
    results["predicted_shipping_days"]
    - results["scheduled_shipping_days"]
)

results["risk_band"] = pd.cut(
    results["predicted_delay_over_schedule"],
    bins=[-np.inf, 0, 1, np.inf],
    labels=["On/within schedule", "Moderate risk", "High risk"]
)

results.to_csv("week4_test_predictions.csv", index=False)
