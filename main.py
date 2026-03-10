from src.data.load_data import load_data
from src.data.preprocess import build_preprocessing_pipeline
from src.models.train_model import train_model
from src.models.evaluate_model import evaluate

import pandas as pd
import matplotlib.pyplot as plt


# Load dataset
df = load_data("data/processed/clean_churn.csv")

# Convert target
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Split features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Detect feature types
num_features = X.select_dtypes(include=["int64", "float64"]).columns
cat_features = X.select_dtypes(include=["object", "string"]).columns

# Build preprocessing pipeline
preprocessor = build_preprocessing_pipeline(num_features, cat_features)

# Train model
model, X_test, y_test = train_model(X, y, preprocessor)

# Evaluate model
evaluate(model, X_test, y_test)


# ---------- Feature Importance Section ----------

model_step = model.named_steps["model"]
preprocessor = model.named_steps["preprocessor"]

# Only run if the model supports feature importance
if hasattr(model_step, "feature_importances_"):

    # Feature names after encoding
    feature_names = preprocessor.get_feature_names_out()

    # Importance values
    importances = model_step.feature_importances_

    # Create dataframe
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    })

    # Sort features
    importance_df = importance_df.sort_values(by="importance", ascending=False)

    print("\nTop 15 Important Features:")
    print(importance_df.head(15))

    # Plot feature importance
    plt.figure(figsize=(10,6))

    top_features = importance_df.head(15)

    plt.barh(top_features["feature"], top_features["importance"])

    plt.xlabel("Importance")
    plt.title("Top 15 Features Influencing Churn")

    plt.gca().invert_yaxis()

    plt.show()