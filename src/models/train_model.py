from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import joblib

def train_model(X, y, preprocessor):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=42
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    param_grid = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [5, 10, 15],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2]
    }
    
    scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=5,
        scoring="f1"
    )

    print("\nCross Validation F1 Scores:", scores)
    print("Average F1 Score:", scores.mean())

    grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    verbose=1
)

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    print("\nBest Parameters:")
    print(grid_search.best_params_)

    joblib.dump(best_model, "models/churn_model.pkl")

    return best_model, X_test, y_test