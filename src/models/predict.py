import joblib
import pandas as pd


MODEL_PATH = "models/churn_model.pkl"


def load_model():
    return joblib.load(MODEL_PATH)


def prepare_input(customer_data, model):
    """
    Ensure input dataframe has the same columns used during training.
    """

    df = pd.DataFrame([customer_data])

    # Expected columns from the pipeline
    expected_columns = model.feature_names_in_

    for col in expected_columns:
        if col not in df.columns:
            df[col] = None

    df = df[expected_columns]

    return df


def predict_churn(customer_data):

    model = load_model()

    df = prepare_input(customer_data, model)

    probability = model.predict_proba(df)[0][1]

    threshold = 0.529

    prediction = int(probability > threshold)

    return prediction, probability


if __name__ == "__main__":

    sample_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.5,
        "TotalCharges": 420.2
    }

    prediction, probability = predict_churn(sample_customer)

    print("Churn Probability:", probability)

    if prediction == 1:
        print("Prediction: Customer likely to churn")
    else:
        print("Prediction: Customer likely to stay")