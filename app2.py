import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load artifacts
model_files = {
    "Logistic Regression": "logistic_regression_model.pkl",
    "KNN": "knn_model.pkl",
    "Decision Tree": "decision_tree_model.pkl",
    "Random Forest": "random_forest_model.pkl",
    "SVM": "svm_model.pkl",
    "XGBoost" : "xgboost_model.pkl"
}

# Load scaler and encoder
scaler = joblib.load("standard_scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Create USER_ID to Name mapping for 53 users
def generate_user_name_mapping(n_users):
    return {i: f"User_{i:02d}" for i in range(n_users)}

user_id_to_name = generate_user_name_mapping(53)

st.title("🔐 Keystroke Dynamics - User Identification")
st.write("Upload a typing sample CSV to predict the user based on keystroke biometrics.")

# Model selection
model_choice = st.selectbox("Choose a model", list(model_files.keys()))

# Load the selected model
model = joblib.load(model_files[model_choice])

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV file with keystroke features", type=["csv"])

if uploaded_file:
    try:
        input_df = pd.read_csv(uploaded_file)

        # Ensure proper shape
        if input_df.shape[1] != 71:
            st.error("Uploaded file must contain exactly 71 feature columns.")
        else:
            # Scale features
            input_scaled = scaler.transform(input_df)

            # Predict
            pred_encoded = model.predict(input_scaled)
            pred_user_ids = label_encoder.inverse_transform(pred_encoded)
            pred_names = [user_id_to_name.get(uid, f"User_{uid}") for uid in pred_user_ids]

            # Legitimate user check
            legitimacy = ["Legitimate" if uid in user_id_to_name else "Unknown" for uid in pred_user_ids]

            # Add probability/confidence if available
            try:
                proba = model.predict_proba(input_scaled)
                confidence = np.max(proba, axis=1)
            except:
                confidence = ["N/A"] * len(pred_user_ids)

            # Display results
            input_df['Predicted_USER_ID'] = pred_user_ids
            input_df['Predicted_Name'] = pred_names
            input_df['Legitimacy'] = legitimacy
            input_df['Confidence'] = confidence

            st.success("Prediction completed successfully!")
            st.dataframe(input_df[['Predicted_USER_ID', 'Predicted_Name', 'Legitimacy', 'Confidence']])

            # Summary
            st.subheader("🔍 Summary")
            summary = input_df['Predicted_Name'].value_counts().reset_index()
            summary.columns = ['User', 'Predicted_Count']
            st.dataframe(summary)

            # Download results
            csv = input_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Results CSV", csv, "keystroke_predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"An error occurred: {e}")
