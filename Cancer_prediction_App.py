
import io
import os
import pickle
import joblib
import urllib.request
import streamlit as st
import pandas as pd
import numpy as np


st.set_page_config(page_title="Cancer Prediction App", layout="wide")

try:
    _cache_decorator = getattr(st, "cache_resource")
except Exception:
    _cache_decorator = None

if _cache_decorator is None:
    _cache_decorator = getattr(st, "experimental_singleton", None)

if _cache_decorator is None:
    def _identity(func):

        return func

    _cache_decorator = _identity


@_cache_decorator

def load_model():
    if not os.path.isfile('model.pkl'):
        urllib.request.urlretrieve("https://raw.githubusercontent.com/INFO523-Fall25-101-201/final-project-taiwo-osunrinde-solo/main/cancer_model.pkl", 'model.pkl')
    # Try joblib/pickle for sklearn-like objects
    try:
        model = joblib.load('model.pkl')
        return model
    except Exception:
        pass

    try:
        with open('model.pkl', "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")


def infer_feature_names_from_model(model):
    # Many scikit-learn estimators expose feature_names_in_
    if hasattr(model, 'feature_names_in_'):
        return list(model.feature_names_in_)
    # If model is a pipeline, try to find the final estimator
    if hasattr(model, 'named_steps'):
        try:
            final = list(model.named_steps.values())[-1]
            if hasattr(final, 'feature_names_in_'):
                return list(final.feature_names_in_)
        except Exception:
            pass
    return None


def prepare_input_row():
    
    with st.form(key='single_input'):
        st.write("Enter feature values for prediction")
        age = st.number_input("Age", min_value=0.0, max_value=150.0, value=30.0, step=1.0)
        gender = st.selectbox("Gender", options=["Male", "Female"], index=0)
        bmi = st.number_input("BMI", min_value=0.0, max_value=1000.0, value=70.0, step=0.1)
        smoking = st.selectbox("Smoking", options=["Yes", "No"], index=0)
        geneticrisk = st.selectbox("GeneticRisk", options=[1, 2,3], index=0)
        physicalactivity = st.number_input("PhysicalActivity", min_value=0.0, max_value=1000.0, value=70.0, step=0.1)
        alcoholintake = st.number_input("AlcoholIntake", min_value=0.0, max_value=1000.0, value=70.0, step=0.1)
        cancerhistory = st.selectbox("CancerHistory", options=["Yes", "No"], index=0)
        submitted = st.form_submit_button("Predict")

    if not submitted:
        return None

    #
    g = str(gender).strip().lower()
    if g.startswith('m'):
        gender_code = 1
    else:
        gender_code = 0

    smoking_code = 1 if str(smoking).strip().lower().startswith('y') else 0

    cancerhistory_code = 1 if str(cancerhistory).strip().lower().startswith('y') else 0

    # Build DataFrame with standard column names (numeric-coded categories)
    df = pd.DataFrame([{
        'Age': float(age),
        'Gender': int(gender_code),
        'BMI': float(bmi),
        'Smoking': int(smoking_code),
        'GeneticRisk': int(geneticrisk),
        'PhysicalActivity': float(physicalactivity),
        'AlcoholIntake': float(alcoholintake),
        'CancerHistory': int(cancerhistory_code),
    }])
    return df


def predict_with_model(model, X: pd.DataFrame):
    # Ensure columns order if model expects feature_names_in_
    feature_names = infer_feature_names_from_model(model)
    if feature_names is not None:
        missing = [c for c in feature_names if c not in X.columns]
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        X = X[feature_names]

    result = X.copy()
    try:
        preds = model.predict(X)
        result['prediction'] = preds
    except Exception as e:
        raise RuntimeError(f"Model prediction failed: {e}")

    # Add probabilities if available
    if hasattr(model, 'predict_proba'):
        try:
            probs = model.predict_proba(X)
            # If binary, provide positive-class prob
            if probs.shape[1] == 2:
                result['probability_positive'] = probs[:, 1]
            else:
                # add each class probability as separate columns
                for i, cls in enumerate(getattr(model, 'classes_', range(probs.shape[1]))):
                    result[f'prob_{cls}'] = probs[:, i]
        except Exception:
            pass

    return result


def main():
    st.title("Cancer Prediction App")

    model = None
    model_path = os.path.join(os.getcwd(), 'model.pkl')

    # Sidebar: model uploader and optional CSV for batch predictions
    st.sidebar.markdown("## Uploads")
    
    uploaded_csv = st.sidebar.file_uploader("Upload CSV for prediction (optional)", type=['csv'])


    # Feature inputs
    st.write("Provide feature values below, or upload a CSV in the sidebar for batch predictions.")
    age = st.number_input("Age", min_value=0.0, max_value=150.0, value=30.0, step=1.0)
    gender = st.selectbox("Gender", options=["Male", "Female"], index=0)
    bmi = st.number_input("BMI", min_value=0.0, max_value=1000.0, value=70.0, step=0.1)
    smoking = st.selectbox("Smoking", options=["Yes", "No"], index=0)
    geneticrisk = st.selectbox("GeneticRisk", options=[1, 2,3], index=0)
    physicalactivity = st.number_input("PhysicalActivity", min_value=0.0, max_value=1000.0, value=70.0, step=0.1)
    alcoholintake = st.number_input("AlcoholIntake", min_value=0.0, max_value=1000.0, value=70.0, step=0.1)
    cancerhistory = st.selectbox("CancerHistory", options=["Yes", "No"], index=0)

    g = str(gender).strip().lower()
    if g.startswith('m'):
        gender_code = 1
    else:
        gender_code = 0

    smoking_code = 1 if str(smoking).strip().lower().startswith('y') else 0

    cancerhistory_code = 1 if str(cancerhistory).strip().lower().startswith('y') else 0

    # Predict only when the user clicks this button
    predict_clicked = st.button("Predict")

    if not predict_clicked:
        st.info("Fill inputs (or upload a CSV) then click Predict to run the model")
        return

    try:
        # Load the model on demand (downloads if missing)
        model = load_model()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return

    # Determine expected features from the model (may be None)
    feature_names = infer_feature_names_from_model(model)

    # If a CSV was uploaded in the sidebar, use it for batch predictions
    if uploaded_csv is not None:
        try:
            df = pd.read_csv(uploaded_csv)
        except Exception as e:
            st.error(f"Failed to read uploaded CSV: {e}")
            return

        # If the model exposes feature names, check for missing columns
        if feature_names is not None:
            missing = [c for c in feature_names if c not in df.columns]
            if missing:
                st.error(f"Uploaded CSV is missing required columns: {missing}")
                return
        try:
            result = predict_with_model(model, df)
            result.rename(columns={'prediction': 'Diagnosis'}, inplace=True)
            result['Diagnosis'] = result['Diagnosis'].map({0: 'No Cancer', 1: 'Has Cancer'})
            result['Gender'] = result['Gender'].map({0: 'Female', 1: 'Male'})
            result['Smoking'] = result['Smoking'].map({0: 'No', 1: 'Yes'})
            result['CancerHistory'] = result['CancerHistory'].map({0: 'No', 1: 'Yes'})
        except Exception as e:
            st.error(str(e))
            return

        st.subheader('Predictions (first 100 rows)')
        st.dataframe(result.head(100))
        csv = result.to_csv(index=False).encode('utf-8')
        st.download_button('Download predictions CSV', data=csv, file_name='predictions.csv', mime='text/csv')
        return

    # Otherwise, use the single-entry inputs
    input_df =pd.DataFrame([{
        'Age': float(age),
        'Gender': int(gender_code),
        'BMI': float(bmi),
        'Smoking': int(smoking_code),
        'GeneticRisk': int(geneticrisk),
        'PhysicalActivity': float(physicalactivity),
        'AlcoholIntake': float(alcoholintake),
        'CancerHistory': int(cancerhistory_code),
    }])


    try:
        result = predict_with_model(model, input_df)
        result.rename(columns={'prediction': 'Diagnosis'}, inplace=True)
        result['Diagnosis'] = result['Diagnosis'].map({0: 'No Cancer', 1: 'Has Cancer'})
        result['Gender'] = result['Gender'].map({0: 'Female', 1: 'Male'})
        result['Smoking'] = result['Smoking'].map({0: 'No', 1: 'Yes'})
        result['CancerHistory'] = result['CancerHistory'].map({0: 'No', 1: 'Yes'})
    except Exception as e:
        st.error(str(e))
        return

    st.subheader('Result')
    st.dataframe(result)
    csv = result.to_csv(index=False).encode('utf-8')
    st.download_button('Download result CSV', data=csv, file_name='prediction_single.csv', mime='text/csv')


if __name__ == '__main__':
    main()
