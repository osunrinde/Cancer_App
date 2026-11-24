# Cancer Prediction App

A Streamlit interface for running a machine-learning model on cancer-related features. The app supports a single-record prediction entered through the form or batch predictions from an uploaded CSV. Results can be downloaded as CSV files.

> **Important:** This is an experimental software project, not a medical device or diagnostic tool. Predictions are not medical advice and should not be used to diagnose, screen, or guide treatment. Consult a qualified healthcare professional for medical decisions.

## Features

- Enter one record using the app's input controls.
- Upload a CSV for batch prediction.
- Load a scikit-learn-compatible model from a local `model.pkl`, or download the configured model when it is missing.
- Use feature names exposed by the model to validate and order input columns.
- Display prediction results and available class probabilities.
- Download results as CSV.

## Input features

The single-record form collects these fields:

| Feature | Input |
|---|---|
| `Age` | Numeric |
| `Gender` | Male or Female; encoded as 1 or 0 |
| `BMI` | Numeric |
| `Smoking` | Yes or No; encoded as 1 or 0 |
| `GeneticRisk` | 1, 2, or 3 |
| `PhysicalActivity` | Numeric |
| `AlcoholIntake` | Numeric |
| `CancerHistory` | Yes or No; encoded as 1 or 0 |

For batch prediction, the CSV must contain the feature columns expected by the model. If the model exposes `feature_names_in_`, the app checks that those columns are present and orders them accordingly. Ensure categorical values in the CSV use the numeric encoding expected by the model.

## Requirements

- Python 3.11 or compatible
- Streamlit
- pandas
- scikit-learn
- joblib

Install dependencies from the repository root:

```bash
pip install -r requirements.txt
```

## Run locally

From the repository root, start the app with:

```bash
streamlit run Cancer_prediction_App.py
```

Streamlit prints a local URL, typically `http://localhost:8501`.

## Using the app

### Single prediction

1. Open the app in your browser.
2. Enter the requested feature values.
3. Click **Predict**.
4. Review the result and download `prediction_single.csv` if needed.

### Batch prediction

1. Prepare a CSV with the model's required feature columns.
2. Upload it using the sidebar file uploader.
3. Click **Predict** to run the batch.
4. Review the displayed results and download `predictions.csv`.

The app displays up to the first 100 batch results in the interface; the download contains the full result table.

## Model

The app looks for `model.pkl` in its working directory. If the file is absent, it downloads a model from the URL configured in `Cancer_prediction_App.py`. It tries to load the model with `joblib` first and falls back to Python `pickle`.

For reproducible or production use, provide a trusted model file locally and verify the model's provenance and compatibility. **Pickle and joblib model files can execute code when loaded; only load model files from trusted sources.**

The app maps binary predictions `0` and `1` to `No Cancer` and `Has Cancer`. If the model supports `predict_proba`, the output also includes probability columns.

## Project files

```text
Cancer_App/
├── Cancer_prediction_App.py  # Streamlit application
├── requirements.txt          # Python dependencies
├── README.md
└── .devcontainer/
    └── devcontainer.json     # Development container configuration
```

## License

No license file is currently included in this repository. Check with the repository owner before using, modifying, or distributing the project.
