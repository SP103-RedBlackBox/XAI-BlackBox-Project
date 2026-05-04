# XAI Black-Box Model Visualizer

An interactive Explainable AI (XAI) web application that predicts whether an individual earns more or less than $50,000 per year and provides transparent, interpretable explanations for each prediction using SHAP and LIME.

## Live Demo
[sp103blackbox.streamlit.app](https://sp103blackbox.streamlit.app)

## Team
| Name | Role |
|---|---|
| Shamir Howlader | Team Lead |
| Phillip Pham | Dev/Doc |
| Dang Tran | Dev/Doc |
| Sai Samudrala | Dev/Doc |

## Overview
This project trains an XGBoost binary classifier on the UCI Adult Census Income dataset and wraps it in a Streamlit web interface. Users can input data manually or upload a CSV file for batch predictions. The dashboard provides prediction confidence scores and explainability visualizations using SHAP and LIME to make the model's decisions transparent and interpretable.

## Features
- Manual single-entry prediction with confidence scores
- Batch CSV upload for multiple predictions
- SHAP explanations: Summary Plot, Waterfall Plot, Force Plot
- LIME local explanations with feature weight breakdown
- Progress bar for batch prediction loading
- Informative error handling for invalid inputs

## Setup & Installation

**1. Clone the repository**
```bash
git clone https://github.com/SP103-RedBlackBox/XAI-BlackBox-Project.git
cd XAI-BlackBox-Project
```

**2. Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the Streamlit app**
```bash
cd streamlit_app
streamlit run app.py
```

## Usage

### Manual Entry
1. Select "Direct value input"
2. Fill in the form fields (age, workclass, education, etc.)
3. Click "Submit data"
4. Click "Predict Salary" to see the prediction and explanations

### CSV Upload
1. Select "Upload .csv file"
2. Upload a CSV file matching the required feature schema
3. Click "Predict Salary" to see batch predictions and explanations
