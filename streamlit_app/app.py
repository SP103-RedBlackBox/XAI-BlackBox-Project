import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import shap
import lime
import time
from pathlib import Path
# Add project root to path
sys.path.append(str(Path().resolve().parent))
from src import explainability_methods

# Initialize SHAP
shap.initjs()

def info_box(text):
    st.markdown(f"""
        <div style="
            background-color: #1a1f2e;
            border: 1px solid #2d3550;
            border-radius: 10px;
            padding: 16px 24px 16px 24px;
            margin: 12px 0;
            color: #d0d4e0;
            font-size: 0.875rem;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto;
            line-height: 1.7;
            overflow-wrap: break-word;
            word-break: break-word;
            white-space: normal;
            display: block;
            width: 100%;
            box-sizing: border-box;
        ">
            {text.strip()}
        </div>
    """, unsafe_allow_html=True)

def main():
    st.markdown("""
        # Welcome to the Income Prediction Dashboard!

        This interactive tool predicts whether an individual is likely to earn **more or less than $50,000 per year** based on the information provided.

        ### What you can do:
        - **Enter data manually** for a single prediction  
        - **Upload a CSV file** to generate predictions in bulk  

        ### Explainable AI Insights:
        In addition to predictions, the dashboard provides:

        - **SHAP explanations** to show how each feature impacts the prediction
        - **LIME explanations** to highlight the most influential factors for individual cases  

        Our goal is to make the model’s decisions more **transparent, interpretable, and trustworthy**.
        """)
    st.divider()
    st.subheader("Input Data Below or Upload a File to Predict Salary >50k or <=50k", text_alignment='center')

    # Data upload choice
    data_choice = st.radio(
        "Choose which method to input data",
        ['Direct value input','Upload .csv file']
    )
    df = None

    # Reset session state if user changes data input method
    if 'last_data_choice' not in st.session_state:
        st.session_state.last_data_choice = data_choice

    if st.session_state.last_data_choice != data_choice:
        st.session_state.df = None
        st.session_state.results_ready = False
        st.session_state.last_data_choice = data_choice

    # User input direct data values
    if data_choice == 'Direct value input':
        left_column, right_column = st.columns(2)

        with st.form('Input Data'):
            # age, workclass, education.num, marital.status, occupation, relationship, race, sex, capital.gain, capital.loss, hours.per.week, native.country
            with left_column:
                st.number_input("Age", min_value=0, value="min", placeholder='Enter age', step=1, key="age")
                st.selectbox(
                    "Workclass",
                    ('Self-emp-inc','Self-emp-not-inc','Private','Local-gov','State-gov','Federal-gov','Without-pay','Never-worked','Other'),
                    index=None,
                    key='workclass'
                )
                st.number_input("Years of Education", min_value=0, value="min", placeholder='Enter age', step=1, key="education_num")
                st.selectbox(
                    "Marital Status", 
                    ('Never-married','Married-civ-spouse','Married-spouse-absent','Married-AF-spouse','Divorced','Separated','Widowed'),
                    index=None,
                    key="marital_status",
                )
                st.text_input("Occupation", key="occupation", placeholder='Enter occupation')
                st.selectbox(
                    "Relationship", 
                    ('Not-in-family','Unmarried','Own-child','Other-relative','Husband','Wife'),
                    index=None,
                    key="relationship",
                )
            with right_column:
                st.selectbox(
                    "Race", 
                    ('White','Black','Asian-Pac-Islander','Amer-Indian-Eskimo','Other'),
                    index=None,
                    key="race",
                )
                st.selectbox(
                    "Sex", 
                    ('Male','Female'),
                    index=None,
                    key="sex",
                )
                st.number_input("Captial Gain", min_value=0, value="min", placeholder='Enter capital age', step=1, key="capital_gain")
                st.number_input("Capital Loss", min_value=0, value="min", placeholder='Enter capital loss', step=1, key="capital_loss")
                st.number_input("Hours Per Week", min_value=0, value="min", placeholder='Enter hours per week', step=1,key="hours_per_week")
                st.selectbox(
                    "Native Country", 
                    sorted(['United States','Mexico','Greece','Vietnam','China','Taiwan','India','Laos','Thailand','Germany','United Kingdom']),
                    index=None,
                    key="native_country",
                    accept_new_options=True
                )

            submit = st.form_submit_button(label="Submit data")

        if submit:
            # Create dataframe from user inputted data
            data = {
                'age': st.session_state.age,
                'workclass': st.session_state.workclass,
                'education.num': st.session_state.education_num,
                'marital.status': st.session_state.marital_status,
                'occupation': st.session_state.occupation,
                'relationship': st.session_state.relationship,
                'race': st.session_state.race,
                'sex': st.session_state.sex,
                'capital.gain': st.session_state.capital_gain,
                'capital.loss': st.session_state.capital_loss,
                'hours.per.week': st.session_state.hours_per_week,
                'native.country': st.session_state.native_country
            }
            st.session_state.df = pd.DataFrame(data, index=[0])
        
        df = st.session_state.get('df', None)

    # User csv file upload
    else:
        file = st.file_uploader("Upload a .csv file", type='csv')

        # Verify file exists, assume file is in the same format as trained csv
        if file is not None:
            # Load dataset
            df = pd.read_csv(file)

            # Drop 'fnlwgt' column as it's not needed for analysis
            df.drop('fnlwgt', axis=1, inplace=True, errors='ignore')

            # Replace '?' with NaN
            df.replace('?', np.nan, inplace=True)

            st.session_state.df = df

        df = st.session_state.get('df', None)        

    if df is not None:
        st.subheader("View Data:")
        st.dataframe(df)
        st.divider()

    # Predict button
    if st.button("Predict Salary"):
        if df is None:
            st.error("Please input data or upload a file before making predictions.")
            return
        else:
            xgb_model, preprocessor = explainability_methods.load_model_preprocessor()
            processed_data = preprocessor.transform(df).toarray()

            progress_bar = st.progress(0, text="Loading model...")
                
            st.session_state.processed_data = processed_data
            st.session_state.xgb_model = xgb_model
            st.session_state.preprocessor = preprocessor
            progress_bar.progress(10, text="Making predictions...")

            st.session_state.prediction = xgb_model.predict(processed_data)
            st.session_state.prediction_proba = xgb_model.predict_proba(processed_data)
            progress_bar.progress(30, text="Computing SHAP values...")

            st.session_state.SHAP_explainer = explainability_methods.load_SHAP_explainer(xgb_model, preprocessor, processed_data)
            st.session_state.shap_values = explainability_methods.get_SHAP_values(st.session_state.SHAP_explainer, processed_data)
            progress_bar.progress(70, text="Computing LIME explanations...")

            st.session_state.LIME_explainer = explainability_methods.load_LIME_explainer(xgb_model, preprocessor)
            progress_bar.progress(100, text="Done!")

            st.session_state.results_ready = True
            progress_bar.empty()  # removes the bar once complete

    # Render results outside the button block
    if st.session_state.get('results_ready', False):
        st.subheader("Model Predictions:")

        # Display predictions for single instance vs multiple instances differently
        if df.shape[0] == 1:
            pred_label = '>50K' if st.session_state.prediction[0] == 1 else '<=50K'
            prob_low = st.session_state.prediction_proba[0][0] * 100
            prob_high = st.session_state.prediction_proba[0][1] * 100
    
            st.write(f"The model predicts the individual makes {pred_label} annually.")
    
            col1, col2 = st.columns(2)
            col1.metric("Probability <=50K", f"{prob_low:.1f}%")
            col2.metric("Probability >50K", f"{prob_high:.1f}%")

        else:
            st.write("Predictions for each individual:")
            prob_low = st.session_state.prediction_proba[:, 0] * 100
            prob_high = st.session_state.prediction_proba[:, 1] * 100
            pred_labels = np.where(st.session_state.prediction == 1, '>50K', '<=50K')
    
            results_df = pd.DataFrame({
            'Prediction': pred_labels,
            'Probability <=50K': prob_low.round(1).astype(str) + '%',
            'Probability >50K': prob_high.round(1).astype(str) + '%'
            })
            st.dataframe(results_df)    

        # Create SHAP and LIME tabs
        shap_tab, lime_tab = st.tabs(["SHAP Explanation", "LIME Explanation"])

        with shap_tab:
            st.header("SHAP Explanations")
            shap_plot_choice = st.selectbox(
                "Choose which SHAP plot to display",
                ['Summary Plot', 'Waterfall Plot', 'Force Plot']
            )
            if shap_plot_choice == 'Summary Plot':
                explainability_methods.SHAP_summary_plot(st.session_state.shap_values, st.session_state.processed_data, st.session_state.preprocessor.get_feature_names_out())

                info_box("""
                    The SHAP sumamary plot shows the impact of each feature on the model's predictions across all instances. Each point represents a SHAP value for a feature and instance, with color indicating the feature value (red = high, blue = low). The plot helps identify which features are most influential and how they affect the prediction direction (positive or negative).
                    """)

            elif shap_plot_choice == 'Waterfall Plot':
                index = st.number_input("Enter index of instance to explain", min_value=0, max_value=st.session_state.processed_data.shape[0]-1, step=1)
                explainability_methods.SHAP_waterfall_plot(st.session_state.shap_values, index)

                info_box("""
                    The SHAP Waterfall plot provides a detailed explanation for a single individual. It shows how each feature contributes to pushing the prediction from the base value (average prediction) to the final predicted probability for that instance. Features that increase the prediction are showin in red, while those that decrease it are shown in blue. The length of the bars indicates the magnitude of the contribution. Keep in mind that the base value is the average model output across the training data, so the plot illustrates how the specific feature values of the instance influence its prediction compared to an average case. Thus, there can be instances where all features push towards a higher income (red bars) but the prediction is still <=50K if the base value is low enough and the contributions are not strong enough to push it over the threshold (>50% probability).        
                    """)

            elif shap_plot_choice == 'Force Plot':
                index = st.number_input("Enter index of instance to explain", min_value=0, max_value=st.session_state.processed_data.shape[0]-1, step=1)
                explainability_methods.SHAP_force_plot(st.session_state.shap_values, index)

                info_box("""
                    The SHAP Force plot is another way to visualize the contribution of each feature for a single instance, similar to the waterfall plot but in a more compact form. It shows how the features push the prediction from the base value to the final output, with features pushing towards higher income shown in red and those pushing towards lower income shown in blue. The length of the arrows indicates the magnitude of the contribution. Like the waterfall plot, it’s important to remember that the base value is the average model output, so the plot illustrates how the specific feature values of the instance influence its prediction compared to an average case. This means that even if all features push towards a higher income (red arrows), the prediction can still be <=50K if the base value is low enough and the contributions are not strong enough to push it over the threshold (>50% probability).
                    """)

        with lime_tab:
            st.header("LIME Explanation")
            index = st.number_input("Enter index of instance to explain", min_value=0, max_value=st.session_state.processed_data.shape[0]-1, step=1, key="lime_index")
            explainability_methods.LIME_explanation(st.session_state.LIME_explainer, index, st.session_state.processed_data, st.session_state.xgb_model)

            info_box("""
                The LIME explanation highlights the most influential features for a specific instance by showing how they contribute to the prediction. It provides a local approximation of the model's behavior around that instance, indicating which features push the prediction towards higher or lower income. The feature weights indicate the strength of their influence, with positive weights pushing towards >50K and negative weights pushing towards <=50K. The accompanying table shows the exact feature contributions in terms of weights, providing a clear view of which factors were most important for that specific prediction.
                 """)


if __name__ == "__main__":
    main()    