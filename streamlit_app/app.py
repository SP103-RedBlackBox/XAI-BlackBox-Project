import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def main():
    st.title("Income Prediction Dashboard", text_alignment='center')
    st.header("Input Data Below or Upload a File to Predict Annual Salary", text_alignment='center')

    user_input = st.toggle("Directly input data values")
    df = None

    if user_input:
        left_column, right_column = st.columns(2)

        #age,workclass,education.num,marital.status,occupation,relationship,race,sex,capital.gain,capital.loss,hours.per.week,native.country
        with left_column:
            st.number_input("Age", value=None, placeholder='Enter age', step=1, key="age")
            st.selectbox(
                "Workclass",
                ('Self-emp-inc','Self-emp-not-inc','Private','Local-gov','State-gov','Federal-gov','Without-pay','Never-worked','Other'),
                index=None,
                key='workclass'
            )
            st.number_input("Years of Education", value=None, placeholder='Enter age', step=1, key="education_num")
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
            st.number_input("Captial Gain", value=None, placeholder='Enter capital age', step=1, key="capital_gain")
            st.number_input("Capital Loss", value=None, placeholder='Enter capital loss', step=1, key="capital_loss")
            st.number_input("Hours Per Week", value=None, placeholder='Enter hours per week', step=1,key="hours_per_week")
            st.selectbox(
                "Native Country", 
                sorted(['United States','Mexico','Greece','Vietnam','China','Taiwan','India','Laos','Thailand','Germany','United Kingdom']),
                index=None,
                key="native_country",
                accept_new_options=True
            )

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

        df = pd.DataFrame(data, index=[0])
    
    file = st.file_uploader("Upload a .csv file", type='csv')

    shap_left_column, lime_right_column = st.columns(2)
    shap_left_column.title('SHAP Prediction')
    lime_right_column.title('LIME Prediction')

if __name__ == "__main__":
    main()    