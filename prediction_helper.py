import pandas as pd
import joblib
model_young = joblib.load ('D:\CODEBASICS\project_1_build_an_app_using_streamlit_resources - MY PRACTICE\app\artifacts\model_young.joblib')
model_rest = joblib.load('D:/CODEBASICS/project_1_build_an_app_using_streamlit_resources - MY PRACTICE/app/artifacts/model_rest.joblib')
scaler_young = joblib.load('D:\CODEBASICS\project_1_build_an_app_using_streamlit_resources - MY PRACTICE\app\artifacts\scaler_young.joblib')
scaler_rest = joblib.load('D:/CODEBASICS/project_1_build_an_app_using_streamlit_resources - MY PRACTICE/app/artifacts/scaler_rest.joblib')

def calculate_normalized_risk(medical_history):
    risk_scores = {
        "diabetes": 6,
        "heart disease": 8,
        "high blood pressure": 6,
        "thyroid": 5,
        "no disease": 0,
        "none": 0
    }
    disease = medical_history.lower().split(" & ")
    total_risk_score = sum([risk_scores[i] for i in disease])
    max_score = 14
    min_score= 0
    normalized_risk_score = (total_risk_score - min_score) / (max_score - min_score)
    return normalized_risk_score


def preprocess_input(input_dict):
    expected_columns = [
        'age', 'number_of_dependants', 'income_lakhs', 'insurance_plan', 'genetical_risk', 'normalized_risk_score',
        'gender_Male', 'region_Northwest', 'region_Southeast', 'region_Southwest', 'marital_status_Unmarried',
        'bmi_category_Obesity', 'bmi_category_Overweight', 'bmi_category_Underweight', 'smoking_status_Occasional',
        'smoking_status_Regular', 'employment_status_Salaried', 'employment_status_Self-Employed'
    ]
    insurance_plan_encoding = {'Bronze': 1, 'Silver': 2, 'Gold': 3}
    df = pd.DataFrame(0,columns=expected_columns,index=[0])


    for key, value in input_dict.items():
        if key == 'Gender' and value == 'Male':
            df['gender_Male'] = 1
        elif key == 'Region':
            if value == 'Northwest':
                df['region_Northwest'] = 1
            elif value == 'Southeast':
                df['region_Southeast'] = 1
            elif value == 'Southwest':
                df['region_Southwest'] = 1
        elif key == 'Marital Status' and value == 'Unmarried':
            df['marital_status_Unmarried'] = 1
        elif key == 'BMI Category':
            if value == 'Obesity':
                df['bmi_category_Obesity'] = 1
            elif value == 'Overweight':
                df['bmi_category_Overweight'] = 1
            elif value == 'Underweight':
                df['bmi_category_Underweight'] = 1
        elif key == 'Smoking Status':
            if value == 'Occasional':
                df['smoking_status_Occasional'] = 1
            elif value == 'Regular':
                df['smoking_status_Regular'] = 1
        elif key == 'Employment Status':
            if value == 'Salaried':
                df['employment_status_Salaried'] = 1
            elif value == 'Self-Employed':
                df['employment_status_Self-Employed'] = 1
        elif key == 'Insurance Plan':  # Correct key usage with case sensitivity
            df['insurance_plan'] = insurance_plan_encoding.get(value, 1)
        elif key == 'Age':  # Correct key usage with case sensitivity
            df['age'] = value
        elif key == 'Number of Dependants':  # Correct key usage with case sensitivity
            df['number_of_dependants'] = value
        elif key == 'Income in Lakhs':  # Correct key usage with case sensitivity
            df['income_lakhs'] = value
        elif key == "Genetical Risk":
            df['genetical_risk'] = value
    df['normalized_risk_score'] = calculate_normalized_risk(input_dict['Medical History'])
    print("Dataframe before scaling: ",df.shape)
    df = handle_scaling(input_dict['Age'], df)
    print("Dataframe after scaling: ", df.shape)

    return df
def handle_scaling(age, df):
    if age <= 25:
        scaler_object = scaler_young
    else:
        scaler_object = scaler_rest
    scaler = scaler_object['scaler']
    col_to_scale = scaler_object['cols_to_scale']
    print(col_to_scale)
    df['income_level'] = None
    df[col_to_scale] = scaler.transform(df[col_to_scale])
    print(df[col_to_scale])
    print(scaler.transform(df[col_to_scale]))
    df = df.drop('income_level',axis=1)
    return df

def predict(input_dict):
    input_df = preprocess_input(input_dict)
    print(input_df.shape)
    print(input_df)

    if input_dict['Age'] <= 25:
        prediction = model_young.predict(input_df)
    else:
        prediction = model_rest.predict(input_df)
    return int(prediction)

