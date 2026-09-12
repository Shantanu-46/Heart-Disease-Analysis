import streamlit as st
import pandas as pd
import joblib 

model =  joblib.load('KNN_heart.pkl')
scaler = joblib.load('scaler.pkl')
expected_column = joblib.load('column.pkl')

st.title("heart stroke predication by 🫀")
st.markdown("provide the following detail")

age = st.slider("age",18,100,40)
sex = st.selectbox("SEX",['M','F'])
chest_pain = st.selectbox("chest pain type", ["ATA","NAP","TA","ASY"])
Resting_BP = st.number_input("Resting_Blood pressure (mm hg)",80,200,120 )
Cholesterol = st.number_input("Cholesterol (mg/dl)" ,100,600,200)
FastingBS = st.selectbox("Fasting blood suger > 120 mg/dl",[0,1] )
Resting_ECG = st.selectbox("RestingECG", ["normal", "ST","LVH"])
max_hr = st.slider("max heart Rate", 60 ,220,150)
Exercise_Angina = st.selectbox("Exercise- included Angina",['Y','N'])
Oldpeak = st.slider("Oldpeak (ST Depression)",0.0,6.0,1.0)
st_slope = st.selectbox ("ST slope", ['up','Flat','Down'])


  # Create a raw input dictionary
if st.button("predict"):
    raw_input = {
        'Age': age,
        'RestingBP': Resting_BP,
        'Cholesterol': Cholesterol,
        'FastingBS': FastingBS,
        'MaxHR': max_hr,
        'Oldpeak': Oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + Resting_ECG: 1,
        'ExerciseAngina_' + Exercise_Angina: 1,
        'ST_Slope_' + st_slope: 1
    }
    input_df = pd.DataFrame([raw_input])

    for col in expected_column:
        if col  not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_column]   

    scaled_input = scaler.transform(input_df)    
    prediction = model.predict(scaled_input)[0]

    if prediction== 1 :
        st.error("💀 High Risk of Heart Disease")
    else:
        st.success("😇 Low Risk of Heart Disease")




