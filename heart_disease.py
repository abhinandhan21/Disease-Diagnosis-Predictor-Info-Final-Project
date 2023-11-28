import streamlit as st
import joblib
gender_map = {'male':0,
              'female':1}
cpt_map = {
    "Typical Angina":0,
    "Atypical Angina":1,
    "Non-Anginal Pain":2,
    "Asymptomatic":3
}
bloodsugar_map = {
    "Less than 120 mg/dl":0,
    "Greater than or equal to 120 mg/dl":1
}
restecg_map = {
    "Normal":0,
    "ST-T Wave Abnormality":1,
    "Left Ventricular Hypertrophy":2
}
eia_map = {
    'yes':0,
    'no':1
}
slp_map = {
    'Upsloping':0,
    'Flat':1,
    'Downsloping':2
}
caa_map = {
    '0':0,
    '1':1,
    '2':2,
    '3':3,
    '4':4
}
thall_map = {
    'Normal':0,
    'Fixed Defect':1,
    'Reversible Defect':2
}
age = st.number_input("Enter The age")
gender = st.selectbox("Enter your gender", ['male', 'female'])
cpt = st.selectbox("Enter the chest pain type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
rbp = st.number_input("Enter Blood Pressure")
bloodsugar = st.selectbox("Fasting blood sugar", ["Less than 120 mg/dl", "Greater than or equal to 120 mg/dl"])
restecg = st.selectbox("Resting ECG result", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
hra = st.number_input("Heart Rate Achieved")
chl = st.number_input("Cholesterol")
eia = st.selectbox("Exercise Induces Angina", ['yes', 'no'])
oldpeak = st.number_input("ST Depression Induced by Exercise Relative to Rest:")
slp = st.selectbox("Slope of the Peak Exercise ST Segment", ['Upsloping', 'Flat', 'Downsloping'])
caa = st.selectbox("Number of Major Vessels Colored by Flourosopy", ['0','1','2','3','4'])
thall = st.selectbox("Thallasemia:", ['Normal', 'Fixed Defect', 'Reversible Defect'])
pred = st.button("Submit")

if pred:
    model = joblib.load('heart.joblib')
    gender = gender_map[gender]
    cpt = cpt_map[cpt]
    bloodsugar = bloodsugar_map[bloodsugar]
    restecg = restecg_map[restecg]
    eia = eia_map[eia]
    slp = slp_map[slp]
    caa = caa_map[caa]
    thall = thall_map[thall]
    res = model.predict([[age, gender, cpt, rbp, bloodsugar, restecg, hra, chl, eia, oldpeak,slp, caa, thall]])
    if res[0] == 0:
        st.write('Yes')
    else:
        st.write('NO')