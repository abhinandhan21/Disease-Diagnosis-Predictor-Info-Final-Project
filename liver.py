import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.svm import SVC
df4=pd.read_csv('indian_liver_patient.csv')
df4 = df4.drop_duplicates()
df4=df4.dropna(how='any')  

df4 = df4.drop(['Age','Gender'],axis=1)
y=df4.Dataset
x=df4.drop(['Dataset'],axis=1)
X_train, X_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size=0.2, 
                                                    random_state=0,
                                                    stratify=df4.Dataset)
model4 = SVC()
model4.fit(X_train,y_train)


st.header("Know If You Are Affected By Liver")
st.write("All The Values Should Be In Range Mentioned")

total_bilirubin=st.number_input("Enter Your total_bilirubin (0.4-75)",min_value=0,max_value=75,step=1)
direct_bilirubin=st.number_input("Enter direct bilirubin (0-20)",min_value=0,max_value=20,step=1)
alkaline_phosphotase=st.number_input("Enter alkaline_phosphotase (60-2200)",min_value=60,max_value=2200,step=1)
almine_aminotransferase=st.number_input("Enter Your Blood Pressure Rate (10-2000)",min_value=10,max_value=2000,step=1)
aspartate_aminotransferase=st.number_input("aspartate aminotransferase (10-5000)",min_value=10,max_value=5000,step=1)
total_protein=st.number_input("total protein (2-10)",min_value=2,max_value=10,step=1)
albumin=st.number_input("albumin",min_value=0,max_value=6,step=1)
albumin_and_globulin_ratio=st.number_input("Enter ratio ",min_value=0,max_value=3,step=1)
    
prediction2=model4.predict([[total_bilirubin,direct_bilirubin,alkaline_phosphotase,almine_aminotransferase,aspartate_aminotransferase,total_protein,albumin,albumin_and_globulin_ratio]])[0]
    
    #prediction part predicts whether the person is affected by Diabetes or not by the help of features taken as input
    #on the basis of prediction the results are displayed
if st.button("Predict"):
    if prediction2==1:
        st.warning("You Might Be Affected By Liver disease")
    elif prediction2==2:
        st.success("You Are Safe")
    