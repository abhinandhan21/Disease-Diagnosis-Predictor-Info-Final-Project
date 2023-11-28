import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv('kidney_disease.csv')
df = df.drop_duplicates()
df=df.dropna(how='any')  
X = df[['age', 'bp', 'al']]
Y = df['ane']
le = LabelEncoder()
le.fit_transform(Y)
model = LogisticRegression()
model.fit(X, Y)

age = st.number_input('Enter the age')
bp = st.number_input('Enter the BP')
al = st.number_input('Enter the al value')

pred = st.button('submit')

if pred:
    predict = model.predict([[age, bp, al]])
    if predict[0] == 0:
        st.write('Yes You have kidney issues')
    else:
        st.write("You dont have any kidney issues")
