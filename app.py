import streamlit as st
import pandas as pd
import numpy as np


st.title("Disease & Diagnosis Predictor")


l1=['Select_Symptom','back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs',
'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',
'swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',
'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine',
'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
'abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum',
'rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion',
'receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen',
'history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf',
'palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose',
'yellow_crust_ooze']

sym1 = st.selectbox('Symptom 1', l1)
sym2 = st.selectbox('Symptom 2', l1)
sym3 = st.selectbox('Symptom 3', l1)
sym4 = st.selectbox('Symptom 4', l1)
sym5 = st.selectbox('Symptom 5', l1)

disease=['Fungal infection','Allergy','GERD','Chronic cholestasis','Drug Reaction',
'Peptic ulcer diseae','AIDS','Diabetes','Gastroenteritis','Bronchial Asthma','Hypertension',
' Migraine','Cervical spondylosis',
'Paralysis (brain hemorrhage)','Jaundice','Malaria','Chicken pox','Dengue','Typhoid','hepatitis A',
'Hepatitis B','Hepatitis C','Hepatitis D','Hepatitis E','Alcoholic hepatitis','Tuberculosis',
'Common Cold','Pneumonia','Dimorphic hemmorhoids(piles)',
'Heartattack','Varicoseveins','Hypothyroidism','Hyperthyroidism','Hypoglycemia','Osteoarthristis',
'Arthritis','(vertigo) Paroymsal  Positional Vertigo','Acne','Urinary tract infection','Psoriasis',
'Impetigo']

l2=[]

for i in range(0,len(l1)):
    l2.append(0)

df=pd.read_csv("disease_dataset.csv")

df.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40}},inplace=True)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report ,confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import KMeans

from sklearn.model_selection import StratifiedKFold
def best_score(p):
    n=list(range(len(df)))
    y=df['prognosis']
    skf=StratifiedKFold(n_splits=4,shuffle=True)
    k=[]
    l=[]
    for train,test in skf.split(n,y):
        x_tr=df.loc[train,l1]
        y_tr=df.loc[train,'prognosis']
        x_te=df.loc[test,l1]
        y_te=df.loc[test,'prognosis']
        l.append([train,test])

        if p==1:
            model=DecisionTreeClassifier()
            model.fit(x_tr,y_tr)
            k.append(model.score(x_te,y_te))
        elif p==2:
            model=RandomForestClassifier()
            model.fit(x_tr,y_tr)
            k.append(model.score(x_te,y_te))

        elif p==3:
            model=MultinomialNB()
            model.fit(x_tr,y_tr)
            k.append(model.score(x_te,y_te))
            
        elif p==4:
            model=KMeans()
            model.fit(x_tr,y_tr)
            k.append(model.score(x_te,y_te))    
    pos=np.argmax(np.array(k))
    #print(pos,k[pos])
    return l[pos]

def DecisionTree():
    ls=best_score(1)
    x_tr=df.loc[ls[0],l1]
    y_tr=df.loc[ls[0],'prognosis']
    x_te=df.loc[ls[1],l1]
    y_te=df.loc[ls[1],'prognosis']
    clf3=DecisionTreeClassifier(criterion='gini',max_depth=32)
    clf3.fit(x_tr,y_tr)
    print('Desion Tree Clasifier')
    #print('Training Accuracy',clf3.score(x_tr,y_tr))
    print('Testing Accuracy',clf3.score(x_te,y_te),end='\n\n')
    psymptoms = [sym1,sym2,sym3, sym4, sym5]
    print(psymptoms)
    count=0
    for i in range(5):
        if psymptoms[i]=='Select Here':
            count+=1
    else:
        for k in range(0,len(l1)):
            for z in psymptoms:
                if(z==l1[k]):
                    l2[k]=1

        x_input = [l2]
        predict = clf3.predict(x_input)
        print(predict)
        return disease[predict[0]]

def randomforest():
    ls=best_score(2)
    x_tr=df.loc[ls[0],l1]
    y_tr=df.loc[ls[0],'prognosis']
    x_te=df.loc[ls[1],l1]
    y_te=df.loc[ls[1],'prognosis']
    clf4=RandomForestClassifier(criterion='gini',n_estimators=65,max_depth=4)
    clf4.fit(x_tr,y_tr)
    print('Random Forest Classifier')
    #print('Training Accuracy',clf4.score(x_tr,y_tr))
    print('Testing Accuracy',clf4.score(x_te,y_te),end='\n\n')
    psymptoms = [sym1, sym2, sym3, sym4, sym5]
    print(psymptoms)
    for k in range(0,len(l1)):
        for z in psymptoms:
            if(z==l1[k]):
                l2[k]=1
    count=0
    for i in range(5):
        if psymptoms[i]=='Select Here':
            count+=1
    else:
        x_input = [l2]
        predict = clf4.predict(x_input)
        print(predict)
        return disease[predict[0]]

def randomforest():
    ls=best_score(2)
    x_tr=df.loc[ls[0],l1]
    y_tr=df.loc[ls[0],'prognosis']
    x_te=df.loc[ls[1],l1]
    y_te=df.loc[ls[1],'prognosis']
    clf4=RandomForestClassifier(criterion='gini',n_estimators=65,max_depth=4)
    clf4.fit(x_tr,y_tr)
    print('Random Forest Classifier')
    #print('Training Accuracy',clf4.score(x_tr,y_tr))
    print('Testing Accuracy',clf4.score(x_te,y_te),end='\n\n')
    psymptoms = [sym1, sym2, sym3, sym4, sym5]
    print(psymptoms)
    for k in range(0,len(l1)):
        for z in psymptoms:
            if(z==l1[k]):
                l2[k]=1
    count=0
    for i in range(5):
        if psymptoms[i]=='Select Here':
            count+=1
    else:
        x_input = [l2]
        predict = clf4.predict(x_input)
        print(predict)
        return disease[predict[0]]

pred1 = st.button('Prediction 1')
pred2 = st.button('Prediction 2')
pred3 = st.button('Prediction 3')

if pred1:
    preda = randomforest()
    st.write(preda)
if pred2:
    predb = DecisionTree()
    st.write(predb)

if pred3:
    predc = randomforest()
    st.write(predc)