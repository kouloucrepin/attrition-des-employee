import streamlit as st
import pandas as pd
import numpy as np
import os
from functions.mod1 import example
import joblib




st.set_page_config(layout="wide")
st.sidebar.image('images/sidebar.png')
st.markdown("<h1 style='text-align: left;color:white;'>"  + "💕👌Prédiction de l'Attrition : Anticiper les Départs pour Optimiser la Rétention des Employés" +" </h1>",unsafe_allow_html=True)

print(len(['Education', 'EnvironmentSatisfaction', 'Implication_dans_emploi',
       'JobLevel', 'Satisfaction_travail', 'Évaluation_performance',
       'Satisfaction_relationnelle', 'StockOptionLevel', 'WorkLifeBalance',
       'Voyage_affaires', 'Department', 'EducationField', 'Genre ', 'JobRole',
       'État_civil', 'Heures_supplémentaires', 'Age', 'DailyRate',
       'DistanceFromHome', 'HourlyRate', 'Revenu_mensuel', 'MonthlyRate',
       'NumCompaniesWorked', 'PercentSalaryHike', 'TotalWorkingYears',
       'TrainingTimesLastYear', 'YearsAtCompany', 'YearsInCurrentRole',
       'YearsSinceLastPromotion', 'YearsWithCurrManager']))

s5, s1,s2,s3,s4 = st.columns([4,1,1,1,1])

Education  = s1.selectbox('Education',[1,2,3,4,5])
EnvironmentSatisfaction_liste = ['faible','moyen','fort','tres fort']
EnvironmentSatisfaction = s1.selectbox('EnvironmentSatisfaction',EnvironmentSatisfaction_liste)
Implication_dans_emploi_liste = [1,2,3,4,5]
Implication_dans_emploi = s1.selectbox('Implication_dans_emploi',Implication_dans_emploi_liste)
JobLevel_liste = [1,2,3,4,5]
JobLevel = s1.selectbox('JobLevel',JobLevel_liste)
Satisfaction_travail_liste = [1,2,3,4]
Satisfaction_travail = s1.selectbox('Satisfaction_travail',Satisfaction_travail_liste)
Évaluation_performance_liste  = ['faible','bon','exelent','exceptionelle'] 
Évaluation_performance  = s1.selectbox('Satisfaction_travail',Évaluation_performance_liste)
Satisfaction_relationnelle_liste  = ['faible','moyen','eleve','tres eleve']
Satisfaction_relationnelle  = s1.selectbox('Satisfaction_relationnelle',Satisfaction_relationnelle_liste)
StockOptionLevel = s1.number_input('StockOptionLevel',min_value=0)
WorkLifeBalance = s2.number_input('WorkLifeBalance',min_value=0)
Voyage_affaires = s2.selectbox('Voyage_affaires',['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'])
Department = s2.selectbox('Department',['Sales', 'Research & Development', 'Human Resources'])
EducationField = s2.selectbox('EducationField',['Life Sciences', 'Other', 'Medical', 'Marketing','Technical Degree', 'Human Resources'])
Genre = s2.radio('Genre',['Female', 'Male'],horizontal=True)
JobRole = s2.selectbox('JobRole',['Sales Executive', 'Research Scientist', 'Laboratory Technician',
       'Manufacturing Director', 'Healthcare Representative', 'Manager',
       'Sales Representative', 'Research Director', 'Human Resources'])

État_civil = s2.selectbox('Etat_civil',['Single', 'Married', 'Divorced'])
Heures_supplémentaires = s2.radio('Heures_supplémentaires',['Yes', 'No'],horizontal=True)
Age = s3.number_input('Age',min_value=0)
DailyRate = s3.number_input('DailyRate',min_value=0)
DistanceFromHome = s3.number_input('DistanceFromHome',min_value=0.0)
HourlyRate = s3.number_input('HourlyRate',min_value=0)
Revenu_mensuel = s3.number_input('Revenu_mensuel',min_value=0)
MonthlyRate = s3.number_input('MonthlyRate',min_value=0)
NumCompaniesWorked = s3.number_input('NumCompaniesWorked',min_value=0)
PercentSalaryHike = s4.number_input('PercentSalaryHike',min_value=0.0,max_value=100.0)
TotalWorkingYears = s4.number_input('TotalWorkingYears',min_value=0)
TrainingTimesLastYear = s4.number_input('TrainingTimesLastYear',min_value=0.0)
YearsAtCompany = s4.number_input('YearsAtCompany',min_value=0)
YearsInCurrentRole = s4.number_input('YearsInCurrentRole',min_value=0)
YearsSinceLastPromotion = s4.number_input('YearsSinceLastPromotion',min_value=0)
YearsWithCurrManager = s4.number_input('YearsWithCurrManager',min_value=0)
s4.markdown('<br>',unsafe_allow_html=True)
but = s4.button('Predire l\'Attrition',type="primary")
model = s3.selectbox('select a model',os.listdir('models'))

vect = [Education,
        EnvironmentSatisfaction_liste.index(EnvironmentSatisfaction)+1,
        Implication_dans_emploi,
        JobLevel,
        Satisfaction_travail,
 Évaluation_performance_liste.index(Évaluation_performance)+1
 ,Satisfaction_relationnelle_liste.index(Satisfaction_relationnelle)+1,
 StockOptionLevel
 ,WorkLifeBalance
 ,Voyage_affaires
 ,Department
 ,EducationField
 ,Genre
 ,JobRole
 ,État_civil
 ,Heures_supplémentaires
 ,Age,
DailyRate,
DistanceFromHome,
HourlyRate,
Revenu_mensuel,
MonthlyRate,
NumCompaniesWorked
,PercentSalaryHike,
TotalWorkingYears,
TrainingTimesLastYear,
YearsAtCompany,
YearsInCurrentRole,
YearsSinceLastPromotion,
YearsWithCurrManager]
models = joblib.load(f'models/{model}') 

liste=['Cet employee est eligible pour rester encore dans l\'entreprise','Cet employee est succeptible de  quitter l\'entreprise']
pred = pd.DataFrame(vect).T.astype(example.dtypes.values)
pred.columns = example.columns
pred = pred.rename(columns={'Genre':'Genre\xa0'})
for i,j in enumerate(pred.columns):
    pred[j] = pred[j].astype(example.dtypes.values[i])


if but:
       pourcent = str(round(models.predict_proba(pred)[0][int(models.predict(pred))]*100,2)) +' %'
       s5.markdown("<h1 style='text-align: center;color:green;'>" +liste[int(models.predict(pred))] + ' Avec une probabilite de ' + pourcent+ "</h1>",unsafe_allow_html=True)
       if int(models.predict(pred)) ==0:
           s5.image('images/rester.png',width=550)
       else:
            s5.image('images/quitter.png',width=550)  


