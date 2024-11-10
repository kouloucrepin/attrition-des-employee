import streamlit as st
import pandas as pd
import numpy as np
from streamlit_extras.metric_cards import style_metric_cards
from functions.mod1 import *
import plotly.subplots as sp


st.sidebar.image('images/sidebar.png')
data = pd.read_excel('data/ml.xlsx')
data.colums=data.columns.str.strip()
data['DistanceFromHome'] = data['DistanceFromHome'].fillna(np.median(data['DistanceFromHome'].dropna()))
Education_codes = {1: 'inférieur au collège', 2: 'collège', 3: 'licence', 4: 'master', 5: 'docteur'}
EnvironmentSatisfaction_codes = {1: 'faible', 2: 'moyen', 3: 'élevée', 4: 'très élevée'}
EvaluationPerformance_codes = {1: 'faible', 2: 'bon', 3: 'excellent', 4: 'exceptionnel'}
ImplicationDansEmploi_codes = {1: 'très peu impliqué', 2: 'peu impliqué', 3: 'impliqué', 4: 'très impliqué', 5: 'exceptionnellement impliqué'}
JobLevel_codes = {1: 'bas', 2: 'intermédiaire', 3: 'supérieur', 4: 'haut', 5: 'exceptionnel'}
SatisfactionRelationnelle_codes = {1: 'faible', 2: 'moyen', 3: 'élevée', 4: 'très élevée'}
SatisfactionTravail_codes = {1: 'faible', 2: 'moyen', 3: 'élevée', 4: 'très élevée'}
StockOptionLevel_codes = {0: "pas d'option", 1: 'standard', 2: 'élevé', 3: 'exceptionnel '}
WorkLifeBalance_codes = {1: 'mauvais', 2: 'bon', 3: 'excellent', 4: 'très élevé'}

Data_ = data.copy()
Data_.Education = Data_.Education.replace(Education_codes)
Data_.EnvironmentSatisfaction = Data_.EnvironmentSatisfaction.replace(EnvironmentSatisfaction_codes)
Data_.Évaluation_performance = Data_.Évaluation_performance.replace(EvaluationPerformance_codes)
Data_.Implication_dans_emploi = Data_.Implication_dans_emploi.replace(ImplicationDansEmploi_codes)
Data_.JobLevel = Data_.JobLevel.replace(JobLevel_codes)
Data_.Satisfaction_relationnelle = Data_.Satisfaction_relationnelle.replace(SatisfactionRelationnelle_codes)
Data_.Satisfaction_travail = Data_.Satisfaction_travail.replace(SatisfactionTravail_codes)
Data_.StockOptionLevel = Data_.StockOptionLevel.replace(StockOptionLevel_codes)


####partie metric
style_metric_cards('gray',border_radius_px=10,box_shadow=False)
s1,s2,s3 = st.columns(3)
s1.metric("Total Employees",data.shape[0])
s2.metric("% Homme",round(data['Genre\xa0'].value_counts(True)[0]*100,3))
s3.metric("Age median des employee",np.median(Data_['Age']))


st.markdown('<br>',unsafe_allow_html=True)

####partie visualisation

#### piechart

s1,s2 = st.columns([1,2])

a = s1.selectbox('',['Attrition','voyage affaire','Departement','Etat Civil','Heures_supplémentaires'])
if a == 'Attrition' :
    fig = graphe_pie(Data_,'Attrition')
    fig = fig.update_layout(
        height=500,
        title = dict(text='<b> Repartition de l\'Attrition des employees </b>',x=0.1,font_color="green",font_size=24)
    )
    s1.plotly_chart(fig)
if a == 'voyage affaire' :
    fig = graphe_pie(Data_,'Voyage_affaires')
    fig = fig.update_layout(
        height=500,
        title = dict(text='<b> Repartition  des employees suivant le <br>status de voyage d\'Affaire </b>',x=0.1,font_color="green",font_size=24)
    )
    s1.plotly_chart(fig)
if a == 'Departement' :
    fig = graphe_pie(Data_,'Department')
    fig = fig.update_layout(
        height=500,
        title = dict(text='<b> Repartition  des employees par <br> Departement </b>',x=0.1,font_color="green",font_size=24)
    )
    s1.plotly_chart(fig)
if a == 'Etat Civil' :
    fig = graphe_pie(Data_,'État_civil')
    fig = fig.update_layout(
        height=500,
        title = dict(text='<b> Repartition  des employees  par <br>status d\'État_civil</b>',x=0.1,font_color="green",font_size=24)
    )
    s1.plotly_chart(fig)
if a == 'Heures_supplémentaires' :
    fig = graphe_pie(Data_,'Heures_supplémentaires')
    fig = fig.update_layout(
        height=500,
        title = dict(text='<b> Repartition  des employees  par <br>status d\'heure supplementaire</b>',x=0.1,font_color="green",font_size=24)
    )
    s1.plotly_chart(fig)



#### barplot
b = s2.selectbox('',['Implication_dans_emploi','JobLevel','StockOptionLevel','WorkLifeBalance','EducationField','EnvironmentSatisfaction'])

fig =  graph_bar_uni(Data_,b)
fig = fig.update_layout(
    height=500,
    title = dict(text=f'<b> Repartition  des employees  par status de {b} </b>',x=0.1,font_color="green",font_size=24),
    xaxis=dict(showgrid=False,title="",color="white",showticklabels=False),yaxis = dict(title='')
)
s2.plotly_chart(fig)


###barplot bivariee
s1,s2 = st.columns([1,2])

c= s1.selectbox('',['Voyage_affaires',
 'Department',
 'Education',
 'EducationField',
 'EnvironmentSatisfaction',
 'Genre\xa0',
 'Implication_dans_emploi',
 'JobLevel',
 'JobRole',
 'Satisfaction_travail',
 'État_civil',
 'Heures_supplémentaires',
 'Évaluation_performance',
 'Satisfaction_relationnelle',
 'StockOptionLevel',
 'WorkLifeBalance'])
s1.plotly_chart(bar_bi_plot(Data_,c))


quant_features = ['Age',
 'DailyRate',
 'DistanceFromHome',
 'HourlyRate',
 'Revenu_mensuel',
 'MonthlyRate',
 'NumCompaniesWorked',
 'PercentSalaryHike',
 'TotalWorkingYears',
 'TrainingTimesLastYear',
 'YearsAtCompany',
 'YearsInCurrentRole',
 'YearsSinceLastPromotion',
 'YearsWithCurrManager']

d = s2.selectbox('',quant_features)

s2.plotly_chart(plot_histogram_with_density(df= Data_ ,variable=d))

n = len(quant_features)
n_cols = 7
n_rows = (n + n_cols - 1) // n_cols

fig = sp.make_subplots(rows=n_rows,
                       cols=n_cols,
                       subplot_titles=['Attrition vs<br> ' + i for i in quant_features],
                       horizontal_spacing=0.01,vertical_spacing=0.10
                       )
for idx, i in enumerate(quant_features):
    row = idx // n_cols + 1
    col = idx % n_cols + 1


    fig1 = px.box(x=Data_[i],color=Data_['Attrition'] )
    fig1.update_layout(
      height=500 ,
      template='plotly_dark'
      )


    for trace in fig1.data:
        fig.add_trace(trace, row=row, col=col)
    
    fig.update_yaxes(showgrid=False,title="",color="white",showticklabels=False, row=row, col=col)
    fig.update_xaxes(showgrid=False,title="",color="white",showticklabels=False, row=row, col=col)
    
fig.update_layout(
    height=400 * n_rows,
    template='plotly_dark',showlegend=True,
    xaxis=dict(showgrid=False, color='white',
               showticklabels=False), boxmode='group'
)

st.plotly_chart(fig)