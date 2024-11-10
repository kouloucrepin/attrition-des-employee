import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objects as go
import joblib
import numpy as np
from sklearn.preprocessing import RobustScaler,StandardScaler,OneHotEncoder,OrdinalEncoder,MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,roc_curve,RocCurveDisplay,confusion_matrix,ConfusionMatrixDisplay,classification_report,accuracy_score,precision_score,recall_score,f1_score,accuracy_score
import os
import plotly.figure_factory as ff
from sklearn.pipeline import Pipeline
st.sidebar.image('images/sidebar.png')
st.set_page_config(layout="wide")
data = pd.read_csv( os.path.join(os.getcwd(), 'data', 'ml.csv'),sep=';')
categorial_var = [
    'Education','EnvironmentSatisfaction',
    'Implication_dans_emploi','JobLevel',
    'Satisfaction_travail','Évaluation_performance',
    'Satisfaction_relationnelle','StockOptionLevel',
    'WorkLifeBalance'
    ]
nominal_var = [
    'Voyage_affaires','Department',
    'EducationField','Genre\xa0',
    'JobRole','État_civil',
    'Heures_supplémentaires'
]
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
data.colums=data.columns.str.strip()
data['DistanceFromHome'] = data['DistanceFromHome'].fillna(np.median(data['DistanceFromHome'].dropna()))

data = data[data.columns[data.nunique()>1]]
Data = data.drop('EmployeeNumber',axis=1)
def detect_n_outliers(variable):
    data = Data[variable]
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = (data < lower_bound) | (data > upper_bound)

    return outliers.sum()

def replace_if_good(variable):
    data = Data[variable]
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    if detect_n_outliers(variable)<Data.shape[0]/5: ## faire le remplacement ssi le nombre de valeur manquantes est negligeable(<4%)
       data[data > upper_bound] =  upper_bound
       data[data < lower_bound] =  lower_bound
    return data

for i in quant_features:
    Data[i] = replace_if_good(i)

s = pd.DataFrame(quant_features,columns=['quant_var'])

s['nb_val_aberantes']=s['quant_var'].apply(detect_n_outliers)


var_quant_without_abb = s.query("nb_val_aberantes==0").quant_var.values.tolist() 
var_quant_with_abb = s.query("nb_val_aberantes>0").quant_var.values.tolist()



vars_ = categorial_var + nominal_var + var_quant_without_abb + var_quant_with_abb
Data.Attrition = Data.Attrition.replace({'No':0,'Yes':1})

column_transformer = ColumnTransformer(
    transformers=[
        ('categorial_var', MinMaxScaler(), categorial_var),
        ('nominal_var',OneHotEncoder(drop="first"),nominal_var),
        ('var_quant_without_abb',StandardScaler() , var_quant_without_abb),
        ('var_quant_with_abb',RobustScaler() , var_quant_with_abb),
    ]
)
column_transformer.fit_transform(Data[vars_])
list_transforme = column_transformer.get_feature_names_out().tolist()

z,w,y = st.columns([3,1,1])

X = Data[vars_]
Y = Data.Attrition
rs = w.number_input('Random state',min_value=0)


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25,stratify=Y,random_state=rs)



def generer_model(model_name, model,column_transformer=column_transformer):
    ''''
    L objectif ici est de mettre en place toute la pipeline
    du traitement a l entrainement du model
    sans trop se fatiguer a ecire les meme ligne de code
    '''
    pipeline = Pipeline([
        ('scaler', column_transformer),
        (model_name, model)
    ])
    return pipeline

def courbe_roc(model ,title='model Logistic'):
    ''''
    L objectif est de generer la courbe roc une fois le model
    apris sans vraimment repeter du code
    '''
    y_probs_train = model.predict_proba(x_train)[:, 1]
    fpr_t, tpr_t, thresholds_t = roc_curve(y_train, y_probs_train)
    roc_auc_t = roc_auc_score(y_train, y_probs_train)

    y_probs_test = model.predict_proba(x_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_probs_test)
    roc_auc = roc_auc_score(y_test, y_probs_test)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr_t, y=tpr_t,
                            mode='lines',
                            name=f'Entraînement (AUC = {roc_auc_t:.2f})',
                            line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=fpr, y=tpr,
                            mode='lines',
                            name=f'Test (AUC = {roc_auc:.2f})',
                            line=dict(color='red')))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                            mode='lines',
                            name='Aucune compétence',
                            line=dict(color='grey', dash='dash')))
    fig.update_layout(title=dict(x=0.5,font_color="white",
                             text=title),
                    xaxis_title='Taux de Faux Positifs',
                  yaxis_title='Taux de Vrais Positifs',
                  template='plotly_dark',paper_bgcolor="#4F4F4F",plot_bgcolor="#4F4F4F" ,
                  showlegend=True)

    return fig

def confusion_matri(model,title='Logistic Regression'):
    cm = confusion_matrix(y_test,model.predict(x_test))




    fig = ff.create_annotated_heatmap(
        z=cm,
        x=['No', 'Yes'],
        y=[ 'No','Yes'],
        colorscale='turbo',
        showscale=False,text=cm,texttemplate='%{text}'
    )

    fig.update_layout(
        title=title,
        xaxis=dict(title='Valeurs Prédites'),
        yaxis=dict(title='Valeurs Réelles'),
        height=600,
        width=600,
        template='plotly_dark',paper_bgcolor="#4F4F4F",plot_bgcolor="#4F4F4F"
    )

    return fig

def metricss(model,name):
    a =  pd.Series({
    'concern':'test',
    'name':name,
    'accuracy':accuracy_score(y_test, model.predict(x_test)),
    'precision':precision_score(y_test, model.predict(x_test)),
    'recall':recall_score(y_test, model.predict(x_test)),
    'f1_score':f1_score(y_test, model.predict(x_test)),
    }).to_frame().T

    b =  pd.Series({
    'concern':'train',
    'name':name,
    'accuracy':accuracy_score(y_train, model.predict(x_train)),
    'precision':precision_score(y_train, model.predict(x_train)),
    'recall':recall_score(y_train, model.predict(x_train)),
    'f1_score':f1_score(y_train, model.predict(x_train)),
    }).to_frame().T

    return pd.concat([b,a],axis=0)

def roc_confusion(fig2,fig1,title):
    fig = sp.make_subplots(rows=1, cols=2,
                            subplot_titles=('Confusion Matrice of  ' + title +'\n', 'Roc curve of   '  + title),column_widths=[0.3,0.7],
                            row_heights=[1000],horizontal_spacing=0.09,vertical_spacing=0.1 )

    # Ajouter les graphiques à la grille
    fig.add_trace(fig1.data[0], row=1, col=2)
    fig.add_trace(fig1.data[1], row=1, col=2)
    fig.add_trace(fig1.data[2], row=1, col=2)
    fig.add_trace(fig2.to_dict()['data'][0], row=1, col=1)
    fig.update_layout(
        height=500 ,
        template='plotly_dark'
    )
    fig.update_yaxes(title_text='Valeurs Réelles', row=1, col=1)
    fig.update_xaxes(title_text='Valeurs Prédites', row=1, col=1)
    fig.update_yaxes(title_text='Taux de Vrais Positifs', row=1, col=2)
    fig.update_xaxes(title_text='Taux de Faux Positifs', row=1, col=2)
    return fig



a = z.radio('',['LogisticRegression','RandomForest','reseau de neurone','xgboost'],horizontal=True)
y.button('Sauvegarder Le model',type='secondary')


if a=='LogisticRegression':
    ml_LR = generer_model('LogisticRegression',LogisticRegression(C=0.1,class_weight= {0: 4, 1: 6},penalty='l2'))
    ml_LR.fit(x_train,y_train)
    st.plotly_chart(roc_confusion(confusion_matri(ml_LR),courbe_roc(ml_LR),'LogisticRegression'))
    s = pd.DataFrame(ml_LR.named_steps['LogisticRegression'].coef_).T.rename(columns={0:'coef'})
    s['color'] = s['coef'].apply(lambda x : 'red' if x<=0 else 'gray')
    s['variable']=list_transforme


    s = s.sort_values(by='coef')[['variable','coef','color']]


    fig = px.bar(s,y='coef',color='color',hover_data='variable')
    fig.update_layout(title=dict(x=0.5,font_color="white",text='coeficient des differentes variables'),
                    template='plotly_dark',
                    showlegend=False,
        xaxis=dict(showgrid=True,color='white',title="",showticklabels=False),)
    s1,s2 = st.columns([3,1])
    s1.plotly_chart(fig)
    s2.markdown('<br><br>',unsafe_allow_html=True)
    s2.dataframe(metricss(ml_LR,'ml_LR_best').reset_index().drop('index',axis=1).transpose(),height=300)
    
    model = ml_LR
    name = 'LogisticRegression'



if a=='RandomForest':
    ml_RF = generer_model('RandomForestClassifier',RandomForestClassifier(class_weight={0: 3, 1: 8},criterion= 'log_loss',max_depth= 8,n_estimators= 100))
    ml_RF.fit(x_train,y_train)
    st.plotly_chart(roc_confusion(confusion_matri(ml_RF),courbe_roc(ml_RF),'RandomForestClassifier'))
    s = pd.DataFrame(ml_RF.named_steps['RandomForestClassifier'].feature_importances_).rename(columns={0:'importance'})
    s['color'] = s['importance'].apply(lambda x : 'red' if x<=0 else 'gray')
    s['variable']=list_transforme
    s = s.sort_values(by='importance')
    fig = px.bar(x=s['importance'],y=s['variable'],color=s['color'],height=500)
    fig = fig.update_layout(title=dict(x=0.5,font_color="white",text='importance des differentes variables'),
                    template='plotly_dark',
                    showlegend=False,
        xaxis=dict(showgrid=True,color='white',title="",showticklabels=True),yaxis=dict(title=''))
    
    s1,s2 = st.columns([3,1])
    s1.plotly_chart(fig)
    s2.markdown('<br><br>',unsafe_allow_html=True)
    s2.dataframe(metricss(ml_RF,'ml_RF_best').reset_index().drop('index',axis=1).transpose(),height=300)
    
    model = ml_RF
    name = 'RandomForest'
    
    
if a=='reseau de neurone':
    ml_MLP = generer_model('MLPClassifier',MLPClassifier(activation= 'logistic',learning_rate= 'adaptive',solver= 'adam',max_iter=500))
    ml_MLP.fit(x_train,y_train)

    st.plotly_chart(roc_confusion(confusion_matri(ml_MLP),courbe_roc(ml_MLP),' MLPClassifier '))
    st.dataframe(metricss(ml_MLP,'ml_MLP_best'))
    
    model = ml_MLP
    name = 'reseau_de_neurone'

if a=='xgboost': 
    ml_xgb= generer_model('xgboost',XGBClassifier(learning_rate= 0.01,
                                                  max_depth=4,n_estimators= 700,
                                                  reg_alpha= 1,reg_lambda=1))  
    ml_xgb.fit(x_train,y_train)
    st.plotly_chart(roc_confusion(confusion_matri(ml_xgb),courbe_roc(ml_xgb),' ml_xgb'))
    data_ = pd.DataFrame(ml_xgb.named_steps['xgboost'].feature_importances_).rename(columns={0:'importance'})
    data_['variable'] = list_transforme
    data_ = data_.sort_values(by='importance',ascending=True)
    data_['variable']=data_.variable.str.split('_',n=3,expand=True)[3]

    fig = px.bar(x=data_['importance'],height=600,y=data_['variable'],template='plotly_dark')
    fig = fig.update_layout(title=dict(x=0.5,font_color="white",text='importance des differentes variables'),
                    showlegend=False,
        xaxis=dict(showgrid=True,color='white',title="",showticklabels=True))
    
    s1,s2 = st.columns([2,1])
    s1.plotly_chart(fig)
    s2.markdown('<br><br>',unsafe_allow_html=True)
    s2.dataframe(metricss(ml_xgb,'xgboost').reset_index().drop('index',axis=1).transpose(),height=300)
    
    model = ml_xgb
    name='xgboost'



if y:
    if os.path.exists(f'models/{name}.pkl'):
        os.remove(f'models/{name}.pkl')
        joblib.dump(model, f'models/{name}.pkl') 
    else:
        joblib.dump(model, f'models/{name}.pkl') 
