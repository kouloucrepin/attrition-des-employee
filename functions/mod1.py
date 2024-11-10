import pandas as pd
import plotly.graph_objects as go
import scipy.stats as stats
import plotly.express as px
import numpy as np


def table(Data_,var):
    inter = Data_[var].value_counts().to_frame().reset_index()
    inter[var] = inter[var].astype(str)
    inter['pourcent']=np.round(Data_[var].value_counts(normalize=True).to_frame().reset_index()['proportion']*100,2)
    inter['text']='n= ' + inter['count'].astype(str) + '  <br> percent =  ' + inter['pourcent'].astype(str) + '%'
    return inter

def graph_bar_uni(Data_,var):
    title = 'repartition de la variable ' + var
    inter = table(Data_,var).sort_values(by='count')
    return px.bar(y=inter[var],x=inter["count"],color=inter[var],text=inter['text'])


def graphe_pie(Data_,variable,size=10):
    titre="Répartion de la variable  " + variable
    fig = px.pie(Data_[variable].value_counts().to_frame().reset_index(),names=variable,
             values="count",hole=0.5,height=350,title=titre)
    fig.update_traces(textinfo='label+percent+value',showlegend=False)
    fig.update_layout(title=dict(font_color="white"))
    return fig


def bar_bi_plot(Data_,i):
    s = pd.crosstab(Data_['Attrition'], Data_[i])
    s = np.round(s.divide(s.sum(axis=1) / 100, axis=0).unstack().reset_index().rename(columns={0: 'value'}), 2)


    bar_fig = px.bar(
        x=s['Attrition'].astype(str),
        y=s['value'],
        color=s[i].astype(str),
        text=s[i].astype(str) + ' :  ' + s['value'].astype(str) ,
        barmode='relative',title='Attrition vs  ' + str(i)
    )
    bar_fig.update_layout(showlegend=False,height=480,
                          xaxis=dict(showgrid=False,title="",color="white"),
                          yaxis=dict(showgrid=False,title="",color="white",showticklabels=False),
                          title = dict(x=0.3,font_color="green",font_size=24),
                          margin=dict(t=34,l=0,r=0,b=5))
    return bar_fig


def plot_histogram_with_density(df, variable):
    
    
    if variable not in df.columns:
        raise ValueError(f"La variable '{variable}' n'existe pas dans la DataFrame.")
    
    histogram = go.Histogram(
        x=df[variable],
        histnorm='probability density', 
        nbinsx=30,
        name='Histogramme',
    )
    x = np.linspace(df[variable].min(), df[variable].max(), 100)
    y = stats.gaussian_kde(df[variable])(x)

    density_curve = go.Scatter(
        x=x,
        y=y,
        mode='lines',
        name='Courbe de Densité',
        line=dict(color='white')
    )

    
    fig = go.Figure(data=[histogram, density_curve])

    
    fig.update_layout(
        barmode='overlay',
        template='plotly_dark',
        xaxis=dict(showgrid=False,title="",color="white"),
        yaxis=dict(showgrid=False,title="",color="white",showticklabels=False),showlegend=False,height=480,
        title = dict(x=0.3,font_color="green",font_size=24,text=f'Distribution de  la variable    {variable}')
    )

    
    return fig
    


example = pd.DataFrame({'Education': {0: 1}, 'EnvironmentSatisfaction': {0: 1}, 'Implication_dans_emploi': {0: 1}, 'JobLevel': {0: 1}, 'Satisfaction_travail': {0: 1}, 'Évaluation_performance': {0: 1}, 'Satisfaction_relationnelle': {0: 1}, 'StockOptionLevel': {0: 0}, 'WorkLifeBalance': {0: 0}, 'Voyage_affaires': {0: 'Travel_Rarely'}, 'Department': {0: 'Sales'}, 'EducationField': {0: 'Life Sciences'}, 'Genre': {0: 'Female'}, 'JobRole': {0: 'Sales Executive'}, 'État_civil': {0: 'Single'}, 'Heures_supplémentaires': {0: 'Yes'}, 'Age': {0: 0}, 'DailyRate': {0: 0}, 'DistanceFromHome': {0: 0.0}, 'HourlyRate': {0: 0}, 'Revenu_mensuel': {0: 0}, 'MonthlyRate': {0: 0}, 'NumCompaniesWorked': {0: 0}, 'PercentSalaryHike': {0: 0.0}, 'TotalWorkingYears': {0: 0}, 'TrainingTimesLastYear': {0: 0.0}, 'YearsAtCompany': {0: 0}, 'YearsInCurrentRole': {0: 0}, 'YearsSinceLastPromotion': {0: 0}, 'YearsWithCurrManager': {0: 0}})