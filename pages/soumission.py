import streamlit as st
import joblib
import pandas as pd
from functions.mod1 import vars_

st.sidebar.image('images/sidebar.png')

st.write('## Cette partie vise a faire les prediction pour un fichiers excel a une seulE feuille contenant les information sur les employees')
z,y = st.columns([2,1])
a = z.radio('',['LogisticRegression.pkl','RandomForest.pkl','reseau_de_neurone.pkl','xgboost.pkl'],horizontal=True)

model = y.file_uploader('Charger un fichier(le fichier excel doit contenir une seul feuille de donnee)',
                        help='Charger un fichier excel contenant les 35 information de votre base de donnees',type=['xls','xlsx'])


if model is not None :
    models = joblib.load(f'models/{a}')

    base = pd.read_excel(model)
    col = base.columns

    try:
        base['prediction'] = models.predict(base[vars_])
    except:
        vars_[12] = 'Genre\xa0'
    base = base.fillna(0)
    base['prediction'] = models.predict(base[vars_])
    
    st.dataframe(base)    