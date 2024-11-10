import streamlit as st

st.set_page_config(layout="wide")

st.sidebar.image('images/sidebar.png')

st.markdown("<h1 style='text-align: center;color:green;'> Prédiction de l'Attrition : Anticiper les Départs pour Optimiser la Rétention des Employés</h1>", unsafe_allow_html=True)
st.markdown("<br><br><br>", unsafe_allow_html=True)
st.markdown("""<div style="border: 2px solid gray; font-size: 20px; padding: 10px; border-radius: 3px; background-color: black;"><span style="color: red;text-align: center; "><br> CONTEXTE DE L'ETUDE.<br></span> 
            Dans un monde professionnel en constante évolution, la gestion des talents est devenue un enjeu crucial pour les entreprises. L'attrition des employés, 
            c'est-à-dire le départ volontaire ou involontaire des collaborateurs, peut avoir des conséquences significatives sur la performance organisationnelle, la
            culture d'entreprise et les coûts opérationnels. En effet, le remplacement d'un employé peut coûter jusqu'à 200 % de son salaire annuel, sans compter  l'impact sur la motivation et l'engagement des équipes restantes.<br><br>Face à ces défis, il est essentiel pour les entreprises de comprendre les facteurs qui influencent l'attrition et de prédire quels employés sont susceptibles de quitter 
           l'organisation. C'est ici qu'intervient le machine learning. En analysant des données historiques sur les employés, y compris des informations sur leur performance, leur 
           satisfaction au travail et d'autres indicateurs clés, il est possible de développer des modèles prédictifs qui identifient les risques d'attrition.<br>
            </div><br><br>""", unsafe_allow_html=True)


st.markdown("""<div style="border: 2px solid gray; font-size: 20px; padding: 10px; border-radius: 3px; background-color: black;"><span style="color: red;text-align: center; "><br> OBJECTIFS DE L APPS.</span>  <br>
- Analyser les Données : Collecter et prétraiter des données relatives aux employés, y compris des caractéristiques démographiques, des évaluations de performance, des enquêtes de satisfaction et des historiques de départs.<br>
- Développer un Modèle Prédictif : Utiliser des algorithmes de machine learning pour créer un modèle capable de prédire la probabilité d'attrition pour chaque employé. Les techniques telles que la régression logistique, les forêts aléatoires et les réseaux de neurones seront explorées.<br>
- Interpréter les Résultats : Fournir des insights exploitables sur les principaux facteurs contribuant à l'attrition, permettant ainsi aux responsables RH de mettre en place des stratégies de rétention ciblées.<br>
- Proposer des Solutions : Formuler des recommandations basées sur les résultats du modèle pour améliorer la satisfaction des employés et réduire les taux d'attrition.<br>
- Proposer des Solutions : Formuler des recommandations basées sur les résultats du modèle pour améliorer la satisfaction des employés et réduire les taux d'attrition.<br>
            <br></div>""", unsafe_allow_html=True)