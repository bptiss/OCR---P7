import streamlit as st
import shap
from shap import Explanation
from shap.plots import waterfall
import requests
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from urllib.parse import urljoin

# local
API_URL = "http://172.31.21.206:5555/"
# link aws:
# API_URL = "https://mc-credit-score-2.herokuapp.com/"

st.set_option('deprecation.showPyplotGlobalUse', False)

# fonction qui recupere les ids des clients
# @st.cache_data
def get_id_clients():
    url_id_clients = urljoin(API_URL, "cust_ids/")
    # Requesting the API and saving the response
    response = requests.get(url_id_clients)
    # Convert from JSON format to Python dict
    content = json.loads(response.content)
    # Getting the values of SK_IDS from the content
    return content['data']

# fonction de prediction qui recupere le score et la decision
# @st.cache_data
def get_score(id_client):
    url_score = urljoin(API_URL, f"score?id_client={id_client}")
    # Requesting the API and saving the response
    response = requests.get(url_score)
    # Convert from JSON format to Python dict
    content = json.loads(response.content)
    # Getting the values of SK_IDS from the content
    score = float(content['score'])
    decision = content['decision']
    return score, decision

# fonction qui recupere les informatons descriptives du client (id_client)
# @st.cache_data
def get_data_client(id_client):
    url_score = urljoin(API_URL, f"data_client/?id_client={id_client}")
    # Requesting the API and saving the response
    response = requests.get(url_score)
    # Convert from JSON format to Python dict
    content = json.loads(response.content)
    data_client = content['data']
    df_data_client = pd.DataFrame([data_client])
    print(f"[TEST VALUE] (get_data_client) df_data_client['Age'] : {df_data_client['Age']}")
    return df_data_client

# fonction qui recupere les infos descriptives des clients similaires de l'API
# @st.cache_data
def get_clients_sim(id_client):
    url_score = urljoin(API_URL, f"clients_similaires/?id_client={id_client}")
    # Requesting the API and saving the response
    response = requests.get(url_score)
    # Convert from JSON format to Python dict
    content = json.loads(response.content)
    data_voisins = content['df_voisins']
    data_voisins_norm = content['df_voisins_norm']
    df_voisins = pd.DataFrame(data_voisins)
    df_voisins_norm = pd.DataFrame(data_voisins_norm)
    return df_voisins_norm, df_voisins

# fonction qui recupere les infos descriptives de tous les clients
# @st.cache_data
def get_infos_clients():
    url_score = urljoin(API_URL, f"/infos_desc_all_clients/")
    # Requesting the API and saving the response
    response = requests.get(url_score)
    # Convert from JSON format to Python dict
    content = json.loads(response.content)
    data_scores_clients = content['scores_clients']
    data_scores_clients_norm = content['norm_scores_clients']
    df_scores_clients = pd.DataFrame(data_scores_clients)
    df_scores_clients_norm = pd.DataFrame(data_scores_clients_norm)
    return df_scores_clients, df_scores_clients_norm


# fonction qui recupere les shap values du client pour l'interpretation locale
@st.cache_data
def loc_feat_imp(id_client):
    url_score = urljoin(API_URL, f"feat_imp/?id_client={id_client}")
    # Requesting the API and saving the response
    response = requests.get(url_score)
    # Convert from JSON format to Python dict
    content = json.loads(response.content)
    # Getting all data for Explanation
    shap_val = json.loads(content['shap_values'])
    base_value = json.loads(content['base_value']) #ajoute
    feat_values = json.loads(content['data'])
    feat_names = json.loads(content['feature_names']) # ajoute
    # shap_val = (content['shap_values'])
    # base_value = (content['base_value'])
    # feat_values = (content['data'])
    # feat_names = (content['feature_names'])
    return shap_val, base_value, feat_values, feat_names

# get feat. imp. global
# @st.cache_data
def get_feat_imp_glob():
    url_score = urljoin(API_URL, "/feat_imp_global/")
    # Requesting the API and saving the response
    response = requests.get(url_score)
    # Convert from JSON format to Python dict
    content = json.loads(response.content)
    feat_imp = content['feat_imp_global']
    df_feat_imp = pd.DataFrame(feat_imp)
    return df_feat_imp

# fct pour plot feat. imp. global
# @st.cache_data
def plot_feature_importances(df):
    # Sort features according to importance
    df = df.sort_values('importance', ascending=False).reset_index(drop=True)
    # Normalize the feature importances to add up to one
    df['importance_normalized'] = df['importance'] / df['importance'].sum()
    # Make a horizontal bar chart of feature importances
    plt.figure(figsize=(14, 10))
    ax = plt.subplot()
    # Need to reverse the index to plot most important on top
    ax.barh(list(reversed(list(df.index[:12]))),
            df['importance_normalized'].head(12),
            align='center', edgecolor='k')
    # Set the yticks and labels
    ax.set_yticks(list(reversed(list(df.index[:12]))))
    ax.set_yticklabels(df['feature'].head(12))
    # Plot labeling
    plt.xlabel('Normalized Importance')
    plt.title('Feature Importances globales des 12 premières caractéristiques')

# fonction boxplot des clients en fct de la classe et place le client actuel (num_client)
# @st.cache_data
def box_plot_all(df, num_client, cols, decision):
    df_melt = df.melt(id_vars=['class'],
                value_vars=cols,
                var_name="variables",
                value_name="values")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.boxplot(data=df_melt, x='variables', y='values',  hue='class', linewidth=1, showfliers=False,
                width=0.4, palette=['tab:green', 'tab:red'], saturation=0.5, ax=ax)

    ax.set_xlabel('')
    ax.set_ylabel("Valeurs normalisées", fontsize=15)
    # données client applicant
    cust_t = df.loc[str(num_client)][cols].to_frame().reset_index()
    df_cust = cust_t.rename(columns={"index": "var", str(num_client): "val"})
    df_cust['color'] = ('green' if decision =='Granted' else 'red')
    sns.swarmplot(data=df_cust, x='var', y='val', linewidth=1, color=df_cust['color'].unique()[0], marker='p', size=8, label='client', ax=ax) #color=df_cust['color'].unique()[0]
    ax.set_xticklabels(ax.get_xticklabels(),rotation=20)
    ax.set(xlabel="", ylabel="Valeurs normalisées")
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles[:3], ["Granted", "Refused", "Client"])
    plt.show()



#Titre de la page
# st.set_page_config(page_title="", layout="wide")


########################################################
# General settings
########################################################
st.set_page_config(
    page_title="Prêt à dépenser - Demandes de prêts",
    # page_icon=icon,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Report a bug" : None,
        "Get help" : "https://github.com/bptiss",
        "About" : 
        '''
        Made by Baptiste LEDESERT 
        in OpenClassRooms Data Scientist Training.

        For more information visit the following 
        [link](https://https://github.com/bptiss/OCR---P7), 
        
        or LinkedIn:[link](www.linkedin.com/in/bledeser)
        '''
    }
)


# Add the CSS file
st.markdown('<link href="styles.css" rel="stylesheet">', unsafe_allow_html=True)

st.title("Prêt à dépenser - Demandes de prêts")
st.write("Dashboard réalisé par Baptiste LEDESERT - Parcours Data Science - projet 7")

#Menu déroulant ids clients
print(str(dt.datetime.now()) + ": st_dashboard: appel request get_id_client for selectbox")
values = sorted(get_id_clients())
values.insert(0, '<Select>')
num = st.sidebar.selectbox(
    "Sélectionner un numéro de client",
    values)

button_score = st.sidebar.button("Statut de la demande")
infos_client = st.sidebar.checkbox("Informations descriptives du client")
influence = st.sidebar.checkbox("Facteurs d'influence")
feat_imp_global = st.sidebar.checkbox("Facteurs d'influence golbaux")
comp_infos_tous_clients = st.sidebar.checkbox("Comparaison avec les autres clients")


print(str(dt.datetime.now()) + ": st_dashboard: retour request get_id_client for selectbox")
if num != '<Select>':
    id_input = num
    # getting the values from the content
    print(str(dt.datetime.now()) + ": st_dashboard: appel request get_score")
    score, decision = get_score(num)
    print(str(dt.datetime.now()) + ": st_dashboard: retour request get_score")
    # Présentation des résultats
    # Titre :
    st.markdown("<h3 style='text-align: center; color: #5A5E6B;'>ANALYSE DU DOSSIER</h3>", unsafe_allow_html=True)

    # Jauge
    if button_score:
        plot_score =1
        if decision == 'Granted':
            st.success("Crédit accepté")
        else:
           st.error("Crédit rejeté")

        gauge = go.Figure(go.Indicator(
            domain={'x': [0, 1], 'y': [0, 1]},
            value=score,
            mode="gauge+number",
            title={'text': "Score", 'font': {'size': 24}},
            gauge={'axis': {'range': [None, 1]},
                   'bar': {'color': "grey"},
                   'steps': [{'range': [0, 0.55], 'color': 'Green'},
                             {'range': [0.55, 0.654], 'color': 'LimeGreen'},
                             {'range': [0.654, 0.656], 'color': 'red'},
                             {'range': [0.656, 0.7], 'color': 'Orange'},
                             {'range': [0.7, 1], 'color': 'Crimson'}],

                   'threshold':
                       {'line': {'color': "red", 'width': 4},
                        'thickness': 1, 'value': score,
                        }
                   }
        ))
        if plot_score:
            st.plotly_chart(gauge, use_container_width=True)
    
    
    if infos_client:
        st.info("Informations sur le client")
        st.write(get_data_client(num))

    # feature importance locale pour client

    if influence:
        st.info("Interprétabilité locale du résultat")
        shap_val, base_value, feat_values, feat_names = loc_feat_imp(num)
        explant = shap.Explanation(np.array(shap_val), 
                              base_value, 
                              data=feat_values, 
                              feature_names=feat_names)
        shp1,shp2=st.columns(2)
        with shp1:
            nb_features = st.slider('Nombre de variables à visualiser', 0, 20, 10)
            fig, ax = plt.subplots()
            ax = shap.waterfall_plot(explant, max_display=nb_features, show=False)

            st.pyplot(fig)
            exp = st.expander("Explanation of the SHAP waterfall plot?")
            exp.markdown("The bottom of the above waterfall plot starts as the expected value of the model output \
                  (E(f(x)) which is in other words, the mean of all predictions). \
                  Each row shows how the positive (red) or negative (blue) contribution of \
                  each feature moves the value from the expected model output over the \
                  the predicted value of the model, of the applicant customer f(x).")

    # features importances globaux du modele

    if feat_imp_global:
        st.info("Importance globale des caractéristiques")
        df_feat_imp = get_feat_imp_glob()
        # Graphiques et infos :
        graphique1, graphique2 = st.columns(2)
        with graphique1:
            fig = plot_feature_importances(df_feat_imp)
            st.pyplot(fig)


    # comparaison descriptives avec d'autres clients

    if comp_infos_tous_clients:
        # Menu déroulant autres clients
        list_comp = ["Groupe clients", "Clients similaires"]
        list_comp.insert(0, '<Select>')
        default_citc="df.loc[num_client][cols]"
        comp_clients = st.sidebar.selectbox(
            "Sélectionner une comparasion",
            list_comp)
        df_infos_scores, df_infos_norm = get_infos_clients()
        features_num = df_infos_scores.select_dtypes(include=np.number).columns.tolist()
        features_num.remove('class')
        features_num.remove('score')
        # features_num.remove('CNT_CHILDREN')
        to_select = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY',
                             'AMT_GOODS_PRICE', 'year_employed', 'Age']
        # comparaison avec un groupe des clients
        
        if comp_clients == "Groupe clients":

            st.info("Comparaison avec un groupe des clients")
            st.write("Données pour un échantillon de clients")
            nb_clients_to_see = st.slider('Nombre de clients à visualiser', 2, 20, 10)
            st.dataframe(df_infos_scores.drop(['class','score','decision'],axis=1).sample(nb_clients_to_see))
            if st.checkbox('boxplot des caractéristiques principales des clients'):

                select_features = st.multiselect("Sélectionner les caractéristiques à visualiser des clients: ",
                                                 sorted(features_num),
                                                 default=to_select) 

                graphique1, graphique2 = st.columns(2)
                with graphique1:
                    fig = box_plot_all(df_infos_norm, num, select_features, decision)
                    st.pyplot(fig)

            if st.checkbox('Analyse bivariée des caractéristiques à sélectionner des clients'):
                # to_select = ['NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY',
                #              'AMT_GOODS_PRICE', 'years_employed', 'NAME_CONTRACT_TYPE']

                to_select.insert(0, '<Select>')
                list_x = to_select
                list_y = to_select
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    a = st.selectbox("Sélectionner une caractéristique X ", list_x)
                with c2:
                    b = st.selectbox("Sélectionner une caractéristique Y", list_y)
                if (a != '<Select>') & (b != '<Select>'):
                    fig = px.scatter(df_infos_scores, x=a, y=b, color="decision", opacity=0.5, width=1000, height=600,
                                     color_discrete_sequence=["red", "green"],
                                     title="Analyse bivariée des caractéristiques sélectionnées pour un groupe des clients")
                    df_cust = df_infos_scores.iloc[df_infos_scores.index == str(num)]
                    fig.add_trace(go.Scatter(x=df_cust[a], y=df_cust[b], mode='markers',
                                             marker_symbol='star', marker_size=30, marker_color="black",
                                             name="Client"))  # showlegend=False))
                    fig.update_layout(
                        font_family="Arial",
                        font_size=15,
                        title_font_color="blue")

                    st.plotly_chart(fig, use_container_width=False)

        # comparaison avec les clients similaires
        if comp_clients == "Clients similaires":
            df_voisins_norm, df_voisins = get_clients_sim(num)
            st.info("Comparaison avec les clients similaires")
            st.dataframe(df_voisins)
            if st.checkbox('boxplot des caractéristiques principales des clients similaires'):
                select_features = st.multiselect("Sélectionner les caractéristiques à visualiser : ",
                                                 sorted(features_num),
                                                 default=to_select)

                graphique1, graphique2 = st.columns(2)
                with graphique1:
                    fig = box_plot_all(df_voisins_norm, num, select_features, decision)
                    st.pyplot(fig)

            if st.checkbox('Analyse bivariée des caractéristiques à sélectionner des clients similaires'):
                to_select_sim = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY',
                                  'AMT_GOODS_PRICE','year_employed', 'Age']
                to_select_sim.insert(0, '<Select>')
                x = to_select_sim
                y = to_select_sim
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    x = st.selectbox("Sélectionner X", x)
                with c2:
                    y = st.selectbox("Sélectionner Y", y)
                if (x != '<Select>') & (y != '<Select>'):

                    df_voisins["Decision"] = ['Granted' if elt == 0 else 'Refused' for elt in df_voisins["class"]]
                    fig = px.scatter(df_voisins, x=x, y=y, color="Decision", opacity=0.6, width=1000, height=600, size="score",
                                      color_discrete_map={"Refused": "red",
                                                         "Granted": "green"},
                                         #color_discrete_sequence=["green", "red"],
                                     title="Analyse bivariée des caractéristiques sélectionnées des clients similaires")
                    df_cust = df_voisins.iloc[df_voisins.index == str(num)]
                    df_cust['color'] = ('green' if decision == 'Granted' else 'red')
                    fig.add_trace(go.Scatter(x=df_cust[x], y=df_cust[y], mode='markers',
                                             marker_symbol='star', marker_size=15, marker_color=df_cust['color'],
                                             name="Client"))
                    #fig.update_xaxes(matches='x')
                    fig.update_layout(
                        font_family="Arial",
                        font_size=15,
                        title_font_color="blue")

                    c1, c2 = st.columns(2)
                    st.plotly_chart(fig, use_container_width=False)
                    with c1:
                        nb = st.expander("Note")
                        nb.write("La taille des points correspond aux scores des clients (voir la jauge du statut de la demande pour comprendre les scores")
