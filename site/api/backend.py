# -*- coding: utf-8 -*-
from flask import Flask, jsonify, request
import pickle
import json
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import shap
import datetime as dt

# load data
df_test = pd.read_csv("../../Data/finals_datasets/sample_test_all_fs.csv", index_col='SK_ID_CURR')

df_infos = pd.read_csv("../../Data/finals_datasets/sample_test_infos_fs.csv", index_col='SK_ID_CURR')

df_feat_imp = pd.read_csv("../../Data/finals_datasets/df_fs.csv")

customers_ids = list(df_test.index.to_series())

# load the model
# Light GBM model avec undersampling
best_model_bank = pickle.load(open('../../models/classif_lightGBM_undersampling.pkl', 'rb'))

# Seuil optimal du modèle pour classer les clients en défaillants ou non
best_treshold = 0.654

# SHAP: explainer of tree saved
SHAP_VALUES_FILENAME = "../../Data/finals_datasets/shap_values.csv"
clf_lgbm_us_shap_values = pickle.load(open(SHAP_VALUES_FILENAME, 'rb'))

# SHAP: expected value saved
SHAP_EXPECTED_VALUES_FILENAME = "../../Data/finals_datasets/expected_values.csv"
clf_lgbm_us_explainer = pickle.load(open(SHAP_EXPECTED_VALUES_FILENAME, 'rb'))

def neigb_mod(df):
    categ = df['class']
    df['Age'] = np.around(df['DAYS_BIRTH']/-365,0) 
    df['year_employed'] = np.around(df['DAYS_BIRTH']/-365,0)
    df_num = df.select_dtypes('number').drop(['score', 'class'], axis=1)
    scaler = MinMaxScaler().fit(df_num)
    df_num_norm = pd.DataFrame(scaler.transform(df_num), columns=df_num.columns,
                               index=df.index)
    neigh = NearestNeighbors(n_neighbors=20)
    neighbors = neigh.fit(df_num_norm)
    df_num_norm['class'] = categ

    return neighbors, df_num_norm

app = Flask(__name__)

neighbors, df_infos_norm = neigb_mod(df_infos)
infos_json = json.loads(df_infos.to_json())
infos_norm_json = json.loads(df_infos_norm.to_json())

# retourner ids clients 'SK_ID_CURR'
@app.route('/cust_ids/')
def clients_ids():
    # Return en json format
    return jsonify({'status': 'ok',
                    'data': customers_ids})


# Retourner les infos descriptives d'un client (SK_ID_CURR)
@app.route('/data_client/', methods=['GET', 'POST'])
def data_client():
    # Parse the http request to get arguments ('SK_ID_CURR')
    id_client = int(request.args.get('id_client'))
    # infos descriptives pour le client (id_client)
    df_infos['Age'] = np.around(df_infos['DAYS_BIRTH']/-365,0) 
    df_infos['year_employed'] = np.around(df_infos['DAYS_BIRTH']/-365,0)
    # print(f"[TEST VALUE] df_infos['Age'][0] : {df_infos.loc[100001,'Age']}")
    feat_client = df_infos.loc[id_client].drop(['score', 'class', 'decision'])
    # Return data
    return jsonify({'data': json.loads(feat_client.to_json())})


# Retourner les infos descriptives de tous les clients
@app.route('/infos_desc_all_clients/')
def all_data_clients():
    return jsonify({'scores_clients': infos_json,
                    'norm_scores_clients': infos_norm_json})


# Retourner feature importance global
@app.route('/feat_imp_global/')
# get globals features importance
def feat_imp_glob():
    df_feat_imp_js = json.loads(df_feat_imp.to_json())
    return jsonify({'feat_imp_global': df_feat_imp_js})


# Retourner le score et la decision pour un client
# si defaillant (credit refused) sinon (credit granted)
@app.route('/score/', methods=['GET', 'POST'])
def scoring():
    # Parse the http request to get arguments ('SK_ID_CURR')
    print(str(dt.datetime.now()) + ": data_api: reception request get_score")
    id_client = int(request.args.get('id_client'))
    # data personnel pour le client (id_client)
    feat_client = df_test.loc[id_client]
    # prediction score pour le client
    score_client = round(best_model_bank.predict_proba([feat_client])[0][1], 3)
    if score_client >= best_treshold:
        select = "Refused "
    else:
        select = "Granted"
    print(str(dt.datetime.now()) + ": data_api: retour request get_score")
    return jsonify({'id_client': id_client,
                    'score': str(score_client),
                    'decision': select})


# Infos descriptives des plus proches voisins
def get_df_voisins(id_client):
    feat_client_norm = df_infos_norm.loc[id_client].drop('class').to_numpy().reshape(1, -1)
    idx = neighbors.kneighbors(feat_client_norm, return_distance=False)
    df_infos['Age'] = np.around(df_infos['DAYS_BIRTH']/-365,0) 
    df_infos['year_employed'] = np.around(df_infos['DAYS_BIRTH']/-365,0)
    df_voisins = df_infos.iloc[idx[0], :].select_dtypes('number')
    df_voisins_norm = df_infos_norm.iloc[idx[0], :]
    return df_voisins, df_voisins_norm



@app.route("/clients_similaires/", methods=["GET"])
def data_voisins():
    id_client = int(request.args.get("id_client"))
    df_voisins, df_voisins_norm = get_df_voisins(id_client)
    #df_client_norm_js = json.loads(df_client_norm.to_json())
    df_voisins_jsn = json.loads(df_voisins.to_json())  # .to_json(orient='index')
    df_voisins_norm_jsn = json.loads(df_voisins_norm.to_json())
    return jsonify({'df_voisins': df_voisins_jsn,
                    'df_voisins_norm': df_voisins_norm_jsn})


# Get locales features importance  du client shap values
@app.route("/feat_imp/", methods=["GET"])
def shap_clients():
    id_client = int(request.args.get("id_client"))
    df_test_tp = df_test.reset_index()
    index = df_test_tp.isin([id_client]).any(axis=1).idxmax()
    bool_col = df_test.select_dtypes(include='bool').columns
    df_test[bool_col]= df_test[bool_col].astype('int8')
    shap_values_json = json.dumps(clf_lgbm_us_shap_values[index][:].tolist())
    base_value_json = json.dumps(clf_lgbm_us_explainer.expected_value)
    data_json = json.dumps(df_test.loc[id_client,:].tolist())
    feature_names_json = json.dumps(df_test.columns.tolist())
    # Return data
    return jsonify({'status': 'ok',
                    'shap_values': shap_values_json,
                    'base_value': base_value_json,
                    'data': data_json, 
                    'feature_names': feature_names_json})

if __name__ == "__main__":
    app.run(port=5555, debug=True)
