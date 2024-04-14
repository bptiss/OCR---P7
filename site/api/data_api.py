# -*- coding: utf-8 -*-
from flask import Flask, jsonify, request
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import shap
import datetime as dt

def memory_optimization(df):
    """
    Method used to optimize the memory usage.

    Parameters:
    -----------------
        df (pandas.DataFrame): Dataset to analyze
        
    Returns:
    -----------------
        df (pandas.DataFrame): Dataset optimized
    """ 
    
    for col in df.columns:
        if df[col].dtype == "int64" and df[col].nunique() == 2:
            df[col] = df[col].astype("int8")
            
    for col in df.columns:
        if df[col].dtype == "float64" and df[col].min() >= -2147483648 and df[col].max() <= 2147483648:
            df[col] = df[col].astype("float32")
            
    return df


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NpEncoder, self).default(obj)


# load data

df_test = pd.read_csv("../../Data/finals_datasets/df_clients_to_predict_feature_selected.csv", index_col='SK_ID_CURR')

# df_info:
df_train = pd.read_csv("../../Data/finals_datasets/df_current_clients_feature_selected.csv", index_col='SK_ID_CURR')
customers_ids = list(df_test.index.to_series())
df_feat_imp = pd.read_csv("../../Data/finals_datasets/df_fs.csv")

# SHAP: explainer of tree saved
SHAP_VALUES_FILENAME = "../../Data/finals_datasets/shap_values.csv"
clf_lgbm_us_shap_values = pickle.load(open(SHAP_VALUES_FILENAME, 'rb'))

# SHAP: expected value saved
SHAP_EXPECTED_VALUES_FILENAME = "../../Data/finals_datasets/expected_values.csv"
clf_lgbm_us_explainer = pickle.load(open(SHAP_EXPECTED_VALUES_FILENAME, 'rb'))
# print(f"clf_lgbm_us_explainer.expected_value: {clf_lgbm_us_explainer.expected_value}")



# load the model
# # Light GBM model avec balanced_class
# best_model_bank = pickle.load(open('lgb_bank_tot', 'rb'))
# LightGBM model with undersampling
best_model_bank = pickle.load(open('../../models/classif_lightGBM_undersampling.pkl', 'rb'))
# Explainer shap pour l'interprétabilité locale
# explainer = shap.TreeExplainer(best_model_bank[1])
# Seuil optimal du modèle pour classer les clients en défaillants ou non
best_treshold = 0.654


def neigb_mod(df):
    categ = df['TARGET']
    df_num = df.select_dtypes('number').drop(['business_score', 'TARGET'], axis=1)
    scaler = MinMaxScaler().fit(df_num)
    df_num_norm = pd.DataFrame(scaler.transform(df_num), columns=df_num.columns,
                               index=df.index)
    neigh = NearestNeighbors(n_neighbors=20)
    neighbors = neigh.fit(df_num_norm)
    df_num_norm['TARGET'] = categ
    print(f"df_num_norm: {df_num_norm}")
    return neighbors, df_num_norm


neighbors, df_train_norm = neigb_mod(df_train)
infos_json = json.loads(df_train.to_json())
infos_norm_json = json.loads(df_train_norm.to_json())
print(f"df_train_norm: {df_train_norm}\n\n\n\n")
# print(f"df_train_norm.dtypes: {df_train_norm.dtypes}")
app = Flask(__name__)


# retourner ids clients 'SK_ID_CURR'
@app.route('/cust_ids/')
def clients_ids():
    # Return en json format
    # print(customers_ids)
    return jsonify({'status': 'ok',
                    'data': customers_ids})


# Retourner les infos descriptives d'un client (SK_ID_CURR)
@app.route('/data_client/', methods=['GET', 'POST'])
def data_client():
    # Parse the http request to get arguments ('SK_ID_CURR')
    id_client = int(request.args.get('id_client'))
    # infos descriptives pour le client (id_client)
    feat_client = df_train.loc[id_client].drop(['business_score', 'TARGET']) #,'decision'

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
    # print(str(dt.datetime.now()) + ": data_api: reception request get_score")
    id_client = int(request.args.get('id_client'))
    # print(id_client)
    # data personnel pour le client (id_client)
    feat_client = df_test.loc[id_client,:]
    
    # prediction score pour le client
    score_client = round(best_model_bank.predict_proba([feat_client])[0][1], 3)
    
    if score_client >= best_treshold:
        select = "Refused "
    else:
        select = "Granted"
    # print(str(dt.datetime.now()) + ": data_api: retour request get_score")
    return jsonify({'id_client': id_client,
                    'score': str(score_client),
                    'decision': select})


# Infos descriptives des plus proches voisins
def get_df_voisins(id_client):
    feat_client_norm = df_infos_norm.loc[id_client].drop('TARGET').to_numpy().reshape(1, -1)
    idx = neighbors.kneighbors(feat_client_norm, return_distance=False)
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

    return jsonify({'status': 'ok',
                    'shap_values': shap_values_json,
                    'base_value': base_value_json,
                    'data': data_json, 
                    'feature_names': feature_names_json})


if __name__ == "__main__":
    app.run(port=5555, debug=True)
