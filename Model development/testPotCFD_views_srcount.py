# python ./testPotCFD_views_srcount.py --env Prod --bu ENB --queryID 3
# python ./testPotCFD_views_srcount.py --env Prod --viewID 673 --queryID 2061
from sklearn.base import TransformerMixin
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.cross_validation import *
from sklearn.grid_search import GridSearchCV
from numpy import loadtxt
from xgboost import XGBClassifier
from xgboost import plot_tree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pymongo
import json
import jsondatetime as json2
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN
import pickle
import os
from keras.models import model_from_json
from keras.models import load_model
import h5py
import datetime
import configparser
import sys
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing import text
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
import datetime

import argparse
import pylab as pl
from sklearn.metrics import roc_curve, auc
import tensorflow as tf
sys.path.insert(0, "/data/ingestion/")
from Utils import *

#Setting up the config.ini file parameters
settings = configparser.ConfigParser()
settings.read('/data/ingestion/config.ini')
stop_words = set(stopwords.words('english'))

#Parser options
options = None

def parse_options():
    parser = argparse.ArgumentParser(description="""Script to predict Potential CFD testing data""")
    parser.add_argument("--env", default="stage", help='Environment', type=str, metavar='E')
    parser.add_argument("--bu", default="", help='BU name, applicable for User defined queries', type=str, metavar='b')
    parser.add_argument("--viewID", default="", help='CQI View ID. If provide BU option will be ignored.', type=str, metavar='v')
    parser.add_argument("--queryID", default="", help='CQI Query ID', type=str, metavar='q')
    parser.add_argument("--cutoff", default="", help='Cut off probability', type=str, metavar='c')
    parser.add_argument("--cfd", default="N", help='CFD Indic', type=str, metavar='i')
    args = parser.parse_args()
    
    return args

#Ensemble model for testing
def stacking_test(test_df, cluster):
    pred_columns = ['Prediction','test_pred']
    stacking_df = test_df[pred_columns]
    
    filename = str(settings.get("Potential_CFD","temp_path_mod_potCFD")) +'/'+str(cluster)+ str(settings.get("Potential_CFD","ensemble_model"))
    with open(filename, 'rb') as f:
        model= pickle.load(f)
    
    test_x = stacking_df.as_matrix()
    y_pred = model.predict_proba(test_x)[:,1]
    
    return y_pred

#Data transformation and imputation of missing values
class DataFrameImputer(TransformerMixin):

    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].median() for c in X],
            index=X.columns)
        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

'''
    #Setting the MongoDB configurations
    hostname = settings.get('csap_prod_database', 'host')
    port = settings.get('csap_prod_database', 'dbPort')
    username = settings.get('csap_prod_database', 'user')
    passwd = settings.get('csap_prod_database', 'passwd')
    db = settings.get('csap_prod_database', 'db')
    mongo_connection_string = "mongodb://" + username + ":" + passwd + "@" + hostname + ":" + port + "/" + db
    client = pymongo.MongoClient(mongo_connection_string)
    db = client.get_database(db)

'''
def test_view(db, view, options):
    #Getting the clusters data
    view_id = options.viewID 
    query_id = options.queryID 
    bu_id = options.bu

    collection = db[settings.get('Potential_CFD', 'proj_cluster')]
    cursor = collection.find({})
    clusters =  pd.DataFrame(list(cursor))
    project_clusters = []
    groups = clusters.groupby('Cluster')
    
    for name, group in groups:
        project_clusters.append(list(group['Project']))

    print(project_clusters)

    #Fetch the data from the respective collection
    if(view):
        vi_col_name = settings.get('Potential_CFD', 'viewPrefix') + str(view_id) + '_' + str(query_id)
        tr_col_name = settings.get('Potential_CFD', 'trainPrefix')
    
    else:
        vi_col_name = settings.get('Potential_CFD', 'viewPrefix') + str(bu_id) + '_' + str(query_id)
        tr_col_name = settings.get('Potential_CFD', 'trainPrefix')

    collection = db[vi_col_name]
    print(vi_col_name)
    cursor = collection.find({})
    test_df =  pd.DataFrame(list(cursor))
    if(test_df.shape[0] == 0):
        return

    if(options.cfd == "N"):
        test_df = test_df[test_df['CFD_INDIC'] == 0]

    if(test_df.shape[0] == 0):
        return

    req_cluster = list(test_df['PROJECT'].unique())
    print(req_cluster)
    if(len(req_cluster) > 2):
        req_cluster = test_df['PROJECT'].value_counts().nlargest(1).index.tolist()
    print(req_cluster)
    print(test_df.shape[0])

    #Get the cluster number if it exists, else create new cluster
    status = False
    for a in ['CSC.sys-doc', 'CSC.autons', 'CSC.asics', 'CSC.hw', 'CSC.general', 'CSC.voice']:
        if a in req_cluster:
            req_cluster.remove(a)

    if req_cluster in project_clusters:
        status = True

    p = 0
    cluster_id = 0
    f_c = []
    for cluster in project_clusters:
        p = p + 1
        if set(req_cluster).issubset(cluster):
            cluster_id = p
            f_c = cluster
            status = True

    te_col_name = settings.get('Potential_CFD', 'testPrefix') + str(cluster_id)
    #status = True
    #cluster_id = 3
    print(cluster_id)
    print(status)
    
    if(status == True):
        #Fetching the cut_off
        print("In test_view printing cutoff"  + str(options.cutoff))
        if (options.cutoff):
          cut_off = float(options.cutoff)
        else:
          collection = db[settings.get('Potential_CFD', 'testPrefix') + str(cluster_id)]
          cursor = collection.find({})
          df =  pd.DataFrame(list(cursor))
          fpr, tpr, thresholds = roc_curve(df['IFD_CFD_INDIC'], df['Final_prediction'])
          roc_auc = auc(fpr, tpr)
          i = np.arange(len(tpr))
          roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
          r = roc.ix[(roc.tf-0).abs().argsort()[:1]]
          cut_off = list(r['thresholds'])[0]/100
        print(cut_off)

        # if(options.cutOff != ""):
        #     cut_off = int(options.cutOff)/100
        # #cut_off = 0.5
        #print(cut_off)
        #del[df]

        #Get all the saved model paths
        model1 = str(settings.get("Potential_CFD","temp_path_mod_potCFD")) + '/cluster' +str(cluster_id) + '_' + str(settings.get("Potential_CFD","xgboost_model"))
        model2 = str(settings.get("Potential_CFD","temp_path_mod_potCFD")) +'/cluster' +str(cluster_id) + '_' + str(settings.get("Potential_CFD","cnn_lstm_model"))
        model3 = str(settings.get("Potential_CFD","temp_path_mod_potCFD")) + '/' + str(settings.get("Potential_CFD","dnn_model")) + '_cluster' + str(cluster_id)+'_ticketCNT'
        model4 = str(settings.get("Potential_CFD","temp_path_mod_potCFD")) + '/' + str(settings.get("Potential_CFD","dnn_model")) + '_cluster' + str(cluster_id)+'_days'

        feature_columns_to_use = ['DE_MANAGER_USERID', 'SEVERITY_CODE', 'LIFECYCLE_STATE_CODE', 'PROJECT', 'PRODUCT', 'COMPONENT', 'ENGINEER', 'SUBMITTER_ID', 'AGE',  'FEATURE', 'RELEASE_NOTE', 'SA_ATTACHMENT_INDIC', 'CR_ATTACHMENT_INDIC', 'UT_ATTACHMENT_INDIC', 'IMPACT', 'ORIGIN', 'IS_CUSTOMER_VISIBLE', 'INCOMING_INDIC', 'BACKLOG_INDIC', 'DISPOSED_INDIC', 'TS_INDIC', 'SS_INDIC', 'OIB_INDIC', 'STATE_ASSIGN_INDIC', 'STATE_CLOSE_INDIC', 'STATE_DUPLICATE_INDIC', 'STATE_FORWARD_INDIC', 'STATE_HELD_INDIC', 'STATE_INFO_INDIC', 'STATE_JUNK_INDIC', 'STATE_MORE_INDIC', 'STATE_NEW_INDIC', 'STATE_OPEN_INDIC', 'STATE_POSTPONE_INDIC', 'STATE_RESOLVE_INDIC', 'STATE_SUBMIT_INDIC', 'STATE_UNREP_INDIC', 'STATE_VERIFY_INDIC', 'STATE_WAIT_INDIC', 'CFR_INDIC', 'S12RD_INDIC', 'S123RD_INDIC', 'MISSING_SS_EVAL_INDIC', 'S123_INDIC', 'S12_INDIC', 'RNE_INDIC', 'UPDATED_BY', 'DEV_ESCAPE_ACTIVITY', 'RELEASED_CODE', 'TEST_EDP_ACTIVITY',  'TEST_EDP_PHASE', 'RESOLVER_ANALYSIS_INDIC', 'SUBMITTER_ANALYSIS_INDIC', 'EDP_ANALYSIS_INDIC', 'RETI_ANALYSIS_INDIC', 'DESIGN_REVIEW_ESCAPE_INDIC', 'STATIC_ANALYSIS_ESCAPE_INDIC', 'FUNC_TEST_ESCAPE_INDIC', 'SELECT_REG_ESCAPE_INDIC', 'CODE_REVIEW_ESCAPE_INDIC', 'UNIT_TEST_ESCAPE_INDIC', 'DEV_ESCAPE_INDIC', 'FEATURE_TEST_ESCAPE_INDIC', 'REG_TEST_ESCAPE_INDIC', 'SYSTEM_TEST_ESCAPE_INDIC', 'SOLUTION_TEST_ESCAPE_INDIC', 'INT_TEST_ESCAPE_INDIC', 'GO_TEST_ESCAPE_INDIC', 'COMPLETE_ESCAPE_INDIC', 'SR_CNT', 'PSIRT_INDIC',  'BADCODEFLAG',   'RISK_OWNER', 'SIR', 'PSIRT_FLAG', 'URC_DISPOSED_INDIC', 'CLOSED_DISPOSED_INDIC', 'REGRESSION_BUG_FLAG']
        nonnumeric_columns = ['DE_MANAGER_USERID', 'LIFECYCLE_STATE_CODE', 'PROJECT', 'PRODUCT', 'COMPONENT', 'ENGINEER', 'SUBMITTER_ID', 'FEATURE', 'RELEASE_NOTE', 'IMPACT', 'ORIGIN', 'IS_CUSTOMER_VISIBLE', 'INCOMING_INDIC', 'BACKLOG_INDIC', 'DISPOSED_INDIC', 'UPDATED_BY', 'DEV_ESCAPE_ACTIVITY', 'RELEASED_CODE', 'TEST_EDP_ACTIVITY', 'TEST_EDP_PHASE', 'BADCODEFLAG',  'RISK_OWNER', 'SIR', 'PSIRT_FLAG', 'REGRESSION_BUG_FLAG']

        big_X = test_df[feature_columns_to_use]
        big_X = big_X.replace(np.nan, '', regex=True)
        big_X_imputed = DataFrameImputer().fit_transform(big_X)

        le = LabelEncoder()
        big_X_imputed["COMPONENT"] = big_X_imputed["COMPONENT"].astype(str)
        big_X_imputed["PRODUCT"] = big_X_imputed["PRODUCT"].astype(str)
        big_X_imputed["SUBMITTER_ID"] = big_X_imputed["SUBMITTER_ID"].astype(str)        
        for feature in nonnumeric_columns:
            big_X_imputed[feature] = big_X_imputed[feature].astype(str)  
            big_X_imputed[feature] = le.fit_transform(big_X_imputed[feature])

        thefile = str(settings.get("Potential_CFD","temp_path_mod_potCFD")) +'/'+ settings.get('Potential_CFD', 'potCFD_features')+str(cluster_id)+'.txt'
        with open (thefile, 'rb') as fp:
            feature_indices = pickle.load(fp)

        big_X_imputed = big_X_imputed.iloc[:, feature_indices]
        test_X = big_X_imputed.as_matrix()
        with open(model1, 'rb') as f:
            clf = pickle.load(f)
        test_X[test_X == ''] = 0
        #print(test_X[3090:3100])
        test_probs = clf.predict_proba(test_X)[:,1]
        print("Model 1 ran")
        test_df["Prediction"]= test_probs
        
        ##################################SECOND MODEL################################
        
        top_words = 10000
        test_data = test_df[["ENCL-Description", "Headline", "ATTRIBUTE"]]
        stemmer = LancasterStemmer()
        i = 0
        
        test_data['ATTRIBUTE']=test_data["ATTRIBUTE"].replace(np.nan, ' ')
        test_data['Headline']=test_data["Headline"].replace(np.nan, ' ')
        test_data["complete"] = test_data["ENCL-Description"].astype(str) + test_data["Headline"].astype(str)+ " "+ test_data["ATTRIBUTE"].astype(str)

        thefile = str(settings.get("Potential_CFD","temp_path_mod_potCFD")) + '/top_words_cluster_' +str(cluster_id)+'.txt'
        with open (thefile, 'rb') as fp:
            top_words = pickle.load(fp)

        f = str(settings.get("Potential_CFD","temp_path_mod_potCFD")) + '/indexes_cluster_' +str(cluster_id)+'.json'
        indexes = json.load(open(f, 'r'))

        testing_data = []
        i = 0
        for text in test_data["complete"]:
            #print(i)
            i = i + 1
            text_list = []
            if(not(pd.isnull(text))):
                for word in nltk.word_tokenize(text):
                    if word.lower() not in ["?", "'s", ">", "<", ",", ":", "'", "''", "--", "`", "``", "...", "", "!", "#", '"', '$', '%', '&', '(', ')', '*', '+', '-', '.', '/', ';', '=', '@', '[', '\\', ']', '^', '_', '{', '}', '|', '~', '\t', '\n',''] and '*' not in word.lower() and '=' not in word.lower() and '++' not in word.lower() and '___' not in word.lower() and (not word.isdigit()) and word.lower() not in stop_words and (len(word) >1):
                        stemmed_word = stemmer.stem(word.lower())
                        if stemmed_word not in top_words:
                            text_list.append(0)
                        else:
                            text_list.append(indexes[stemmed_word])
                testing_data.append(text_list)

        max_text_length = 150
        X_test = sequence.pad_sequences(testing_data, maxlen=max_text_length)

        model = load_model(model2)
        prediction = model.predict(X_test)
        print("Model 2 ran")
        test_df["test_pred"] = prediction
        test_df["Final_prediction"] = stacking_test(test_df, cluster_id)
        
        ##############################Model3##############################        
        print("Starting model 3")
        print(test_df[['Final_prediction', 'test_pred']])
        print(cut_off)
        test_df1 = test_df[test_df['test_pred'] >= float(cut_off)] #CHnage it back to Final_prediction
        #print(test_df1)        
        if(test_df1.shape[0] > 0):
            #test_df1['month_created'] = pd.to_datetime(test_df1['SUBMITTED_DATE']).dt.month
            #test_df1['year_created'] = pd.to_datetime(test_df1['SUBMITTED_DATE']).dt.year

            test_df1['COMPONENT'] = test_df1['COMPONENT'].astype(str)
            test_df1['PRODUCT'] = test_df1['PRODUCT'].astype(str)
            test_df1['SEVERITY_CODE'] = test_df1['SEVERITY_CODE'].astype(str)
            test_df1['SS_INDIC'] = test_df1['SS_INDIC'].astype(str)
            test_df1['TS_INDIC'] = test_df1['TS_INDIC'].astype(str)

            thefile = str(settings.get("Potential_CFD","temp_path_mod_potCFD")) +'/'+ settings.get('Potential_CFD', 'potCFD_features')+'dnn_'+str(cluster_id)+'.txt'
            with open (thefile, 'rb') as fp:
                new_feature_columns_to_use = pickle.load(fp)

            feature_columns_to_use = new_feature_columns_to_use #+ ['month_created', 'year_created']
            categorical_features = new_feature_columns_to_use
            continuous_features = [] #['month_created', 'year_created']

            for feature in categorical_features:
                test_df1[feature] = test_df1[feature].astype(str)

            new_test_df = test_df1[feature_columns_to_use]
            
            engineered_features = []
            for continuous_feature in continuous_features:
                engineered_features.append(tf.contrib.layers.real_valued_column(continuous_feature))

            for categorical_feature in categorical_features:
                sparse_column = tf.contrib.layers.sparse_column_with_hash_bucket(categorical_feature, hash_bucket_size=1000)
                engineered_features.append(tf.contrib.layers.embedding_column(sparse_id_column=sparse_column, dimension=16,combiner="sum"))

            regressor2 = tf.contrib.learn.DNNRegressor(feature_columns=engineered_features, hidden_units=[64, 32, 10], model_dir=model3)

            #TensorFlow input functions for Text Analysis
            def input_fn(df, training = True):
                continuous_cols = {k: tf.constant(df[k].values) for k in continuous_features}
                categorical_cols = {k: tf.SparseTensor(indices=[[i, 0] for i in range(df[k].size)],values=df[k].values,dense_shape=[df[k].size, 1]) for k in categorical_features}
                feature_cols = dict(list(continuous_cols.items()) + list(categorical_cols.items()))
                if training:
                    label = tf.constant(df[LABEL_COLUMN].values)
                    return feature_cols, label
                
                return feature_cols

            def train_input_fn():
                return input_fn(train_df1)

            def eval_input_fn():
                return input_fn(evaluate_df)

            def test_input_fn():
                return input_fn(new_test_df, False)
            
            #Predicting SR tickets
            predicted_output = regressor2.predict(input_fn=test_input_fn)#input_fn(new_test_df, False))
            test_df1['Ticket_Predictions'] = list(predicted_output)
            
            #Predicting days ahead
            regressor2 = tf.contrib.learn.DNNRegressor(feature_columns=engineered_features, hidden_units=[64, 32, 10], model_dir=model4)
            predicted_output = regressor2.predict(input_fn=test_input_fn)#input_fn(new_test_df, False))
            test_df1['Days_Predictions'] = list(predicted_output)

            now = datetime.datetime.now()
            test_df1[test_df1['Ticket_Predictions'] == 0]['Ticket_Predictions'] = 1
            test_df1[test_df1['Ticket_Predictions'] < 0]['Ticket_Predictions'] = 0
            test_df1[test_df1['Days_Predictions'] < 0]['Days_Predictions'] = 0
            #test_df1['days_ahead'] = (pd.to_datetime(test_df1['SUBMITTED_DATE']) - now)/np.timedelta64(1, 'D') + test_df1['Days_Predictions']

            test_df2 = test_df[['IDENTIFIER', 'LIFECYCLE_STATE_CODE', 'DISPOSED_INDIC', 'CFD_INDIC', 'AGE', 'ATTRIBUTE', 'COMPONENT', 'DE_MANAGER_USERID', 'ENCL-Description', 'ENGINEER', 'Headline', 'IMPACT', 'PRIORITY_CODE', 'PRODUCT','PROJECT', 'SS_INDIC','TS_INDIC', 'SEVERITY_CODE', 'SUBMITTED_DATE', 'SUBMITTER_ID', 'TICKETS_COUNT', 'VERSION_TEXT', 'IFD_CFD_INDIC','Prediction', 'test_pred', 'Final_prediction']]
            #test_df2 = test_df[['IDENTIFIER', 'LIFECYCLE_STATE_CODE', 'DISPOSED_INDIC', 'CFD_INDIC', 'Prediction', 'test_pred', 'Final_prediction']]
            test_df3 = test_df1[['IDENTIFIER', 'Ticket_Predictions', 'Days_Predictions']]

            final_test_df = pd.DataFrame()
            final_test_df = test_df2.join(test_df3.set_index('IDENTIFIER'), on='IDENTIFIER')
            final_test_df = final_test_df.drop_duplicates('IDENTIFIER')
            final_test_df['Prediction'] = final_test_df['Prediction']*100
            final_test_df['Final_prediction'] = final_test_df['test_pred']*100 #Change it back to Final_prediction
            final_test_df['test_pred'] = final_test_df['test_pred']*100
            final_test_df['days_ahead'] = (pd.to_datetime(final_test_df['SUBMITTED_DATE']) - now)/np.timedelta64(1, 'D') + final_test_df['Days_Predictions']
            final_test_df['Cluster'] = cluster_id
            final_test_df['last_run_date'] = now.strftime("%Y-%m-%d")

            final_test_df = final_test_df[final_test_df['test_pred'] >= cut_off*100] #CHnage it back to Final_prediction
            print(final_test_df.shape)
            #print(test_df1.shape)
            #Inserting data to view results collection
            if(view):
                vi_col_name_results = settings.get('Potential_CFD', 'viewPrefix') + str(view_id) + '_' + str(query_id) + '_results'
                collection = db[vi_col_name_results]
            
            else:
                vi_col_name_results = settings.get('Potential_CFD', 'viewPrefix') + str(bu_id) + '_' + str(query_id) + '_results'
                collection = db[vi_col_name_results]

            records = json2.loads(final_test_df.T.to_json(date_format='iso')).values()
            collection.create_index([("IDENTIFIER", pymongo.ASCENDING), ("last_run_date", pymongo.ASCENDING)], unique=True)
            print(collection.index_information())
            try:
                collection.insert(records)
                print("Inserted data to results collection")
            except pymongo.errors.DuplicateKeyError:
                print("Duplicates records in collection, so not inserting...")

            #Inserting data to View Mapper collection
            collection = db[settings.get('Potential_CFD', 'Pot_cfd_viewCluster')]
            df = pd.DataFrame(columns = ['viewSetCollectionName', 'trainedOnCollectionName', 'testCollectionName', 'clusterId', 'viewId', 'queryId', 'BU', 'projectList', 'csap_last_run_date', 'cutoff'])
            proj_list = ",".join(f_c)
            dat = now.strftime("%Y-%m-%d")
            #print(dat)
            if(view):
                df.loc[0] = [vi_col_name_results, tr_col_name, te_col_name, int(cluster_id), int(view_id), int(query_id), bu_id, proj_list, dat, float(cut_off*100)]
            
            else:
                print("here")
                df.loc[0] = [vi_col_name_results, tr_col_name, te_col_name, int(cluster_id), view_id, int(query_id), str(bu_id), proj_list, dat, float(cut_off*100)]

            records = json2.loads(df.T.to_json(date_format='iso')).values()
            collection.insert(records)
            print("Inserted data to View mapper collection")
        else:
            print("No predicted CFDs in this ViewSet")


def get_queries(db, config, options):
    queryList = list()
    qry = dict()
    confCollection = db[config.get('cqi_config','cqi_data')]
    print(confCollection)
    if options.viewID != "0":
        qry = { "view_enabled": "Y", "bu_enabled": "Y" , "query_enabled": "Y", "ifdcfd":"Y" }
        #qry["view_id"] = options.viewID
        #if options.queryID :
        #    qry["query_id"] = int(options.queryID)
#             qry = { "view_id": int(options.viewID), "query_id" : int(options.queryID)}
        cursor = confCollection.find(qry)
        for rec in cursor:
            #print(rec)
            queryList.append(rec)
    else:
        confCollection = db[config.get('CQI_Connect_Info','user_query_cname')]
        print(confCollection)
        #qry = { "bu": options.bu }
        #if options.queryID :
        #    qry["query_id"] = int(options.queryID)
#       #      qry = { "bu": int(options.bu), "query_id" : int(options.queryID)}
        cursor = confCollection.find(qry)
        #clusters =  pd.DataFrame(list(cursor))
        #print(clusters)
        for rec in cursor:
            #print(rec)
            queryList.append(rec)
    return queryList


def main():
    #Setting the parser options
    options = parse_options()
 
    if(options.env.lower() == "prod"):
        key = "csap_prod_database"
    
    elif(options.env.lower() == "stage"):
        key = "csap_stage_database"

    db = get_db(settings, key)
    #options.bu = 'COMPUTING SYSTEMS'
    #options.queryID = 1070
    #options.viewID = 45
    #test_view(db,True, options)
    
    if not options.queryID:
        # CQI Queries
        print("INFO: Processing CQI Queries")
        queries = get_enabled_queries(settings,db,"ifdcfd",options.viewID, options.queryID, options.bu)
        #print(queries)
        for q in queries:
            options.bu = str(q["bu_name"])
            options.queryID = int(q["query_id"])
            options.viewID = int(q["view_id"])
            if((q["ifdcfd_cutoff"] is not None) and (q["ifdcfd_cutoff"] != '')):
             options.cutoff = float(q["ifdcfd_cutoff"])
            else:
             options.cutoff = None 
            if(options.viewID == 0):
             print(options.cutoff, q["ifdcfd_cutoff"])
             test_view(db,False, options)
            else:
             print(options.cutoff, q["ifdcfd_cutoff"])
             test_view(db,True, options)
            

    elif(options.bu):
        print('inside BU')
        options.viewID = "0"
        print("in bu before queries")
        queries = get_enabled_queries( settings,db,"ifdcfd",options.viewID, options.queryID, options.bu)
        print("queries done")
        for q in queries:
            print(options.bu)
            if(str(q["bu_name"]) == options.bu):
                print(1, options.cutoff)
                if((options.cutoff is None) or (options.cutoff == '')):
                    if((q["ifdcfd_cutoff"] is not None) and (q["ifdcfd_cutoff"] != '')):
                        options.cutoff = float(q["ifdcfd_cutoff"])
                break
        print("Before calling test view " + str(options.cutoff))
        test_view(db, False, options)   
        #test_view(db, False, options)
    
    else:
        test_view(db, True, options)
    
if __name__ == "__main__":
    main()
