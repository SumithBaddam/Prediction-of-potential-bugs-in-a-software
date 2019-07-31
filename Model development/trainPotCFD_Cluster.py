#2 changes needed. Read the features from config file and aggretare clusters using some optimal code
import jsondatetime as json2
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
import nltk
from sklearn.base import TransformerMixin
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import tensorflow as tf
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
import shutil
import argparse
import pylab as pl
from sklearn.metrics import roc_curve, auc
sys.path.insert(0, "/data/ingestion/")
from Utils import *
from functools import partial
import datetime

#Setting up the config.ini file parameters
settings = configparser.ConfigParser()
settings.read('/data/ingestion/config.ini')
stop_words = set(stopwords.words('english'))

#Parser options
options = None

def parse_options():
    parser = argparse.ArgumentParser(description="""Script to predict Potential CFD testing data""")
    parser.add_argument("--env", default="stage", help='Environment', type=str, metavar='E')
    parser.add_argument("--testdate", default="201706", help ='yyyy-mm', type=str, metavar='t')
    parser.add_argument("--cluster", default="", help ='Comma seperated cluster-ids', type=str, metavar='c')
    args = parser.parse_args()
    
    return args

#Special characters
chars = ["?", "'s", ">", "<", ",", ":", "'", "''", "--", "`", "``", "...", "", "!", "#", '"', '$', '%', '&', '(', ')', '*', '+', '-', '.', '/', ';', '=', '@', '[', '\\', ']', '^', '_', '{', '}', '|', '~', '\t', '\n', '', 'user', 'use', 'name', 'the', 'set', 'age', 'get', 'set', 'id', 'case', 'this', 'that', 'they', 'etc']
def get_word(word):
    if word not in chars and '*' not in word and '=' not in word and '++' not in word and '___' not in word and (not word.isdigit()):
        if(word not in stop_words and len(word) > 1):
            return True
    #print(word)
    return False

#Ensemble model for training
def stacking_train(df, cluster):
    pred_columns=['IFD_CFD_INDIC','Prediction','test_pred']
    stacking_df=df[pred_columns]
    train_x = stacking_df.drop('IFD_CFD_INDIC', axis=1).as_matrix()
    train_y = stacking_df['IFD_CFD_INDIC'].as_matrix()
    X_resampled, y_resampled = SMOTE(kind='borderline1').fit_sample(train_x, train_y) #SMOTE
    model = XGBClassifier(objective='binary:logistic')
    model.fit(X_resampled, y_resampled)
    print('fitting done')
    filename = str(settings.get("Potential_CFD","temp_path_mod_potCFD")) + '/' + str(cluster) + str(settings.get("Potential_CFD","ensemble_model"))
    if os.path.exists(filename):
        os.remove(filename)
        print("File Removed!")    
    with open(filename, 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    y_pred = model.predict_proba(train_x)[:,1]   
    return y_pred

#Ensemble model for testing
def stacking_test(test_df, cluster):
    pred_columns=['Prediction','test_pred']
    stacking_df=test_df[pred_columns]
    filename = str(settings.get("Potential_CFD","temp_path_mod_potCFD")) + '/' + str(cluster) + str(settings.get("Potential_CFD","ensemble_model"))
    with open(filename, 'rb') as f:
        model= pickle.load(f)   
    test_x = stacking_df.as_matrix()
    y_pred = model.predict_proba(test_x)[:,1]   
    return y_pred

#XGBoost model for training and testing with the hyperparameters
def xgb_training(train_X, train_y, test_X, feature_selection, p):
    X_resampled, y_resampled = train_X, train_y
    clf = XGBClassifier(objective='binary:logistic')
    clf.fit(X_resampled, y_resampled)   
    if(feature_selection == True):
        model = clf #clf.best_estimator_
        print(model.feature_importances_)
        t = model.feature_importances_ > 0
        print(sum(t))
        feature_indices = [i for i, x in enumerate(t) if x]
        return feature_indices    
    else:
        filename = str(settings.get("Potential_CFD","temp_path_mod_potCFD")) + '/cluster'+str(p)+ '_' + str(settings.get("Potential_CFD","xgboost_model"))
        
        if os.path.exists(filename):
            os.remove(filename)
            print("File Removed!")        
        with open(filename, 'wb') as handle:
            pickle.dump(clf, handle, protocol=pickle.HIGHEST_PROTOCOL)        
        test_probs = clf.predict_proba(test_X)[:,1]
        train_probs = clf.predict_proba(train_X)[:,1]       
        return test_probs, train_probs

#Data transformation and imputation of missing values
class DataFrameImputer(TransformerMixin):
    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].median() for c in X],
            index=X.columns)       
        return self
    def transform(self, X, y=None):
        return X.fillna(self.fill)

def train_model(db, cluster, test_date):
    #Getting the clusters data
    collection = db[settings.get('Potential_CFD', 'proj_cluster')]
    cursor = collection.find({})
    clusters =  pd.DataFrame(list(cursor))
    project_clusters = []
    cluster_status = True
    groups = clusters.groupby('Cluster')

    for name, group in groups:
        project_clusters.append(list(group['Project']))

    #db.Project_Clusters.aggregate([{$group :{"_id": {"Cluster":"$Cluster"} ,"Projects" : {$push: "$Project"}}}])

    p = 0
    if(cluster != "all"):
        cluster_status = False
        clusters = cluster.split(',')
        req_cluster = []    
        for i in range(0, len(clusters)):
            req_cluster.append(project_clusters[int(clusters[i]) - 1])  
        project_clusters = req_cluster

    print(project_clusters)
    #Problem is how to manage cluster ID ?? p won't be needed now. Something else needs to be there
    #Running the 3 models at cluster level...
    for cluster in project_clusters:
        if(not cluster_status):
            p = int(clusters[0])
            clusters = clusters[1:]
        else:
            p = p + 1
        print('Running on cluster ', p)
        print(cluster)
        df = pd.DataFrame()
        #Fetching the data for each project in the cluster
        cluster = cluster[:4]#['CSC.datacenter']
        for proj in cluster:
            df2 = pd.DataFrame()
            collection = db[settings.get('Potential_CFD', 'trainPrefix') + proj.replace('.', '_')]
            cursor = collection.find({})
            if(cursor.count() > 300000):
                cursor = collection.find().limit(300000)
            print(proj)
            df2 =  pd.DataFrame(list(cursor))
            df = df.append(df2)
                #df3 = pd.read_csv('/auto/vgapps-cstg02-vapps/analytics/csap/ingestion/opfiles/potCFD/Train/180505/CSC.labtrunk/BugFinal.csv')
                #print(df3['SUBMITTED_DATE'])
                #to_datetime_fmt = partial(pd.to_datetime, format='%Y-%m-%d %H:%M:%S')
                #df3['SUBMITTED_DATE'] = df3['SUBMITTED_DATE'].apply(to_datetime_fmt)
                #print(df3['SUBMITTED_DATE'])
                #print(df['SUBMITTED_DATE'])
                #df = df.append(df3)
                #cluster = ['CSC.sys','CSC.labtrunk', 'CSC.sibu.dev']
        df = df[df['LIFECYCLE_STATE_CODE'].isin(['C', 'J', 'U', 'D', 'M', 'R', 'V'])]

        print(df['PROJECT'].unique())

        #Test and train split
        #test_date = options.testdate[:4] + '-' + options.testdate[4:]
        #print(test_date)
        test_df = df[df['SUBMITTED_DATE'] >= str(test_date)]
        train_df = df
        #print(list(test_df.columns))

        majority = train_df[train_df["IFD_CFD_INDIC"] == 0]
        minority = train_df[train_df["IFD_CFD_INDIC"] == 1]
        majority = majority.sample(n=len(minority)*3)

        train_df = majority
        train_df = train_df.append(minority)

        print(train_df.shape)
        print(test_df.shape)
        del [[df, majority, minority]]

        #These are the set of columns from which the model need to choose the best features
        feature_columns_to_use = ['DE_MANAGER_USERID', 'SEVERITY_CODE', 'PROJECT', 'PRODUCT', 'COMPONENT', 'ENGINEER', 'SUBMITTER_ID', 'AGE',  'FEATURE', 'RELEASE_NOTE', 'SA_ATTACHMENT_INDIC', 'CR_ATTACHMENT_INDIC', 'UT_ATTACHMENT_INDIC', 'IMPACT', 'ORIGIN', 'IS_CUSTOMER_VISIBLE', 'INCOMING_INDIC', 'BACKLOG_INDIC', 'DISPOSED_INDIC', 'TS_INDIC', 'SS_INDIC', 'OIB_INDIC', 'STATE_ASSIGN_INDIC', 'STATE_CLOSE_INDIC', 'STATE_DUPLICATE_INDIC', 'STATE_FORWARD_INDIC', 'STATE_HELD_INDIC', 'STATE_INFO_INDIC', 'STATE_JUNK_INDIC', 'STATE_MORE_INDIC', 'STATE_NEW_INDIC', 'STATE_OPEN_INDIC', 'STATE_POSTPONE_INDIC', 'STATE_RESOLVE_INDIC', 'STATE_SUBMIT_INDIC', 'STATE_UNREP_INDIC', 'STATE_VERIFY_INDIC', 'STATE_WAIT_INDIC', 'CFR_INDIC', 'S12RD_INDIC', 'S123RD_INDIC', 'MISSING_SS_EVAL_INDIC', 'S123_INDIC', 'S12_INDIC', 'RNE_INDIC', 'UPDATED_BY', 'DEV_ESCAPE_ACTIVITY', 'RELEASED_CODE', 'TEST_EDP_ACTIVITY',  'TEST_EDP_PHASE', 'RESOLVER_ANALYSIS_INDIC', 'SUBMITTER_ANALYSIS_INDIC', 'EDP_ANALYSIS_INDIC', 'RETI_ANALYSIS_INDIC', 'DESIGN_REVIEW_ESCAPE_INDIC', 'STATIC_ANALYSIS_ESCAPE_INDIC', 'FUNC_TEST_ESCAPE_INDIC', 'SELECT_REG_ESCAPE_INDIC', 'CODE_REVIEW_ESCAPE_INDIC', 'UNIT_TEST_ESCAPE_INDIC', 'DEV_ESCAPE_INDIC', 'FEATURE_TEST_ESCAPE_INDIC', 'REG_TEST_ESCAPE_INDIC', 'SYSTEM_TEST_ESCAPE_INDIC', 'SOLUTION_TEST_ESCAPE_INDIC', 'INT_TEST_ESCAPE_INDIC', 'GO_TEST_ESCAPE_INDIC', 'COMPLETE_ESCAPE_INDIC', 'SR_CNT', 'PSIRT_INDIC',  'BADCODEFLAG',   'RISK_OWNER', 'SIR', 'PSIRT_FLAG', 'URC_DISPOSED_INDIC', 'CLOSED_DISPOSED_INDIC', 'REGRESSION_BUG_FLAG']
        nonnumeric_columns = ['DE_MANAGER_USERID', 'PROJECT', 'PRODUCT', 'COMPONENT', 'ENGINEER', 'SUBMITTER_ID', 'FEATURE', 'RELEASE_NOTE', 'IMPACT', 'ORIGIN', 'IS_CUSTOMER_VISIBLE', 'INCOMING_INDIC', 'BACKLOG_INDIC', 'DISPOSED_INDIC', 'UPDATED_BY', 'DEV_ESCAPE_ACTIVITY', 'RELEASED_CODE', 'TEST_EDP_ACTIVITY', 'TEST_EDP_PHASE', 'BADCODEFLAG',  'RISK_OWNER', 'SIR', 'PSIRT_FLAG', 'REGRESSION_BUG_FLAG']

        #Data imputation
        big_X = train_df[feature_columns_to_use].append(test_df[feature_columns_to_use])
        big_X = big_X.replace(np.nan, '', regex=True)
        big_X_imputed = DataFrameImputer().fit_transform(big_X.iloc[:,:])

        le = LabelEncoder()

        for feature in nonnumeric_columns:
            big_X_imputed[feature] = big_X_imputed[feature].astype(str)

        for feature in nonnumeric_columns:
            big_X_imputed[feature] = le.fit_transform(big_X_imputed[feature])

        train_X = big_X_imputed[0:train_df.shape[0]].as_matrix()
        test_X = big_X_imputed[train_df.shape[0]::].as_matrix()
        train_y = train_df['IFD_CFD_INDIC']
        train_X[train_X == ''] = 0
        test_X[test_X == ''] = 0

        #Running the model and hypertuning to find the best features
        feature_indices = xgb_training(train_X, train_y, test_X, True, p)

        new_features_list = []
        for i in feature_indices:
            new_features_list.append(feature_columns_to_use[i])

        #Dumping the features to a file
        thefile = str(settings.get("Potential_CFD","temp_path_mod_potCFD")) + '/' + settings.get('Potential_CFD', 'potCFD_features') + str(p) + '.txt'
        with open(thefile, 'wb') as fp:
            pickle.dump(feature_indices, fp, protocol=2)

        #Dumping the features into a collection
        features_df = pd.DataFrame(columns = ['clusterId', 'features_list', 'date'])
        features_df.loc[0] = [p, ",".join(new_features_list), datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        coll = 'PotCFD_Features'
        records = json2.loads(features_df.T.to_json(date_format='iso')).values()
        db[coll].insert(records)

        #Data Imputation
        big_X_imputed = big_X_imputed.iloc[:, feature_indices]
        train_X = big_X_imputed[0:train_df.shape[0]].as_matrix()
        test_X = big_X_imputed[train_df.shape[0]::].as_matrix()
        train_y = train_df['IFD_CFD_INDIC']
        train_X[train_X == ''] = 0
        test_X[test_X == ''] = 0

        #Running the model
        test_probs, train_probs = xgb_training(train_X, train_y, test_X, False, p)
        print("Model 1 ran")
        test_df["Prediction"]= test_probs
        train_df["Prediction"]= train_probs

        ##################################SECOND MODEL - TEXT##################################
        train_data = train_df[["ENCL-Description", "Headline", "ATTRIBUTE", "IFD_CFD_INDIC"]]
        test_data = test_df[["ENCL-Description", "Headline", "ATTRIBUTE", "IFD_CFD_INDIC"]]

        train_data['ATTRIBUTE'] = train_data["ATTRIBUTE"].replace(np.nan, ' ')
        test_data['ATTRIBUTE'] = test_data["ATTRIBUTE"].replace(np.nan, ' ')
        train_data['Headline'] = train_data["Headline"].replace(np.nan, ' ')
        test_data['Headline'] = test_data["Headline"].replace(np.nan, ' ')
        train_data['ENCL-Description'] = train_data["ENCL-Description"].replace(np.nan, ' ')
        test_data['ENCL-Description'] = test_data["ENCL-Description"].replace(np.nan, ' ')

        #Compiling all the text data into single column
        train_data["complete"] = train_data["ENCL-Description"].astype(str) + " " + train_data["Headline"].astype(str) + " "+ train_data["ATTRIBUTE"].astype(str)
        test_data["complete"] = test_data["ENCL-Description"].astype(str) + " "+ test_data["Headline"].astype(str) + " " + test_data["ATTRIBUTE"].astype(str)

        top_words = 10000

        corpus_words = []
        unique_words = []
        stemmer = SnowballStemmer("english")
        i = 0

        #Building the vocabulary set
        for text in train_data["complete"]:
            #print(i)
            i = i + 1
            for word in nltk.word_tokenize(text):
                case = get_word(word.lower())
                if case:
                    stemmed_word = stemmer.stem(word.lower())
                    if(stemmed_word == 'is'):
                        print(word)
                    corpus_words.append(stemmed_word)
                    if stemmed_word not in unique_words:
                        unique_words.append(stemmed_word)

        fdist1 = FreqDist(corpus_words)
        num_words = 5000
        top_words_freq = fdist1.most_common(num_words)

        top_words = []
        for word in top_words_freq:
            top_words.append(word[0])

        thefile = str(settings.get("Potential_CFD","temp_path_mod_potCFD")) + '/top_words_cluster_' +str(p)+'.txt'
        with open(thefile, 'wb') as fp:
            pickle.dump(top_words, fp, protocol=2)

        indexes = {}
        i = 1
        for word in top_words:
            indexes[word] = i
            i = i + 1

        f = str(settings.get("Potential_CFD","temp_path_mod_potCFD")) + '/indexes_cluster_' +str(p)+'.json'
        f = open(f,'w')
        json1 = json.dumps(indexes, indent=4)
        f.write(json1)
        f.close()

        #Mapping the vocabulary and indices for both train and test datasets
        training_data = []
        i = 0
        for text in train_data["complete"]:
            #print(i)
            i = i + 1
            text_list = []
            for word in nltk.word_tokenize(text):
                case = get_word(word)
                if case: #word not in chars and '*' not in word and '=' not in word and '++' not in word and '___' not in word and (not word.isdigit()):
                    stemmed_word = stemmer.stem(word.lower())
                    if stemmed_word not in top_words:
                        text_list.append(0)
                    else:
                        text_list.append(indexes[stemmed_word])
            training_data.append(text_list)

        testing_data = []
        i = 0
        for text in test_data["complete"]:
            #print(i)
            i = i + 1
            text_list = []
            if(not(pd.isnull(text))):
                for word in nltk.word_tokenize(text):
                    case = get_word(word)
                    if case: #word not in chars and '*' not in word and '=' not in word and '++' not in word and '___' not in word and (not word.isdigit()):
                        stemmed_word = stemmer.stem(word.lower())
                        if stemmed_word not in top_words:
                            text_list.append(0)
                        else:
                            text_list.append(indexes[stemmed_word])
                testing_data.append(text_list)

        #Getting the train and test labels
        y_train = train_data.IFD_CFD_INDIC
        y_test = test_data.IFD_CFD_INDIC

        #Restricting the sentence size to 150
        max_text_length = 150

        #Padding the shorter sentences
        X_train = sequence.pad_sequences(training_data, maxlen=max_text_length)
        X_test = sequence.pad_sequences(testing_data, maxlen=max_text_length)

        #Building the CNN-LSTM model with embedding layer on top    
        embedding_vecor_length = 32
        model = Sequential()
        model.add(Embedding(num_words+1, embedding_vecor_length, input_length=max_text_length))
        model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        print(model.summary())

        #Fitting the data into the model
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2, batch_size=64)
        print("Model 2 ran")

        filename = str(settings.get("Potential_CFD","temp_path_mod_potCFD")) +'/cluster'+str(p)+'_'+ str(settings.get("Potential_CFD","cnn_lstm_model"))#'/auto/vgapps-cstg02-vapps/analytics/csap/ingestion/scripTS_INDIC/tesTS_INDIC/model2_cluster'+str(p)+'.h5'
        if os.path.exists(filename):
            os.remove(filename)
            print("File Removed!")

        #Saving the model
        model.save(filename)

        #Model predictions
        prediction = model.predict(X_test)

        test_df["test_pred"] = prediction
        prediction_tr = model.predict(X_train)
        train_df["test_pred"] = prediction_tr

        #Getting the final prediction using ensemble methods
        train_df["Final_prediction"] = stacking_train(train_df, p)
        test_df["Final_prediction"] = stacking_test(test_df, p)

        ###############################Third model - Prediction of time###############################

        #Calculating the optimal cut-off threshold
        fpr, tpr, thresholds = roc_curve(test_df['IFD_CFD_INDIC'], test_df['Final_prediction'])
        roc_auc = auc(fpr, tpr)
        i = np.arange(len(tpr))
        roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
        r = roc.ix[(roc.tf-0).abs().argsort()[:1]]
        cut_off = list(r['thresholds'])[0]

        train_df1 = train_df[train_df['IFD_CFD_INDIC'] == 1]
        test_df1 = test_df[test_df['Final_prediction'] > cut_off]

        train_df_days = train_df1.copy()
        test_df_days = test_df1.copy()

        train_df_days['days_taken'] = (pd.to_datetime(train_df_days['IFD_CFD_INDIC_DATE']) - pd.to_datetime(train_df_days['SUBMITTED_DATE'])).dt.days
        test_df_days['days_taken'] = (pd.to_datetime(test_df_days['IFD_CFD_INDIC_DATE']) - pd.to_datetime(test_df_days['SUBMITTED_DATE'])).dt.days

        #Making changes here to train on TICKETS_COUNT instead of days_taken

        continuous_features = [] #['month_created', 'year_created']
        train_df1['TICKETS_COUNT'] = train_df1['TICKETS_COUNT'].astype(float)

        new_feature_columns_to_use = []
        for feature in new_features_list:
            if(sum(train_df1[feature].isnull()) == 0):
                new_feature_columns_to_use.append(feature)

        thefile = str(settings.get("Potential_CFD","temp_path_mod_potCFD")) + '/' + settings.get('Potential_CFD', 'potCFD_features') + 'dnn_' + str(p) + '.txt'
        with open(thefile, 'wb') as fp:
            pickle.dump(new_feature_columns_to_use, fp, protocol=2)

        #TensorFlow input functions for Text Analysis
        def input_fn(df, training = True):
            continuous_cols = {k: tf.constant(df[k].values) for k in continuous_features}
            categorical_cols = {k: tf.SparseTensor(indices=[[i, 0] for i in range(df[k].size)],values=df[k].values,dense_shape=[df[k].size, 1]) for k in categorical_features}
            feature_cols = dict(list(continuous_cols.items()) + list(categorical_cols.items()))
            
            if training:
                label = tf.constant(df[LABEL_COLUMN].values)
                #print('done')
                return feature_cols, label
            
            return feature_cols

        def train_input_fn():
            return input_fn(train_df1)

        def eval_input_fn():
            return input_fn(evaluate_df)

        def test_input_fn():
            return input_fn(new_test_df, False)

        feature_columns_to_use = new_feature_columns_to_use[:] + ['TICKETS_COUNT'] #['days_taken','month_created', 'year_created']

        categorical_features = new_feature_columns_to_use[:]
        #categorical_features =feature_columns_to_use
        for feature in categorical_features:
            train_df1[feature] = train_df1[feature].astype(str)
            test_df1[feature] = test_df1[feature].astype(str)

        train_df1['TICKETS_COUNT'] = train_df1['TICKETS_COUNT'].astype(float)
        train_df1 = train_df1[feature_columns_to_use]
        y_train = train_df1['TICKETS_COUNT'].astype(float)
        new_test_df = test_df1[feature_columns_to_use]
        y_test1 = test_df1['TICKETS_COUNT'].astype(float)

                #Building the features for Deep Neural Networks
        LABEL_COLUMN = 'TICKETS_COUNT'

        engineered_features = []
        for continuous_feature in continuous_features:
            engineered_features.append(tf.contrib.layers.real_valued_column(continuous_feature))

        for categorical_feature in categorical_features:
            sparse_column = tf.contrib.layers.sparse_column_with_hash_bucket(categorical_feature, hash_bucket_size=1000)
            engineered_features.append(tf.contrib.layers.embedding_column(sparse_id_column=sparse_column, dimension=16,combiner="sum"))

        MODEL_DIR = str(settings.get("Potential_CFD","temp_path_mod_potCFD")) + '/'+str(settings.get("Potential_CFD","dnn_model"))+'_cluster'+str(p)+'_ticketCNT'
        print(MODEL_DIR)
        if os.path.exists(MODEL_DIR):
            shutil.rmtree(MODEL_DIR, ignore_errors=False, onerror=None)

        regressor = tf.contrib.learn.DNNRegressor(feature_columns=engineered_features, hidden_units=[64, 32, 10], model_dir=MODEL_DIR)

        #Model fitting
        wrap = regressor.fit(input_fn=train_input_fn, steps=2000)
        print("Model 3 ran")

        #Prediction
        predicted_output = wrap.predict(input_fn=test_input_fn)
        test_df1['Ticket_Predictions'] = list(predicted_output)

        #PREDICTING DAYS AS WELL
        #Imputing the month, year and days taken fields
        #train_df1 = train_df[train_df['IFD_CFD_INDIC'] == 1]
        #test_df1 = test_df[test_df['Final_prediction'] > cut_off]
        #train_df1['month_created'] = pd.to_datetime(train_df1['SUBMITTED_DATE']).dt.month
        #train_df1['year_created'] = pd.to_datetime(train_df1['SUBMITTED_DATE']).dt.year
        #test_df1['month_created'] = pd.to_datetime(test_df1['SUBMITTED_DATE']).dt.month
        #test_df1['year_created'] = pd.to_datetime(test_df1['SUBMITTED_DATE']).dt.year
        #train_df1['days_taken'] = (pd.to_datetime(train_df1['IFD_CFD_INDIC_DATE']) - pd.to_datetime(train_df1['SUBMITTED_DATE'])).dt.days
        #test_df1['days_taken'] = (pd.to_datetime(test_df1['IFD_CFD_INDIC_DATE']) - pd.to_datetime(test_df1['SUBMITTED_DATE'])).dt.days

        #Converting object dtype to string dtype
        '''
        train_df1['COMPONENT'] = train_df1['COMPONENT'].astype(str)
        train_df1['PRODUCT'] = train_df1['PRODUCT'].astype(str)
        train_df1['SEVERITY_CODE'] = train_df1['SEVERITY_CODE'].astype(str)
        train_df1['SS_INDIC'] = train_df1['SS_INDIC'].astype(str)
        train_df1['TS_INDIC'] = train_df1['TS_INDIC'].astype(str)
        test_df1['COMPONENT'] = test_df1['COMPONENT'].astype(str)
        test_df1['PRODUCT'] = test_df1['PRODUCT'].astype(str)
        test_df1['SEVERITY_CODE'] = test_df1['SEVERITY_CODE'].astype(str)
        test_df1['SS_INDIC'] = test_df1['SS_INDIC'].astype(str)
        test_df1['TS_INDIC'] = test_df1['TS_INDIC'].astype(str)
        '''
        #continuous_features = ['month_created', 'year_created']
        train_df_days['days_taken'] = train_df_days['days_taken'].fillna(0)

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
            return input_fn(train_df_days)

        def eval_input_fn():
            return input_fn(evaluate_df)

        def test_input_fn():
            return input_fn(new_test_df, False)

        feature_columns_to_use = new_feature_columns_to_use[:] + ['days_taken']
        categorical_features = new_feature_columns_to_use[:]

        for feature in categorical_features:
            train_df_days[feature] = train_df_days[feature].astype(str)
            test_df_days[feature] = test_df_days[feature].astype(str)

        train_df_days = train_df_days[feature_columns_to_use]
        y_train = train_df_days['days_taken']
        new_test_df = test_df_days[feature_columns_to_use]
        y_test1 = test_df_days['days_taken']

        #Building the features for Deep Neural Networks
        LABEL_COLUMN = 'days_taken'

        engineered_features = []
        for continuous_feature in continuous_features:
            engineered_features.append(tf.contrib.layers.real_valued_column(continuous_feature))

        for categorical_feature in categorical_features:
            sparse_column = tf.contrib.layers.sparse_column_with_hash_bucket(categorical_feature, hash_bucket_size=1000)
            engineered_features.append(tf.contrib.layers.embedding_column(sparse_id_column=sparse_column, dimension=16,combiner="sum"))

        MODEL_DIR = str(settings.get("Potential_CFD","temp_path_mod_potCFD")) + '/'+str(settings.get("Potential_CFD","dnn_model"))+'_cluster'+str(p)+'_days'
        print(MODEL_DIR)
        if os.path.exists(MODEL_DIR):
            shutil.rmtree(MODEL_DIR, ignore_errors=False, onerror=None)

        regressor = tf.contrib.learn.DNNRegressor(feature_columns=engineered_features, hidden_units=[64, 32, 10], model_dir=MODEL_DIR)

        #Model fitting
        wrap = regressor.fit(input_fn=train_input_fn, steps=7000)
        print("Model 3 ran")

        #Prediction
        predicted_output = wrap.predict(input_fn=test_input_fn)
        test_df1['Days_Predictions'] = list(predicted_output)
        test_df1['days_taken'] = test_df_days['days_taken']


        #Calculating the days remaining
        now = datetime.datetime.now()
        test_df1['Ticket_Predictions'] = test_df1['Ticket_Predictions'].fillna(0)
        test_df1['Ticket_Predictions'] = test_df1['Ticket_Predictions'].round().astype(int)
        test_df1['Days_Predictions'] = test_df1['Days_Predictions'].round().astype(int)
        test_df1[test_df1['Days_Predictions'] < 0]['Days_Predictions'] = 0
        test_df1[test_df1['Ticket_Predictions'] < 0]['Ticket_Predictions'] = 0
        #test_df1['days_ahead'] = (test_df1['SUBMITTED_DATE'] - now )/np.timedelta64(1, 'D') + test_df1['Days_Predictions']
        test_df2 = test_df[['IDENTIFIER','IFD_CFD_INDIC','Final_prediction', 'FOUND_DURING']]
        test_df3 = test_df1[['IDENTIFIER', 'Ticket_Predictions', 'TICKETS_COUNT', 'Days_Predictions', 'days_taken']]

        final_test_df = test_df2.join(test_df3.set_index('IDENTIFIER'), on='IDENTIFIER')
        final_test_df = final_test_df.drop_duplicates('IDENTIFIER')
        final_test_df['Final_prediction'] = final_test_df['Final_prediction']*100
        print("Predictions completed...")

                #Writing the test data into a collection
        final_test_df.reset_index(drop = True, inplace = True)
        records = json2.loads(final_test_df.T.to_json(date_format='iso')).values()
        test_collection = settings.get('Potential_CFD', 'testPrefix')+ str(p)+'_ticketCNT'
        print(db[test_collection], len(records))
        print(final_test_df.iloc[3])
        db[test_collection].drop()
        db[test_collection].insert_many(records)
        print("Inserted to collection")

'''
def get_collection_details():
    #Setting the MongoDB configurations
    hostname = settings.get('csap_prod_database', 'host')
    port = settings.get('csap_prod_database', 'dbPort')
    username = settings.get('csap_prod_database', 'user')
    passwd = settings.get('csap_prod_database', 'passwd')
    db = settings.get('csap_prod_database', 'db')
	mongo_connection_string = "mongodb://csaprw:csaprw123@sjc-wwpl-fas4.cisco.com:27017/?authSource=csap_prd"
	client=pymongo.MongoClient(mongo_connection_string)
	db=client.get_database('csap_prd')
    return db
'''

def main():
    #db = get_collection_details()
    options = parse_options()
    
    if(options.env.lower() == "prod"):
        key = "csap_prod_database"
    
    elif(options.env.lower() == "stage"):
        key = "csap_stage_database"
    
    db = get_db(settings, key)
    print(db)
    print(options.testdate)
    test_date = options.testdate[:4] + '-' + options.testdate[4:]
    print(test_date)
    
    train_model(db, options.cluster, str(test_date))


if __name__ == "__main__":
    main()

