import pandas as pd
import xgboost as xgb #If you want to import, we need to have xgboost folder in our directory...
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

username = "csaprw"
passwd = "csaprw123"
hostname = "sjc-wwpl-fas4"
port = "27017"
db = "csap_prd"

mongo_connection_string="mongodb://"+username+":"+passwd+"@"+hostname+":"+port+"/"+db
client=pymongo.MongoClient(mongo_connection_string)
db=client.get_database(db)
collection = db["Projects_list"]
cursor = collection.find({}) # query
projects =  pd.DataFrame(list(cursor))
project_list = projects.Project.unique()

collection = db["PotentialCFD"]#db["Pot_CFD1_testprd150101"]
'''df = pd.DataFrame()
for project in project_list:
    cursor = collection.find({"Project": project}) # query
    df2 =  pd.DataFrame(list(cursor))
    df = df.append(df2)
'''
#df=pd.read_csv("PotentialCFD.csv", encoding = 'utf-8')

from sklearn.base import TransformerMixin
class DataFrameImputer(TransformerMixin):
    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].median() for c in X],
            index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.fill)

feature_columns_to_use = ['Component', 'Product', 'Submitter', 'Age', 'Impact', 'Engineer', 'DE-manager', 'SS', 'TS', 'Version', 'Tickets-count', 'Found'] #Severity not working
nonnumeric_columns = ['Component', 'Product', 'Submitter',  'Impact', 'Engineer', 'DE-manager', 'Version', 'SS', 'TS', 'Found']

import gc

for project in project_list:
    cursor = collection.find({"Project": project})
    train_df = pd.DataFrame(list(cursor))
    #train_df = df[df["Project"]==project]
    big_X = train_df[feature_columns_to_use]#.append(test_df[feature_columns_to_use])
    big_X_imputed = DataFrameImputer().fit_transform(big_X)
    le = LabelEncoder()
    big_X_imputed["Product"] = big_X_imputed["Product"].astype(str)
    big_X_imputed["Version"] = big_X_imputed["Version"].astype(str)
    big_X_imputed["Component"] = big_X_imputed["Component"].astype(str)
    for feature in nonnumeric_columns:
        big_X_imputed[feature] = le.fit_transform(big_X_imputed[feature])
    train_X = big_X_imputed[0:train_df.shape[0]].as_matrix()
    #test_X = big_X_imputed[train_df.shape[0]::].as_matrix()
    train_y = train_df['IFD_CFD_INDIC']
    xgb_model = xgb.XGBClassifier()
    xgb_model.fit(train_X, train_y)
    plot_tree(xgb_model)
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(150, 150)
    fig.savefig("plots/"+project+'.png')
    print("Plotted " + project + " image")
