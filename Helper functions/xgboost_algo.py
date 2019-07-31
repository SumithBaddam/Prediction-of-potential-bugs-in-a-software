# For parameter tuning use this URL...will be very useful
# http://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
#We run it in UBUNTU 16.04...Not working here
import pandas as pd
import xgboost as xgb #If you want to import, we need to have xgboost folder in our directory...
from sklearn.preprocessing import LabelEncoder
import numpy as np

'''
import pymongo
import pandas as pd
username = "csaprw"
passwd = "csaprw123"
hostname = "sjc-wwpl-fas4"
port = "27017"
db = "csap_prd"

mongo_connection_string="mongodb://"+username+":"+passwd+"@"+hostname+":"+port+"/"+db
client=pymongo.MongoClient(mongo_connection_string)
db=client.get_database(db)
collection = db["Pot_CFD1_testprd150101"]
cursor = collection.find({}) # query
df =  pd.DataFrame(list(cursor))

from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(df, test_size = 0.2)

a = train_df[train_df["IFD_CFD_INDIC"]==0]
b = train_df[train_df["IFD_CFD_INDIC"]==1]
a = a.sample(n=len(b))
train_df = a
train_df = train_df.append(b)

a = test_df[test_df["IFD_CFD_INDIC"]==0]
b = test_df[test_df["IFD_CFD_INDIC"]==1]
a = a.sample(n=len(b))
test_df = a
test_df = test_df.append(b)

'''

# Load the data from Mongo, create month and year, downsample the data, split it into train and test
train_df = pd.read_csv('Train_data.csv', header=0)
test_df = pd.read_csv('Test_data.csv', header=0)
#print train_df[0:3]
##print train_df.columns

# Imputing the missing values for numerical(median) and string(mode) attributes
from sklearn.base import TransformerMixin
class DataFrameImputer(TransformerMixin):
    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].median() for c in X],
            index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.fill)
        
#feature_columns_to_use = ['Component', 'Product', 'Severity', 'Assigner', 'Submitter', 'Status', 'month', 'year'] #
feature_columns_to_use = ['Component', 'Product', 'Submitter', 'Age', 'Impact', 'Engineer', 'DE-manager', 'Priority'] #Severity not working

#nonnumeric_columns = ['Component', 'Product', 'Assigner', 'Submitter', 'Status'] #
nonnumeric_columns = ['Component', 'Product', 'Submitter',  'Impact', 'Engineer', 'DE-manager', 'Priority']

big_X = train_df[feature_columns_to_use].append(test_df[feature_columns_to_use])
big_X_imputed = DataFrameImputer().fit_transform(big_X)

# XGBoost doesn't (yet) handle categorical features automatically, so we need to change
# them to columns of integer values.
# See http://scikit-learn.org/stable/modules/preprocessing.html#preprocessing for more
# details and options
le = LabelEncoder()
for feature in nonnumeric_columns:
    big_X_imputed[feature] = le.fit_transform(big_X_imputed[feature])

# Prepare the inputs for the model
train_X = big_X_imputed[0:train_df.shape[0]].as_matrix()
test_X = big_X_imputed[train_df.shape[0]::].as_matrix()
train_y = train_df['IFD_CFD_INDIC']

# This example uses the current build of XGBoost, from https://github.com/dmlc/xgboost
gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(train_X, train_y)
predictions = gbm.predict(test_X)
from sklearn.metrics import confusion_matrix
confusion_matrix(test_df['IFD_CFD_INDIC'], predictions) #77% precision

##############################With SMOTE#####################################
complete_data = pd.read_csv("MyData.csv", header=0)
complete_data['month'] = pd.to_datetime(complete_data['Created-on']).dt.month
complete_data['year'] = pd.to_datetime(complete_data['Created-on']).dt.year

from sklearn.model_selection import train_test_split
feature_columns_to_use = ['Component', 'Product', 'Severity', 'Assigner', 'Submitter', 'Status', 'month', 'year']
nonnumeric_columns = ['Component', 'Product', 'Assigner', 'Submitter', 'Status']
X = complete_data[feature_columns_to_use]
y = complete_data['IFD_CFD_INDIC']

train_df, test_df, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

from sklearn.base import TransformerMixin
class DataFrameImputer(TransformerMixin):
    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].median() for c in X],
            index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.fill)

big_X = train_df.append(test_df)
big_X_imputed = DataFrameImputer().fit_transform(big_X)

le = LabelEncoder()
for feature in nonnumeric_columns:
    big_X_imputed[feature] = le.fit_transform(big_X_imputed[feature])

# Prepare the inputs for the model
train_X = big_X_imputed[0:train_df.shape[0]].as_matrix()
test_X = big_X_imputed[train_df.shape[0]::].as_matrix()

from imblearn.over_sampling import SMOTE, ADASYN
X_resampled, y_resampled = SMOTE(kind='borderline1').fit_sample(train_X, y_train)

gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(X_resampled, y_resampled)
predictions = gbm.predict(test_X)
#test_df["Prediction"]= predictions #test_df.to_csv("Predictions_xgboost.csv",encoding='utf-8')
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, predictions)
