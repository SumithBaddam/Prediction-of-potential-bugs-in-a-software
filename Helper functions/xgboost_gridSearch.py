# For parameter tuning use this URL...will be very useful
# http://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/

import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split

from sklearn.cross_validation import *
from sklearn.grid_search import GridSearchCV

# Load the data
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
        
feature_columns_to_use = ['Component', 'Product', 'Severity', 'Assigner', 'Submitter', 'Status', 'month', 'year']
nonnumeric_columns = ['Component', 'Product', 'Assigner', 'Submitter', 'Status']

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

xgb_model = xgb.XGBClassifier()

#brute force scan for all parameters, here are the tricks
#usually max_depth is 6,7,8
#learning rate is around 0.05, but small changes may make big diff
#tuning min_child_weight subsample colsample_bytree can have 
#much fun of fighting against overfit 
#n_estimators is how many round of boosting
#finally, ensemble xgboost with multiple seeds may reduce variance
parameters = {'nthread':[2], #when use hyperthread, xgboost may become slower
              'objective':['binary:logistic'],
              'learning_rate': [0.05], #so called `eta` value
              'max_depth': [7],
              'min_child_weight': [1],
              'silent': [1],
              'subsample': [0.8],
              'colsample_bytree': [0.7,0.6],
              'n_estimators': [300], #number of trees, change it to 1000 for better results
              'seed': [1337]}


clf = GridSearchCV(xgb_model, parameters, n_jobs=10, 
                   cv=StratifiedKFold(train_y, n_folds=5, shuffle=True), 
                   scoring='roc_auc',
                   verbose=2, refit=True)

clf.fit(train_X, train_y)


best_parameters, score, _ = max(clf.grid_scores_, key=lambda x: x[1])
print('Raw AUC score:', score)
for param_name in sorted(best_parameters.keys()):
    print("%s: %r" % (param_name, best_parameters[param_name]))

test_probs = clf.predict_proba(test_X)[:,1]
predictions = (test_probs > 0.45).astype('int')


#gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(train_X, train_y)
#predictions = gbm.predict(test_X)
from sklearn.metrics import confusion_matrix
confusion_matrix(test_df['IFD_CFD_INDIC'], predictions)

