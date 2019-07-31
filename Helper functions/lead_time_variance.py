import pandas as pd
import pymongo
import sys
sys.path.insert(0, "/data/ingestion/")
from Utils import *

#Setting up the config.ini file parameters
settings = configparser.ConfigParser()
settings.read('/data/ingestion/config.ini')

def main():
    #db = get_collection_details()
    options = parse_options()
    
    if(options.env == "Prod"):
        key = "csap_prod_database"
    
    else:
        key = "csap_stage_database"

    db = get_db(settings, key)
	lead_time_variance(db)

def lead_time_variance(db):

collection = db["potCFD_Check"]
cursor = collection.find({"$where":"this.IFD_CFD_INDIC_DATE > this.last_run_date"})
cursor = db.potCFD_Check.find({"$where":"this.IFD_CFD_INDIC_DATE > this.last_run_date"})
true_cfds =  pd.DataFrame(list(cursor))

true_cfds = pd.read_csv('potCFD_Check.csv')
a = true_cfds[['View','Security', 'Conversion Date']]
a = a.dropna()
days_predicted = []
days_actual = []

for index, row in a.iterrows():
	print(index)
	bugid = row['Security']
	converted_date = row['Conversion Date']
	if(row['View'] == 'Security'):
		coll = 'Security_1_results'
		days_predicted.append(0)
		days_actual.append(0)
	else:
		coll = row['View']
		cursor = db[coll].find({'IDENTIFIER': bugid})
		bug =  pd.DataFrame(list(cursor))
		print(bug)
		bug = bug.iloc[0]
		days_predicted.append(bug['Days_Predictions'])
		days_actual.append(int((pd.to_datetime(converted_date) - bug['SUBMITTED_DATE']).days))

a['Predicted days'] = days_predicted
a['Actual days'] = days_actual
a.to_csv('potCFD_Check_days.csv', encoding = 'utf-8')
