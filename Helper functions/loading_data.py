import pymongo
import pandas as pd
username = "csaprw"
passwd = "csaprw123"
hostname = "sjc-wwsl-fas4"
port = "27017"
db = "csap_stg"

mongo_connection_string="mongodb://"+username+":"+passwd+"@"+hostname+":"+port+"/"+db
client=pymongo.MongoClient(mongo_connection_string)
db=client.get_database(db)
collection = db["Pot_CFD_cqi"]
cursor = collection.find({}) # query

df =  pd.DataFrame(list(cursor))
print(df)
#new_df = df[["Headline", "Engineer", "Component"]]
#new_df = new_df.reset_index()