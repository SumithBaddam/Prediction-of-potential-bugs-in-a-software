import configparser
import pymongo
from pymongo import MongoClient
from datetime import date, timedelta
from isoweek  import Week
import json
import requests
from requests.auth import HTTPBasicAuth
import cx_Oracle
import pprint
from bson import json_util

'''
    This is a generic utility package for ingestion of the Engineer data at Query id and Queryid/DE Manager level. 
'''

def get_config_object():
    # Read Configuration
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config

'''
    This function returns a dictionary of parameters from config.init based on the environment requested for.
'''
def get_env_parameters(env):
    # Get the config object after parsing config.ini
    config = get_config_object()
    parameterDic = {}
    if env.lower() == "prod":
        env = "Prod"
        db_type = "Prod"
        hostname = config.get('csap_prod_database', 'host')
        port = config.get('csap_prod_database', 'dbPort')
        username = config.get('csap_prod_database', 'user')
        passwd = config.get('csap_prod_database', 'passwd')
        db = config.get('csap_prod_database', 'db')
        type = config.get('csap_prod_database', 'type')
        mongo_connection_url = "mongodb://" + username + ":" + passwd + "@" + hostname + ":" + port  + "/?authSource=" +db
        config_source = config.get('cqi_engineering_data_ingestion', 'cqi_config_source_collection')
        endpoint = config.get('cqi_engineering_data_ingestion', 'cqi_engineer_data_endpoint')
        endpointusername = config.get('cqi_engineering_data_ingestion', 'cqi_generic_username')
        endpointpassword = config.get('cqi_engineering_data_ingestion', 'cqi_generic_password')
        destination = config.get('cqi_engineering_data_ingestion', 'cqi_engineer_data_collection')
        number_of_years_of_data = config.get('cqi_engineering_data_ingestion', 'number_of_years_of_data')
        cqi_orcl_db_sid = config.get('cqi_oracle_prod_database', 'db')
        cqi_orcl_db_username = config.get('cqi_oracle_prod_database', 'user')
        cqi_orcl_db_password = config.get('cqi_oracle_prod_database', 'passwd')
        cqi_orcl_db_host = config.get('cqi_oracle_prod_database', 'host')
        cqi_orcl_db_dbPort = config.get('cqi_oracle_prod_database', 'dbPort')
        cqi_orcl_db_dbType = config.get('cqi_oracle_prod_database', 'type')
        cqi_eng_query = config.get('cqi_engineering_data_ingestion', 'cqi_eng_query')
        parameterDic.setdefault('env', env)
        parameterDic.setdefault('hostname', hostname)
        parameterDic.setdefault('port', port)
        parameterDic.setdefault('username', username)
        parameterDic.setdefault('passwd', passwd)
        parameterDic.setdefault('db', db)
        parameterDic.setdefault(db, type)
        parameterDic.setdefault('mongo_connection_url', mongo_connection_url)
        parameterDic.setdefault('dbtype', type)
        parameterDic.setdefault('cqi_config_src', config_source)
        parameterDic.setdefault('cqi_endpoint', endpoint)
        parameterDic.setdefault('cqi_endpoint_username', endpointusername)
        parameterDic.setdefault('cqi_endpoint_password', endpointpassword)
        parameterDic.setdefault('eng_data_destination', destination)
        parameterDic.setdefault('number_of_years_of_data', number_of_years_of_data)
        parameterDic.setdefault('cqi_orcl_db_sid', cqi_orcl_db_sid)
        parameterDic.setdefault('cqi_orcl_db_username', cqi_orcl_db_username)
        parameterDic.setdefault('cqi_orcl_db_password', cqi_orcl_db_password)
        parameterDic.setdefault('cqi_orcl_db_host', cqi_orcl_db_host)
        parameterDic.setdefault('cqi_orcl_db_dbPort', cqi_orcl_db_dbPort)
        parameterDic.setdefault(cqi_orcl_db_sid, cqi_orcl_db_dbType)
        parameterDic.setdefault('cqi_eng_query', cqi_eng_query)
    else:
        #Currently we are doing in Prod only and hence not configuring stage properties here. If needed we will have to replace the
        #prod setting with that of stage and rest of the script will work fine.
        env = "Prod"
        db_type = "Prod"
        hostname = config.get('csap_prod_database', 'host')
        port = config.get('csap_prod_database', 'dbPort')
        username = config.get('csap_prod_database', 'user')
        passwd = config.get('csap_prod_database', 'passwd')
        db = config.get('csap_prod_database', 'db')
        type = config.get('csap_prod_database', 'type')
        mongo_connection_url = "mongodb://" + username + ":" + passwd + "@" + hostname + ":" + port + "/?authSource=" + db
        config_source = config.get('cqi_engineering_data_ingestion', 'cqi_config_source_collection')
        endpoint = config.get('cqi_engineering_data_ingestion', 'cqi_engineer_data_endpoint')
        endpointusername = config.get('cqi_engineering_data_ingestion', 'cqi_generic_username')
        endpointpassword = config.get('cqi_engineering_data_ingestion', 'cqi_generic_password')
        destination = config.get('cqi_engineering_data_ingestion', 'cqi_engineer_data_collection')
        number_of_years_of_data = config.get('cqi_engineering_data_ingestion', 'number_of_years_of_data')
        cqi_orcl_db_sid = config.get('cqi_oracle_prod_database', 'db')
        cqi_orcl_db_username = config.get('cqi_oracle_prod_database', 'user')
        cqi_orcl_db_password = config.get('cqi_oracle_prod_database', 'passwd')
        cqi_orcl_db_host = config.get('cqi_oracle_prod_database', 'host')
        cqi_orcl_db_dbPort = config.get('cqi_oracle_prod_database', 'dbPort')
        cqi_orcl_db_dbType = config.get('cqi_oracle_prod_database', 'type')
        cqi_eng_query = config.get('cqi_engineering_data_ingestion', 'cqi_eng_query')
        parameterDic.setdefault('env', env)
        parameterDic.setdefault('hostname', hostname)
        parameterDic.setdefault('port', port)
        parameterDic.setdefault('username', username)
        parameterDic.setdefault('passwd', passwd)
        parameterDic.setdefault('db', db)
        parameterDic.setdefault(db, type)
        parameterDic.setdefault('mongo_connection_url', mongo_connection_url)
        parameterDic.setdefault('dbtype', type)
        parameterDic.setdefault('cqi_config_src', config_source)
        parameterDic.setdefault('cqi_endpoint', endpoint)
        parameterDic.setdefault('cqi_endpoint_username', endpointusername)
        parameterDic.setdefault('cqi_endpoint_password', endpointpassword)
        parameterDic.setdefault('eng_data_destination', destination)
        parameterDic.setdefault('number_of_years_of_data', number_of_years_of_data)
        parameterDic.setdefault('cqi_orcl_db_sid', cqi_orcl_db_sid)
        parameterDic.setdefault('cqi_orcl_db_username', cqi_orcl_db_username)
        parameterDic.setdefault('cqi_orcl_db_password', cqi_orcl_db_password)
        parameterDic.setdefault('cqi_orcl_db_host', cqi_orcl_db_host)
        parameterDic.setdefault('cqi_orcl_db_dbPort', cqi_orcl_db_dbPort)
        parameterDic.setdefault(cqi_orcl_db_sid, cqi_orcl_db_dbType)
        parameterDic.setdefault('cqi_eng_query', cqi_eng_query)
    return parameterDic

'''
    This function returns a dictionary of parameters from config.init based on the environment requested for.
'''
def get_db_connection_obj(env, dbname):
    envParameters = get_env_parameters(env)
    #Check what kind of the DB object we need to buid
    if envParameters[dbname] == 'mongo':
        #build and return mongo object
        client = MongoClient(envParameters['mongo_connection_url'])
        return client[dbname]
    if envParameters[dbname] == 'oracle':
        #build and return Oracle client
        dsn_tns = cx_Oracle.makedsn(envParameters['cqi_orcl_db_host'], envParameters['cqi_orcl_db_dbPort'], envParameters['cqi_orcl_db_sid'])
        db = cx_Oracle.connect(user=envParameters['cqi_orcl_db_username'], password=envParameters['cqi_orcl_db_password'], dsn=dsn_tns)
        return db
'''
    Returns the week end date(Saturday) for specified number of years starting current year(calcualted based on todays date
'''
def get_weekend_dates_for_years(number_of_years):
    #Calculate the current week number for current year
    weeks = Week.thisweek().week
    #For previous years
    for i in range(1, number_of_years):
        #calculate the year
        year = date.today().year - i
        #Check how many weeks are there in the year calculated above and add it to weeks
        weeks = weeks + Week.last_week_of_year(year).week
    # define an empty list to store the week end dates
    week_array = []
    #Now we will have to loops "weeks" number of times and get the week end date
    dt = date.today()
    for i in range(1, weeks):
        strd = dt - timedelta(days=dt.isoweekday() + 1 + (7 * i))
        week_array.append(strd.strftime('%m-%d-%Y'))
    return week_array


'''
    Make a post request
'''
def make_restful_post_request(env, url, payload):
    envParameters = get_env_parameters(env)
    #get the end point details
    endpoint = envParameters[url]
    username = envParameters['cqi_endpoint_username']
    password = envParameters['cqi_endpoint_password']
    #Build the authentication header
    auth = HTTPBasicAuth(username, password)
    #Build the header
    headers = {'Content-type': 'application/json',
            'Accept': 'application/json'}
    #convert the object containing data into json
    postdata = json.dumps(payload)
    #print(postdata)
    #Make the post call
    ret = requests.post(url=endpoint, headers=headers, data=postdata, auth=auth)
    return ret


'''
    Get all query id's
'''
def get_all_configured_query_ids(env):
    # Get the env paramters for the specified env
    envParameters = get_env_parameters(env)
    #Array to store the query ids
    queryid_list = []
    #Get the db connection so that we can connect to the mongo DB
    db = get_db_connection_obj(envParameters['env'], envParameters['db'])
    #Get the collection from where we will query the ids. This is configured in the config file.
    collection = db[envParameters['cqi_config_src']]
    #Define a pipeline to group by the query ids and fetch the BU and query id list
    pipeline = [{'$match': {'bu_enabled': {'$eq': "Y"}, 'view_enabled': {'$eq': "Y"}, 'query_enabled': {'$eq': "Y"}}},{ '$group': {'_id': "$query_id", 'bu': { '$first': "$bu_name"}}}]
    data = collection.aggregate(pipeline)
    for rec in data:
        queryid_list.append(rec)
    return queryid_list


'''
    import json into mongo collection
'''
def import_json_into_mongo(dbConnectionObj, json):
    #What ever json is passed will be inserted using the connection object passed.
    dbConnectionObj.insert_many(json_util.loads(json))