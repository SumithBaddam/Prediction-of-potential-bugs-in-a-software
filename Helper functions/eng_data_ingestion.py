#!/auto/vgapps-cstg02-vapps/csap/python/bin/python3
# -*- coding: utf-8 -*-
# Set up time: 2:00 PM ISTimport sys

import pymongo
from pymongo import MongoClient
from pymongo import cursor
import os
import json
import time
from datetime import date, timedelta
from isoweek  import Week
from bson import json_util

import configparser
import argparse
import collections
import csv
import data_ingestion_utilities
from pprint import pprint


'''
    This script is used to fetch Engineer data from CQI Oracle database. This data is at Query ID level and will be
    populated in the CSAP Mongo database. Collection names will follow the rule <queryid>_CQI_IBD_ENG".
    
    Input command line parameters:
        --env  -> prod
        --weekenddate -> data as String in the format format mm-dd-yyyy
'''

logfile = open('ingestionlog.log', 'w')
logfile.write("################## Ingestion Started ##################\n")


'''
    Function to parse the command line arguments
'''
def parse_options():
    parser = argparse.ArgumentParser(description="Ingestion script for fetching engineer data using CQI REST service")

    parser.add_argument("--env",
                        default='stage',
                        help='Stage or Prod',
                        type=str,
                        metavar='en')

    parser.add_argument("--weekenddate",
                        default='',
                        help='Week end date in mm-dd-yyyy format',
                        type=str,
                        metavar='en')

    args = parser.parse_args()
    return args


def main():
    #parse the command line variables
    options = parse_options()

    #Get the env paramters for the specified env
    envParameters = data_ingestion_utilities.get_env_parameters(options.env)

    #Check if there is a specific week end that is passed if so execute only for that week end else process for
    #number of years specified in number_of_years_of_data parameter
    if data_ingestion_utilities.get_env_parameters(options.weekenddate) == '':
        weeks = []
        #Calculate the week end dates to be passed to the CQI service for last 3 years
        years = int(envParameters['number_of_years_of_data'])
        weeks = data_ingestion_utilities.get_weekend_dates_for_years(years)
        #print(weeks)
    else:
        weeks = []
        years = int(envParameters['number_of_years_of_data'])
        weeks = data_ingestion_utilities.get_weekend_dates_for_years(years)

    print(weeks)
    #get all the view ids configured in the cqi config table
    queryids  = data_ingestion_utilities.get_all_configured_query_ids(envParameters['env'])
    #print(queryids)

    #get the db connection so that we can use it to insert the data coming from CQI web service
    cqidb = data_ingestion_utilities.get_db_connection_obj(envParameters['env'], envParameters['cqi_orcl_db_sid'])

    # get the db connection so that we can use it to insert the data
    csapdb = data_ingestion_utilities.get_db_connection_obj(envParameters['env'], envParameters['db'])


    #For every view id configured
    for queryid in queryids:
        # Derive the destination collection name
        ibdDest = queryid['bu'].replace(" ", "").upper()[:16] + "_CQI_IBD_ENG";

        # get the collection
        mongocol = csapdb[ibdDest]

        #For every week
        for week in weeks:

            # We will take all CFD defects and insert the eng data
            cfdDispEngCur = cqidb.cursor()

            # Prepare the query
            cfdDispEngCur.prepare("SELECT 'IFD' AS cfdifd, b.query_id, TO_CHAR(b.weekend_date,'mm-dd-yyyy') AS week_end_date, COUNT(DISTINCT a.engineer) AS distinct_eng_count FROM fact_defect_base a, fact_defect_query_result b WHERE a.defect_id = b.defect_id AND a.week_end_date = b.weekend_date AND b.weekend_date = TO_DATE(:weekenddate,'mm-dd-yyyy') AND b.query_id =:queryid AND a.ifd_indic = 1 AND a.severity_code IN ( 1,2,3 ) AND b.disposed_indic = 1 GROUP BY 'IFD', b.query_id, b.weekend_date UNION ALL SELECT 'CFD' AS cfdifd, b.query_id, TO_CHAR(b.weekend_date,'mm-dd-yyyy') AS week_end_date, COUNT(DISTINCT a.engineer) AS distinct_eng_count FROM fact_defect_base a, fact_defect_query_result b WHERE a.defect_id = b.defect_id AND a.week_end_date = b.weekend_date AND b.weekend_date = TO_DATE(:weekenddate,'mm-dd-yyyy') AND b.query_id =:queryid AND a.cfd_indic = 1 AND a.severity_code IN ( 1,2,3 ) AND b.disposed_indic = 1 GROUP BY 'CFD', b.query_id, b.weekend_date")

            #bind the parameters
            cfdDispEngCur.execute(None, {'weekenddate': week, 'queryid': int(queryid['_id']), 'weekenddate': week, 'queryid': int(queryid['_id'])})

            #Define a array to store the results
            curRows = []
            # for every result
            for row in cfdDispEngCur:
                #Covert the tuple into json object
                curRows.append({'QUERY_ID': row[1], 'week_end_date': row[2], 'unique_eng_count': row[3], 'CFDIFD': row[0]})

            #If there are no rows log a message
            if len(curRows) == 0:
                logMsg = "\nNo Data available for date :" + week + ", Query ID : " + str(int(queryid['_id']))
                logfile.write(logMsg)
                print(logMsg)
            else:
                data_ingestion_utilities.import_json_into_mongo(mongocol, json.dumps(curRows))


    logfile.write("\n################## Ingestion Ended Successfully ##################\n")
    logfile.close()





if __name__ == "__main__":
    main()