import configparser
import pymongo
from pymongo import MongoClient
from datetime import date, timedelta
from isoweek  import Week
import json
import requests
from requests.auth import HTTPBasicAuth
#import cx_Oracle
import pprint
from bson import json_util
import datetime
from dateutil import relativedelta
import pandas as pd
from pandas import DataFrame
#from pyhive import hive
import os
from pymongo import cursor
import subprocess
#from django.utils.encoding import smart_str
from pymongo.errors import BulkWriteError
import re

def get_iqs_db(config):
    ip = config.get('IQS','faDbHost')
    port = config.get('IQS', 'faDbPort')
    SID = config.get('IQS', 'faDBSName')
    user = config.get('IQS', 'faDBUser')
    passwd = config.get('IQS', 'faDBPswd')

    dsn_tns = cx_Oracle.makedsn(ip, port, service_name=SID)
    conn = cx_Oracle.connect(user, passwd, dsn_tns)

    return conn

def get_FADetails_for_SRs(config, srList, pfStr, opDir):

    sr_list = formINClause(srList, 'str',300)
    conn = get_iqs_db(config)

    sqlStr1 = ( "SELECT  " +
                "" +
                "    FA.SERVICE_REQUEST_NUMBER SR_NUMBER," +
                "    FA.PRODUCT_FAMILY PF," +
                "    FA.FA_CASE_NUMBER FA_CASE_NUMBER," +
                "    FA.ROOT_CAUSE_CODE FA_ROOT_CAUSE_CODE," +
                "    FA.SYMPTOM_CODE SYMPTOM_CODE," +
                "    FA.CASE_REVIEW_SUMMARY CASE_REVIEW_SUMMARY," +
                "    FA.FAULT_DUPLICATION_RESULTS FAULT_DUPLICATION_RESULTS," +
                "    FA.IS_FAULT_DUPLICATED IS_FAULT_DUPLICATED," +
                "    FA.FAULT_ISOLATION_THEORY  FAULT_ISOLATION_THEORY," +
                "    FA.FAULT_ISOLATION_RESULTS FAULT_ISOLATION_RESULTS," +
                "    FA.FAILURE_MODE_CONCAT FAILURE_MODE_CONCAT," +
                "    FA.CASE_STATUS CASE_STATUS," +
                "    FA.CLOSURE_CODE_DESC   CLOSURE_CODE_DESC," +
                "    FA.SERVICE_REQUEST_TITLE_DESC  SERVICE_REQUEST_TITLE_DESC," +
                "    FA.PRELIMNRY_FAILURE_ANLYS PRELIMNRY_FAILURE_ANLYS," +
                "    FA.COMPLETED_FAILURE_ANLYS COMPLETED_FAILURE_ANLYS," +
                "    FA.VISUAL_INSP_CODE_DESC VISUAL_INSP_CODE_DESC," +
                "    FA.CUSTOMER_NAME CUSTOMER_NAME," +
                "    FA.DATE_ORIGINATED   DATE_ORIGINATED," +
                "    FA_AFCTD.ITEM_NUMBER ITEM_NUMBER," +
                "    FA_AFCTD.ITEM_CATEGORY  ITEM_CATEGORY  ," +
                "    FA_AFCTD.ITEM_DESCRIPTION  ITEM_DESCRIPTION  ," +
                "    FA_AFCTD.ITEM_TYPE  ITEM_TYPE  ," +
                "    FA_AFCTD.REF_DESIGNATOR_LOCATION REF_DESIGNATOR_LOCATION," +
                "    FA_AFCTD.SITE_RECEIVED_SN   SITE_RECEIVED_SN," +
                "    FA_AFCTD.FAILURE_MODE FAILURE_MODE," +
                "    FA_AFCTD.DESCRIPTION DESCRIPTION," +
                "    FA_AFCTD.FAILURE_CODE FAILURE_CODE," +
                "    FA_AFCTD.PID_RESULTS PID_RESULTS," +
                "    FA_AFCTD.ISSUE_CLASSIFICATIONS   ISSUE_CLASSIFICATIONS," +
                "    FA_AFCTD.COMPONENT_LOT_CODE COMPONENT_LOT_CODE," +
                "    FA_AFCTD.COMPONENT_DATE_CODE COMPONENT_DATE_CODE," +
                "    FA_AFCTD.SITE_RECEIVED_SW_REVISION SITE_RECEIVED_SW_REVISION," +
                "    FA_AFCTD.CDETS#  FA_CDETS," +
                "    RC.RC_CASE_NUMBER RC_CASE_NUMBER," +
                "    RC.RC_CLOSURE_CODE  RC_CLOSURE_CODE," +
                "    RC.RC_CLOSURE_MODE  RC_CLOSURE_MODE ," +
                "    RC.CASE_STATUS RC_CASE_STATUS," +
                "    RC.PRELIM_RC_DETAIL RC_PRELIM_RC_DETAIL," +
                "    RC.FINAL_RC_DETAIL   RC_FINAL_RC_DETAIL," +
                "    RC.AFCTD_SN   RC_AFCTD_SN ," +
                "    RC.PROBLEM_DESC  RC_PROBLEM_DESC ," +
                "    RC_AFCTD.ITEM_DESC    RC_AFCTD_ITEM_DESC ," +
                "    RC.COMMODITY   COMMODITY," +
                "    RC_AFCTD.MANUFACTURER  RC_AFCTD_MANUFACTURER," +
                "    RC_AFCTD.MANUFACTURER_PART_NUM  RC_AFCTD_MPN," +
                "    RC_AFCTD.COMPONENT_SN_NUMBER  COMPONENT_SN_NUMBER," +
                "    RC_AFCTD.COMPONENT_DATE_CODE  RC_AFCTD_COMPONENT_DATE_CODE," +
                "    RC_AFCTD.COMPONENT_LOT_CODE RC_AFCTD_COMPONENT_LOT_CODE," +
                "    RC_AFCTD.PARENT_PART_NUMBER   RC_AFCTD_PARENT_PART_NUMBER  ," +
                "    RC_AFCTD.PARENT_SN_NUMBER  RC_AFCTD_PARENT_SN_NUMBER ," +
                "    RC_AFCTD.COMMODITY_CODE_LEVEL1 COMMODITY_CODE_LEVEL1 ," +
                "    RC_AFCTD.COMMODITY_CODE_LEVEL2 COMMODITY_CODE_LEVEL2, " +
                "    RC_AFCTD.COMMODITY_CODE_LEVEL3 COMMODITY_CODE_LEVEL3 " +
                "    RC_AFCTD.CDETS RC_CDETS," +
                "    RC_AFCTD.SITE_RECEIVED_SW_REV RCSITE_RECEIVED_SW_REV" +
                "    " +
                "FROM" +
                "    PQRPT_QMS_FA_CASE_D FA," +
                "    PQRPT_QMS_FA_AFCTD_ITEM_D FA_AFCTD," +
                "    PQRPTN_QMS_RC_CASE RC," +
                "    PQRPTN_QMS_RC_AFCTD_ITEMS RC_AFCTD" +
                " " +
                "WHERE"
                )
    sqlStr2  = ("    AND FA.CLOSURE_CODE = 'Fault Isolated'" +
                "    AND FA.FA_CASE_NUMBER = FA_AFCTD.CASE_NUMBER" +
                "    AND RC.PARENT_FA_CASE_NUMBER  (+)= FA.FA_CASE_NUMBER " +
                "    AND RC_AFCTD.RC_CASE_NUMBER (+)= RC.RC_CASE_NUMBER" +
                "    AND RC_AFCTD.ITEM_NUMBER (+)= FA_AFCTD.ITEM_NUMBER  "
                )

    wholeData = pd.DataFrame()

    for srs in sr_list:
        sqlStr = sqlStr1 + " FA.SERVICE_REQUEST_NUMBER IN ( " + srs + ") " + sqlStr2
        faData = pd.read_sql(sqlStr, conn)

        if len(faData) > 0 :
            wholeData = wholeData.append(faData)

    if len(wholeData) > 0:
        wholeData.to_csv( opDir + config.get("IQS","FARCFilePrefix")  + pfStr + ".csv")

    return wholeData


def set_env():
    path_list = ["/auto/vgapps-cstg02-vapps/python/bin","/auto/vgapps-cstg02-vapps/python/lib/python3.6/site-packages/"]
    py_path_list = ["/auto/vgapps-cstg02-vapps/python","/auto/vgapps-cstg02-vapps/python/lib/python3.6/site-packages/"]
    ora_path_list  = ["/auto/vgapps-cstg02-vapps/analytics/executables/lib64","/auto/vgapps-cstg02-vapps/analytics/executables/libc6_2.17","/usr/cisco/packages/oracle/current/lib","/usr/lib64","/usr/local/lib"]

    os.environ["PYTHONPATH"] = os.pathsep.join(py_path_list)
    os.environ["PATH"] =  os.pathsep.join(path_list) + os.pathsep  + os.environ["PATH"]
    os.environ["ORACLE_HOME"] = "/usr/cisco/packages/oracle/current/"
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(ora_path_list)

def bulkMongoInsert(collName, df):

    bulkInsert = collName.initialize_unordered_bulk_op()
    try:
        records = json.loads(df.T.to_json(date_format='iso')).values()
#         dbInst[collName].insert_many(records, ordered=False, bypass_document_validation=True)
        collName.insert_many(records, ordered=False, bypass_document_validation=True)
#         collName.insert_many(records, ordered=False)
    except BulkWriteError as bwe:
        print(bwe.details)
    except UnicodeEncodeError as bwe:
        print(bwe.details)

def get_db_details(config, key):

    dbDetails = {}

    dbDetails['host'] = config.get(key, 'host')
    dbDetails['port'] = config.get(key, 'dbPort')
    dbDetails['db'] = config.get(key, 'db')
    dbDetails['user'] = config.get(key, 'user')
    dbDetails['pwd'] = config.get(key, 'passwd')

    return dbDetails

def get_db( config, key):

    dbConnDetails = get_db_details(config, key)

    db_connection = "mongodb://" + dbConnDetails['user'] + ":" + dbConnDetails['pwd'] + "@" + dbConnDetails['host'] + ":" + dbConnDetails['port'] + "/?authSource=" + dbConnDetails['db']
    print(db_connection)
    dbCl = MongoClient(db_connection)
    output_db = dbCl[dbConnDetails['db']]

    return output_db

def get_cqi_db( config, key):

#     dbConnDetails = get_db_details(config, key)

    db_connection = "mongodb://sjc-wwpl-cqi5.cisco.com:27019/"

    print(db_connection)
    dbCl = MongoClient(db_connection)
    output_db = dbCl["cqidev"]

    return output_db


def get_last_saturday( format="%y%m%d"):

    today = datetime.datetime.now()
    start = today - datetime.timedelta((today.weekday() + 1) % 7)
    sat = start + relativedelta.relativedelta(weekday=relativedelta.SA(-1))
    return sat.strftime(format)

def get_last_sunday(format="%y%m%d"):

    today = datetime.datetime.now()
    start = today - datetime.timedelta((today.weekday() + 1) % 7)
    sun = sat + relativedelta.relativedelta(weekday=relativedelta.SU(-1))
    return sun.strftime(format)



def get_FA_RC_details(config, pf, stDate):
    ip = config.get('fa_oracle_database','host')
    port = config.get('fa_oracle_database', 'dbPort')
    SID = config.get('fa_oracle_database', 'db')
    user = config.get('fa_oracle_database', 'user')
    passwd = config.get('fa_oracle_database', 'passwd')

    dsn_tns = cx_Oracle.makedsn(ip, port, SERVICE_NAME=SID)
    conn = cx_Oracle.connect(user, passwd, dsn_tns)
    faData = pd.read_sql(sql, conn, params = {'pf': pf, 'stDate': stDate})
    c = conn.cursor()

    sql  = "SELECT RMA.HEADER_ID HEADER_ID, RMA.ORDER_NUMBER RMA_ORDER_NUM, RMA.ORIG_SYS_DOCUMENT_REF SR_NUMBER, RPIDS.ITEM_NAME PID, FA.PRODUCT_FAMILY PF, FA.FA_CASE_NUMBER FA_CASE_NUMBER, FA.ROOT_CAUSE_CODE FA_ROOT_CAUSE_CODE, FA.SYMPTOM_CODE SYMPTOM_CODE, FA.CASE_REVIEW_SUMMARY CASE_REVIEW_SUMMARY, FA.FAULT_DUPLICATION_RESULTS FAULT_DUPLICATION_RESULTS, FA.IS_FAULT_DUPLICATED IS_FAULT_DUPLICATED, FA.FAULT_ISOLATION_THEORY  FAULT_ISOLATION_THEORY, FA.FAULT_ISOLATION_RESULTS FAULT_ISOLATION_RESULTS, FA.FAILURE_MODE_CONCAT FAILURE_MODE_CONCAT , FA.CASE_STATUS CASE_STATUS, FA.CLOSURE_CODE_DESC   CLOSURE_CODE_DESC  , FA.SERVICE_REQUEST_TITLE_DESC  SERVICE_REQUEST_TITLE_DESC  , FA.PRELIMNRY_FAILURE_ANLYS PRELIMNRY_FAILURE_ANLYS, FA.COMPLETED_FAILURE_ANLYS COMPLETED_FAILURE_ANLYS, FA.VISUAL_INSP_CODE_DESC VISUAL_INSP_CODE_DESC, FA.CUSTOMER_NAME CUSTOMER_NAME, FA.DATE_ORIGINATED   DATE_ORIGINATED , FA_AFCTD.REF_DESIGNATOR_LOCATION REF_DESIGNATOR_LOCATION, FA_AFCTD.COMMODITY_CODE_LEVEL2  COMMODITY_CODE_LEVEL2  , FA_AFCTD.SITE_RECEIVED_SN   SITE_RECEIVED_SN  , FA_AFCTD.FAILURE_MODE FAILURE_MODE, FA_AFCTD.DESCRIPTION DESCRIPTION, FA_AFCTD.FAILURE_CODE FAILURE_CODE, FA_AFCTD.PID_RESULTS PID_RESULTS, FA_AFCTD.ISSUE_CLASSIFICATIONS   ISSUE_CLASSIFICATIONS  , RC.RC_CASE_NUMBER RC_CASE_NUMBER, RC.RC_CLOSURE_CODE  RC_CLOSURE_CODE  , RC.RC_CLOSURE_MODE  RC_CLOSURE_MODE , RC.CASE_STATUS RC_CASE_STATUS, RC.PRELIM_RC_DETAIL RC_PRELIM_RC_DETAIL, RC.FINAL_RC_DETAIL   RC_FINAL_RC_DETAIL  , RC.AFCTD_SN   RC_AFCTD_SN  , RC.PROBLEM_DESC  RC_PROBLEM_DESC  , RC_AFCTD.ITEM_DESC    RC_AFCTD_ITEM_DESC   , RC_AFCTD.MANUFACTURER  RC_AFCTD_MANUFACTURER, RC_AFCTD.MANUFACTURER_PART_NUM  RC_AFCTD_MPN, RC_AFCTD.COMPONENT_SN_NUMBER  RC_AFCTD_CPN , RC_AFCTD.COMPONENT_DATE_CODE  RC_AFCTD_COMPONENT_DATE_CODE, RC_AFCTD.COMPONENT_LOT_CODE RC_AFCTD_COMPONENT_LOT_CODE, RC_AFCTD.PARENT_PART_NUMBER   RC_AFCTD_PARENT_PART_NUMBER  , RC_AFCTD.PARENT_SN_NUMBER  RC_AFCTD_PARENT_SN_NUMBER FROM PQRPT_TSS_ORDER_HEADERS_F RMA, pqrpt_tss_order_lines_f RPIDS, PQRPT_QMS_FA_CASE_D FA, PQRPT_QMS_FA_AFCTD_ITEM_D FA_AFCTD, PQRPTN_QMS_RC_CASE RC, PQRPTN_QMS_RC_AFCTD_ITEMS RC_AFCTD WHERE RMA.HEADER_ID = RPIDS.HEADER_ID AND RPIDS.LINE_CATEGORY = 'RETURN' AND RMA.ORDER_NUMBER = FA.SERVICE_ORDER_NUMBER AND FA.PRODUCT_FAMILY = :pf AND FA.DATE_ORIGINATED   >  :stDate AND FA.CLOSURE_CODE = 'Fault Isolated' AND FA.FA_CASE_NUMBER = FA_AFCTD.CASE_NUMBER AND FA.FA_CASE_NUMBER = RC.PARENT_FA_CASE_NUMBER AND RC.RC_CASE_NUMBER = RC_AFCTD.RC_CASE_NUMBER "
    c.execute(sql)
    row = c.fetchone()

    return row

# Get Bug Headline, Description from Data Lake CDETS
def getBugDesc(config, qStr):
    conn = hive.Connection(host=config.get("cdets_audit_info","host"), port=config.get("cdets_audit_info","port"), username=config.get("cdets_audit_info","user"),password=config.get("cdets_audit_info","passwd"),auth='LDAP',configuration={'hive.auto.convert.join':'false','mapred.mappers.tasks':'25','mapred.job.shuffle.input.buffer.percent':'0.50','mapreduce.map.memory.mb':'12000','mapreduce.reduce.memory.mb':'12000','mapred.reduce.child.java.opts':'-Xmx12000m','mapred.map.child.java.opts':'-Xmx12000m','hive.exec.reducers.bytes.per.reducer':'104857600','hive.optimize.skewjoin':'true'})
    sqlStr = "select cdets_etl_bug.identifier Identifier,cdets_etl_bug.headline headline,cdets_etl_bug.description description from quality_cdtetlprd_siebel.cdets_etl_bug where identifier in (" + qStr + ")  "
    df = pd.read_sql( sqlStr, conn)
    df = df.replace(r'\\n',' ', regex=True)
    df = df.replace(r',',' ', regex=True)
    df.columns = ["IDENTIFIER","Headline","ENCL-Description"]
    df["Headline"] = df["Headline"].str.encode('ascii','ignore')
    df["ENCL-Description"] = df["ENCL-Description"].str.encode('ascii','ignore')
    return df

# Get Bug Audit from Data Lake CDETS
def getBugAudit( config, qStr, stDate=None, enDate=None ):

    conn = hive.Connection(host=config.get("cdets_audit_info","host"), port=config.get("cdets_audit_info","port"), username=config.get("cdets_audit_info","user"),password=config.get("cdets_audit_info","passwd"),auth='LDAP',configuration={'hive.auto.convert.join':'false','mapred.mappers.tasks':'25','mapred.job.shuffle.input.buffer.percent':'0.50','mapreduce.map.memory.mb':'12000','mapreduce.reduce.memory.mb':'12000','mapred.reduce.child.java.opts':'-Xmx12000m','mapred.map.child.java.opts':'-Xmx12000m','hive.exec.reducers.bytes.per.reducer':'104857600','hive.optimize.skewjoin':'true'})
    sqlStr = "select cdets_etl_audit.identifier Identifier,to_date(cdets_etl_audit.last_mod_on) IFD_CFD_INDIC_DATE,cdets_etl_audit.old_val OldValue,cdets_etl_audit.new_val NewValue,cdets_etl_audit.field_name FieldName from quality_cdtetlprd_siebel.CDETS_ETL_AUDIT where identifier in (" + qStr + ") and field_name in ('Found') and new_val='customer-use' "
    if stDate:
        sqlStr = sqlStr + " and cdets_etl_audit.last_mod_on between '" + stDate +"' and '"+ enDate +"' "
    df = pd.read_sql( sqlStr, conn)
    df.columns = ["IDENTIFIER","IFD_CFD_INDIC_DATE","OldValue","NewValue","FieldName"]
    return df

# Get Bug list from QDDTS for a given query definition
def run_QDDTS_query( query, opFile):

    qCmd = "/usr/cisco/bin/query.pl \" " + query  + "\" > "  + opFile
    subprocess.check_call(qCmd, shell=True, stderr=subprocess.STDOUT)

    cmd = "sed -i '1iIdentifier' " + opFile
    subprocess.check_call(cmd, shell=True, stderr=subprocess.STDOUT)

    bugList = pd.read_csv(opFile,low_memory=False)
    bugList.columns = ["IDENTIFIER"]

    return bugList

def run_QDDTS_qbugval( queryOPfile, opFile, bugFields):

    tempFile = opFile + ".bk"
    qCmd = "cat "+ queryOPfile + " | /usr/cisco/bin/qbugval.pl -sep \"##CSAP_INGEST##\" " + bugFields + "> " + tempFile
    subprocess.check_call(qCmd, shell=True, stderr=subprocess.STDOUT)

    cmd = "sed -i 's/,/ /g' " + tempFile
    print(cmd)
    subprocess.check_call(cmd, shell=True, stderr=subprocess.STDOUT)

    cmd = "sed -i \"s/'//g\" " + tempFile
    print(cmd)
    subprocess.check_call(cmd, shell=True, stderr=subprocess.STDOUT)

    cmd = "sed -i 's/\"//g' " + tempFile
    print(cmd)
    subprocess.check_call(cmd, shell=True, stderr=subprocess.STDOUT)


    cmd = "sed -i 's/##CSAP_INGEST##/,/g' " + tempFile
    print(cmd)
    subprocess.check_call(cmd, shell=True, stderr=subprocess.STDOUT)


    cmd = "/eifdata/web/common/bin/perl -p0e 's/\\n(?!(CSC|$))/ /g' " + tempFile +  "> " + opFile
    print(cmd)
    subprocess.check_call(cmd, shell=True, stderr=subprocess.STDOUT)


    bugFields = bugFields.replace(" ",",")
    cmd = "sed -i '1i" + bugFields + "' " + opFile
    print(cmd)
    subprocess.check_call(cmd, shell=True, stderr=subprocess.STDOUT)
    bugData = pd.read_csv(opFile,low_memory=False)
#     bugData = bugData.dropna(axis=0,how='any')
#     bugData.to_csv(opFile, index=False)

    return bugData

# Run CQI query
def run_CQI_Query(config, sql, ipParams):

    ip = config.get('cqi_oracle_prod_database', 'host')
    port = config.get('cqi_oracle_prod_database', 'dbPort')
    SID = config.get('cqi_oracle_prod_database', 'db')
    user = config.get('cqi_oracle_prod_database', 'user')
    passwd = config.get('cqi_oracle_prod_database', 'passwd')
    op_path = config.get('script_results_loc', 'results_path')

    dsn_tns = cx_Oracle.makedsn(ip, port, SID)
    conn = cx_Oracle.connect(user, passwd, dsn_tns)
    cqiData = pd.read_sql(sql, conn, params = ipParams)

    return cqiData


# Get latest WEEK END DATE from CQI
def get_cqi_recent_run(config):
    ip = config.get('cqi_oracle_prod_database', 'host')
    port = config.get('cqi_oracle_prod_database', 'dbPort')
    SID = config.get('cqi_oracle_prod_database', 'db')
    user = config.get('cqi_oracle_prod_database', 'user')
    passwd = config.get('cqi_oracle_prod_database', 'passwd')
    op_path = config.get('script_results_loc', 'results_path')

    dsn_tns = cx_Oracle.makedsn(ip, port, SID)
    conn = cx_Oracle.connect(user, passwd, dsn_tns)
    c = conn.cursor()

    sql  = "SELECT MAX(WEEK_END_DATE) FROM FACT_DEFECT_BASE"
    c.execute(sql)
    row = c.fetchone()
    return row[0].strftime("%y%m%d")

def formINClause(bugs, type=None, ln=1000):
    inStrings = []
    bugLen =  len(bugs)

    if bugLen > ln :
        start_indx = 0
        for indx in range(bugLen):
            end_indx = start_indx + ln
            if end_indx > bugLen:
                end_indx = bugLen
                if start_indx == end_indx:
                    break
            temp_potCFD_list = bugs[start_indx:end_indx]
            if type and type == "int":
                tmpStr = ",".join(temp_potCFD_list)
                oneStr =  tmpStr
            else:
                tmpStr = "','".join(temp_potCFD_list)
                oneStr = "'" + tmpStr + "'"
            inStrings.append(oneStr)
            start_indx = end_indx
    else:
        if type and type == "int":
            tmpStr = ",".join(bugs)
            oneStr =  tmpStr
        else:
            tmpStr = "','".join(bugs)
            oneStr = "'" + tmpStr + "'"
        inStrings.append(oneStr)
    return inStrings


def get_enabled_queries( config, dbInst, usecase, viewID=None, queryID=None, bu=None) :

    queries = list()

    # Get CQI Queires
    cqiConfig = config.get('cqi_config','cqi_data')

    queryList = []
    queryList.append({"view_enabled":"Y"})
    queryList.append({"bu_enabled":"Y"})
    queryList.append({"query_enabled":"Y"})
    queryList.append({usecase:"Y"})

    if viewID:
        queryList.append({"view_id":int(viewID)})

    if queryID:
        queryList.append({"query_id":int(queryID)})

    if bu:
        queryList.append({"bu_name":bu})
    print(queryList)
    print("query list 1 printed")


    resColl = dbInst[cqiConfig]
    res = resColl.find({"$and":queryList},{"view_id":1,"bu_name":1,"query_id":1})

    for i in res:
#         print(i)
        realmQ = config.get('CQI_Connect_Info','query_cname')
        resColl2 = dbInst[realmQ]

        res2 = resColl2.find_one({"$and": [ {"query_id":str(int(i["query_id"]))},{"view_id":str(int(i["view_id"]))} ]}  ,{"_id":0,"query_id":1,"view_name":1,"query_name":1,"query_definition":1,"ifdcfd_cutoff":1})
#         print(res2)
        if res2:
          tmp = {}
          tmp["view_id"] = i["view_id"]
          tmp["view_name"] = res2["view_name"]
          tmp["bu_name"] = i["bu_name"]
          tmp["query_id"] = i["query_id"]
          tmp["query_name"] = res2["query_name"]
          tmp["query_definition"] = res2["query_definition"]
          if "ifdcfd_cutoff" in res2.keys():
           tmp["ifdcfd_cutoff"] = res2["ifdcfd_cutoff"]
          else:
           tmp["ifdcfd_cutoff"] = None
          queries.append(tmp)

    # Get User Defined Queires
    userQueries = config.get('CQI_Connect_Info','user_query_cname')
    queryList = []
    if queryID:
        queryList.append({"query_id":int(queryID)})

    if bu:
        queryList.append({"bu":bu})
    print(queryList)
    print("querylist 2 printed")
    cursor = None
    if len(queryList) > 0:
        cursor = dbInst[userQueries].find({"$and":queryList},{"_id":0})
    else:
        cursor = dbInst[userQueries].find({},{"_id":0})

    for rec in cursor:
        tmp = {}
        tmp["view_id"] = rec["view_id"]
        tmp["view_name"] = rec["view_name"]
        tmp["bu_name"] = rec["bu"]
        tmp["query_id"] = rec["query_id"]
        tmp["query_name"] = rec["query_name"]
        tmp["query_definition"] = rec["query_definition"]
        if "ifdcfd_cutoff" in rec.keys():
         tmp["ifdcfd_cutoff"] = rec["ifdcfd_cutoff"]
        else:
         tmp["ifdcfd_cutoff"] = None
        queries.append(tmp)

    return queries

def getAuditDataComp( config, qStr ):
    conn = hive.Connection(host=config.get("cdets_audit_info","host"), port=config.get("cdets_audit_info","port"), username=config.get("cdets_audit_info","user"),password=config.get("cdets_audit_info","passwd"),auth='LDAP',configuration={'hive.auto.convert.join':'false','mapred.mappers.tasks':'25','mapred.job.shuffle.input.buffer.percent':'0.50','mapreduce.map.memory.mb':'7000','mapreduce.reduce.memory.mb':'7000','mapred.reduce.child.java.opts':'-Xmx8000m','mapred.map.child.java.opts':'-Xmx8000m','hive.exec.reducers.bytes.per.reducer':'104857600','hive.optimize.skewjoin':'true'})
    sqlStr = "select cdets_etl_audit.identifier Identifier,cdets_etl_audit.last_mod_on ModifiedDate,cdets_etl_audit.old_val OldValue,cdets_etl_audit.new_val NewValue,cdets_etl_audit.field_name FieldName from quality_cdtetlprd_siebel.CDETS_ETL_AUDIT where identifier in (" + qStr + ") and field_name in ('Component')"

    df = pd.read_sql( sqlStr, conn)
    return df

def get_hive_connection(config):

    conn = hive.Connection(host=config.get("SR_Source","hive_host"), port=config.get("SR_Source","hive_port"), username=config.get("SR_Source","hive_user"),password=config.get("SR_Source","hive_pwd"),auth='LDAP',configuration={'hive.execution.engine':'tez','hive.vectorized.execution.enabled':'true','hive.vectorized.execution.reduce.enabled':'true','hive.auto.convert.join':'false','mapred.mappers.tasks':'25','mapred.job.shuffle.input.buffer.percent':'0.50','mapreduce.map.memory.mb':'9000','mapreduce.reduce.memory.mb':'9000','mapred.reduce.child.java.opts':'-Xmx9000m','mapred.map.child.java.opts':'-Xmx9000m','hive.exec.reducers.bytes.per.reducer':'104857600','hive.optimize.skewjoin':'true'})

    return conn

def get_kpi_sr_details(config, pfStr, opDir, refresh=2, stDate=None, enDate=None):

    print("In get_kpi_sr_details")

    conn = get_hive_connection(config)
    queryColNames = config.get("IQS","SRDetails_Query_Fields")
    queryColNames = queryColNames.strip('\"')

    colNames = queryColNames.split(",")
    colNames = [  re.sub(r'(.*\.)(.*)', r'\2',col) for col in colNames]

    sqlStr = ""
    if refresh == 1:
        sqlStr = "SELECT DISTINCT " + queryColNames + "  FROM caaggr.kpi_sr_details details WHERE details.product_family  IN ('" + pfStr +  "') and details.rma_count > 0 "
        if stDate:
            sqlStr = sqlStr + " AND to_date(details.sr_create_timestamp) between to_date('" + stDate + "') and to_date('" + enDate + "') "

    else:
        sqlStr = "SELECT DISTINCT " + queryColNames + " ,inc.incident_number FROM caaggr.kpi_sr_details details JOIN tsbi_pl.tss_incidents_current_f inc "
        sqlStr = sqlStr + " ON details.sr_number = inc.incident_number  AND details.product_family  IN ('" + pfStr +  "') and details.rma_count > 0 "
        if stDate:
            sqlStr = sqlStr + " AND to_date(inc.bl_last_update_date) between to_date('" + stDate + "') and to_date('" + enDate + "') "

    print(sqlStr)

    df = pd.DataFrame()

    try:
        df = pd.read_sql(sqlStr, conn )
        if refresh == 2:
            df.drop(df.columns[len(df.columns)-1], axis=1, inplace=True)
#         df.columns = colNames.split(",")
        print(df.shape)
        df.columns = colNames
        year = stDate.split("-")[0]
        detailsFile = opDir + "/" +"SRDetails_" + pfStr + ".csv"
        df.to_csv(detailsFile, index=False)
    except Exception as e:
        print(e)
        raise e

    conn.close()
    return df, detailsFile


def get_sr_notes(config, pf, incStr, opDir, refresh=2, stDate=None, enDate=None, itr=0):

    print("In get_sr_notes")

    conn = get_hive_connection(config)

    sqlStr = ""
    if refresh == 1:
        sqlStr = "SELECT notesData.caseNumber,  notesData.note_type, notesData.notes , notesData.notes_detail FROM  tsbi_pl.tss_incident_notes_f notesData WHERE "
        sqlStr = sqlStr + " notesData.caseNumber IN ( " + incStr + ") "
#         sqlStr = sqlStr + " notesData.source_object_id IN ( " + incStr + ") "
    else:
        sqlStr = "SELECT notesData.caseNumber,notesData.notes,notesData.note_type, notesData.notes , notesData.notes_detail FROM  tsbi_pl.tss_incident_notes_f notesData WHERE "
        sqlStr = sqlStr + " notesData.caseNumber IN ( " + incStr + ") "
#         sqlStr = sqlStr + " notesData.source_object_id IN ( " + incStr + ") "

        if stDate:
            sqlStr = sqlStr + " AND to_date(notesData.bl_last_update_date) between to_date('" + stDate + "') and to_date('" + enDate + "') "

#     print(sqlStr)

    df = pd.DataFrame()
    colNames = ["casenumber","note_type","notes","notes_detail"]
#     colNames = ["sr_number","notes","notes_detail"]
#     notesFileTemp = config.get("SR_Source","srFilesLoc")  + pf + "/" + "SRNotes_" + pf + "_" + str(itr) + "_temp.csv"
    year = stDate.split("-")[0]
    notesFile = opDir + "/"+ "SRNotes_" + pf + "_" + str(itr) + ".csv"

    try:

        # Combine all notes
#         if itr == 0 :
#             df = pd.read_csv(notesFile)
#         else:
        df = pd.read_sql(sqlStr, conn )
        df.columns = colNames
        df.to_csv(notesFile, index=False)

        print(df.shape)

        df.to_csv(notesFile, index=False)
        # Remove special Characters like ^M, ^@
        cmd = "sed -i 's/\\x0//g' " + notesFile
        print(cmd)
        subprocess.check_call(cmd, shell=True, stderr=subprocess.STDOUT)


        df = pd.read_csv(notesFile,low_memory=False)

    except Exception as e:
        print(e)
        raise e

    conn.close()
    return df

def get_FA_SRCases_PF(config, pf, stDate, enDate):
    conn = get_hive_connection(config)
    queryColNames = config.get("SR_Source","SRDetails_Query_Fields")
    queryColNames = queryColNames.strip('\"')

    colNames = config.get("SR_Source","SRDetails_Fields")
    colNames = colNames.strip('\"')


    sql = "SELECT DISTINCT " +  queryColNames + ",notesData.notes, notesData.note_type, notesData.notes_detail FROM caaggr.kpi_sr_details details JOIN tsbi_pl.tss_incident_notes_f notesData ON "
    sql = sql + " details.caseNumber = notesData.casenumber "
#     sql = sql + " AND notesData.note_type in ('CISCO_RES_SUMMARY','CS_PROBLEM','CISCO_KT_PROB_ANALYSIS','CISCO_CUST_SYMPTOM','CISCO_CASE_REVIEW') " +
    if stDate:
            sql = sql + " AND to_date(details.sr_create_timestamp) between to_date('" + stDate + "') and to_date('" + enDate + "') "

    print(sql)
    df = pd.DataFrame()
    colNamesArr = colNames.split(",")
    colNamesArr.append("notes")
    colNamesArr.append("note_type")
    colNamesArr.append("notes_detail")

    try:
        df = pd.read_sql(sql, conn )
        print(df.shape)
        df.columns = colNamesArr
    except Exception as e:
        print(e)
        raise e

    conn.close()

    return df




def get_SRCN_inc_id(config, srStr):
    conn = get_hive_connection(config)


    sql = "SELECT  DISTINCT inc.incident_id, inc.casenumber  FROM tsbi_pl.tss_incidents_current_f inc "
    sql = sql +  " WHERE inc.incident_id IN (" + srStr + ")"

    print(sql)
    df = pd.DataFrame()

    try:
        df = pd.read_sql(sql, conn )
        print(df.shape)
        df.columns = ["sr_number","casenumber"]
    except Exception as e:
        print(e)
        raise e

    conn.close()

    return df

def get_SRNotes_casenumber(config, srStr):
    conn = get_hive_connection(config)

    sql = "SELECT  notesData.casenumber,notesData.note_type, notesData.notes_details  FROM tsbi_pl.tss_incident_notes_f notesData  "
    sql = sql +  " WHERE notesData.casenumber IN (" + srStr + ")"

    print(sql)
    df = pd.DataFrame()
    colNamesArr = colNames.split(",")

    try:
        df = pd.read_sql(sql, conn )
        print(df.shape)
        df.columns = colNamesArr
    except Exception as e:
        print(e)
        raise e

    conn.close()

    return df

def get_SRNotes_casenumber_join(config, srStr):
    conn = get_hive_connection(config)
    queryColNames = config.get("SR_Source","FA_SRDetails_Query_Fields")
    queryColNames = queryColNames.strip('\"')

    colNames = config.get("SR_Source","FA_SRDetails_Fields")
    colNames = colNames.strip('\"')


    sql = "SELECT  " +  queryColNames + " FROM tsbi_pl.tss_incidents_current_f inc LEFT OUTER JOIN tsbi_pl.tss_incident_notes_f notesData  "
    sql = sql +  " ON inc.casenumber = notesData.casenumber "
    sql = sql +  " AND inc.incident_id IN (" + srStr + ")"

    print(sql)
    df = pd.DataFrame()
    colNamesArr = colNames.split(",")

    try:
        df = pd.read_sql(sql, conn )
        print(df.shape)
        df.columns = colNamesArr
    except Exception as e:
        print(e)
        raise e

    conn.close()

    return df

def get_SRNotes_src_obejct_id(config, srStr):
    conn = get_hive_connection(config)
    queryColNames = config.get("SR_Source","FA_SRDetails_Query_Fields")
    queryColNames = queryColNames.strip('\"')

    colNames = config.get("SR_Source","FA_SRDetails_Fields")
    colNames = colNames.strip('\"')


    sql = "SELECT  " +  queryColNames + " FROM tsbi_pl.tss_incident_notes_f notesData  "
    sql = sql +  " where notesData.source_object_id IN (" + srStr + ")"

    print(sql)
    df = pd.DataFrame()
    colNamesArr = colNames.split(",")

    try:
        df = pd.read_sql(sql, conn )
        print(df.shape)
        df.columns = colNamesArr
    except Exception as e:
        print(e)
        raise e

    conn.close()

    return df
def get_FA_SRCases(config, srStr):

    conn = get_hive_connection(config)
    queryColNames = config.get("SR_Source","FA_SRDetails_Query_Fields")
    queryColNames = queryColNames.strip('\"')

    colNames = config.get("SR_Source","FA_SRDetails_Fields")
    colNames = colNames.strip('\"')


    sql = "SELECT  " +  queryColNames + " FROM tsbi_pl.tss_incidents_current_f details LEFT OUTER JOIN tsbi_pl.tss_incident_notes_f notesData ON "
    sql = sql + " details.caseNumber = notesData.casenumber "
# #     sql = sql + " AND notesData.note_type in ('CISCO_RES_SUMMARY','CS_PROBLEM','CISCO_KT_PROB_ANALYSIS','CISCO_CUST_SYMPTOM','CISCO_CASE_REVIEW') " +
    sql = sql +  " AND details.incident_id IN (" + srStr + ")"
#     sql = "SELECT  " +  queryColNames + " FROM tsbi_pl.tss_incident_notes_f notesData  "
#     sql = sql + " details.caseNumber = notesData.casenumber "
#     sql = sql + " AND notesData.note_type in ('CISCO_RES_SUMMARY','CS_PROBLEM','CISCO_KT_PROB_ANALYSIS','CISCO_CUST_SYMPTOM','CISCO_CASE_REVIEW') " +
#     sql = sql +  " where notesData.casenumber IN (" + srStr + ")"

#     print(sql)
    df = pd.DataFrame()
    colNamesArr = colNames.split(",")
#     colNamesArr.append("note_type")
#     colNamesArr.append("notes_detail")

    try:
        df = pd.read_sql(sql, conn )
        print(df.shape)
        df.columns = colNamesArr
    except Exception as e:
        print(e)
        raise e

    conn.close()

    return df

def get_SRCases(config, pfStr, opDir, refresh=2, stDate=None, enDate=None ):

    # Get SR Numbers first
    sr_details, sr_details_file = get_kpi_sr_details(config, pfStr, opDir, refresh,  stDate, enDate)
    year = stDate.split("-")[0]
#     sr_details = pd.read_csv(opDir +  "/" + "SRDetails_" + pfStr + ".csv", low_memory=False)
#     colNames = sr_details.columns
#     colNames = [  re.sub(r'(.*\.)(.*)', r'\2',col) for col in colNames]
#     sr_details.columns = colNames

    final_sr_data =  pd.DataFrame()
    notesFile = opDir + "/" + "SRNotes_" + pfStr + ".csv"

    if len(sr_details) >0 :
        print("Found New SR Cases")

        sr_details  = sr_details[["sr_number","caseNumber"]]
        sr_details = sr_details.sort_values(by=["caseNumber"])
        incs = sr_details["caseNumber"].unique().astype(str)
        inc_list = formINClause(incs,"int")

        notes_detail = pd.DataFrame()

        i = 0

        tempDir = opDir + "/Notes"
        if not os.path.exists(tempDir):
                os.makedirs(tempDir)

        for incl in inc_list:
#             ndetails = get_sr_notes(config, pfStr, incl, tempDir, refresh, stDate, enDate, i)
            ndetails = pd.read_csv(tempDir + "/" + "SRNotes_" + pfStr + "_" + str(i) + ".csv")
            if len(ndetails) > 0:
#                 ndetails = pd.read_csv(tempDir + "/" + "SRNotes_" + pfStr + "_" + str(i) + ".csv")
                i = i + 1
                notes_detail = notes_detail.append(ndetails)

#         notes_detail = pd.read_csv(opDir + "/" + "SRNotes_" + pfStr + ".csv")
        notes_detail.columns = ["caseNumber","note_type","notes","notes_detail"]



        if len(notes_detail) > 0:


            final_sr_data = pd.merge(sr_details, notes_detail, how='left', on=['caseNumber'])

            finalFile = opDir + "/" + config.get("SR_Source","FilePrefix")  + pfStr + ".csv"
            final_sr_data.to_csv(finalFile, index=False)

            cmd = "sed -i 's/\\x0//g' " + finalFile
            print(cmd)
            subprocess.check_call(cmd, shell=True, stderr=subprocess.STDOUT)
#         else:
#             final_sr_data = sr_details


#         unlink(tempDir)
#         notes_detail['caseNumber'] = notes_detail['caseNumber'].astype('int')
#         finalFile = opDir + "/" + config.get("SR_Source","FilePrefix")  + pfStr + ".csv"
#         final_sr_data.to_csv(finalFile, index=False)
#
#         # Remove special Characters like ^M, ^@
#         cmd = "sed -i 's/\\x0//g' " + finalFile
#         print(cmd)
#         subprocess.check_call(cmd, shell=True, stderr=subprocess.STDOUT)

    else:
        print("No new changes found")

#     os.unlink(notesFile)
    return final_sr_data




def getSRData_for_PF( config, pfStr, yr='2016', stDate=None, enDate=None ):
    print("pfStr: "+pfStr)
    print("yr: "+yr)
    print("stDate: "+stDate)
    print("enDate: "+enDate)

    conn = hive.Connection(host=config.get("SR_Source","hive_host"), port=config.get("SR_Source","hive_port"), username=config.get("SR_Source","hive_user"),password=config.get("SR_Source","hive_pwd"),auth='LDAP',configuration={'hive.execution.engine':'tez','hive.vectorized.execution.enabled':'true','hive.vectorized.execution.reduce.enabled':'true','hive.auto.convert.join':'false','mapred.mappers.tasks':'25','mapred.job.shuffle.input.buffer.percent':'0.50','mapreduce.map.memory.mb':'9000','mapreduce.reduce.memory.mb':'9000','mapred.reduce.child.java.opts':'-Xmx9000m','mapred.map.child.java.opts':'-Xmx9000m','hive.exec.reducers.bytes.per.reducer':'104857600','hive.optimize.skewjoin':'true'})
    sqlStr = "select details.*, inc.incident_number, notesData.notes, notesData.notes_detail from caaggr.kpi_sr_details details join tsbi_pl.tss_incidents_current_f inc on inc.incident_number= details.sr_number join tsbi_pl.tss_incident_notes_f notesData on inc.incident_id= notesData.source_object_id and details.sr_hw_product_erp_family in ('"+ pfStr + "') "

    df = pd.DataFrame()


    if stDate:
        sqlStr = sqlStr + " and to_date(inc.bl_creation_date) between to_date('" + stDate + "') and to_date('" + enDate + "') "
    print("Before executing query")
    try:
        print(sqlStr)
        df = pd.read_sql(sqlStr, conn)
        df.columns = ["sr_number","caseNumber","bl_customer_key","sr_resolution_code","sr_customer_activity_code","sr_current_sa_number","sr_current_severity","sr_closed_date","sr_hw_product_erp_family","sr_hw_product_erp_platform","sr_hw_product_number","sr_hw_product_platform","sr_underlying_cause_desc","end_date_active","bl_delete_flag",'sr_sw_product_version_number','sr_sw_product_version','sr_sw_platform','sr_sw_product_part_description','sr_sw_product_part_number','sr_initial_severity','sr_max_severity','sr_contact_cco_id', 'sr_contact','sr_problem_code', 'sr_product_serial_number','sr_problem_summary',
 'sr_troubleshooting_description', 'sr_underlying_cause_code','sr_outage', 'sr_incident_type','sr_complexity_level','sr_email_contact_details','sr_create_timestamp',
 'sr_current_entitlement_type','sr_close_date_calendar_key','sr_sub_tech_name', 'sr_defect_number', 'sr_current_status', 'sr_creator_work_group','sr_defect_cnt', 'sr_owner_cec_id','sr_owner_work_group',
 'sr_tac_hw_family','sr_tech_name','sr_customer_activity_desc','c3_bug_severity', 'creation_contract_svc_line_id','hw_version_act_id','sw_version_id','tac_product_sw_key', 'updated_cot_tech_key','bl_contact_key', 'sr_duns_number','sr_cust_mkt_segment',
 'sr_cust_vertical_mkt','customer_country','cust_theater','sr_customer','customer_region_name','item_name','inventory_item_id','line_category','line_number',
 'line_status_code','order_number', 'service_level_code','service_level_desc', 'bug_headline','bug_status' ,'cdets_severity', 'bug_description','badcodefixid','identifier',
 'closed_on','created_on','impact', 'de_manager' ,'found' ,'origin', 'original_found', 'product', 'project', 'regression' ,'resolved_on', 'component',
 'submitted_on' ,'platform', 'sr_closetime_cal_month_name','sr_closetime_cal_quarter_name','assr_closetime_cal_year_name', 'sr_close_calendar_year_name','fiscal_month_id',
 'sr_technology_group_name', 'sr_ent_business_unit', 'sr_ent_product_family' ,'rma_count', 'shipped_qty', 'covered_id', 'product_family' ,'sr_product_id',
 'fn_subscription' ,'sr_survey_opt_out' ,'ext_be', 'ext_sub_be','int_be','int_sub_be', 'sales_territory_name','l1_sales_territory_descr', 'l2_sales_territory_descr', 'l3_sales_territory_descr', 'l4_sales_territory_descr','l5_sales_territory_descr', 'l6_sales_territory_descr','iso_country_code','iso_country_name', 'cco_id', 'access_level','contact_party_name', 'submitter_company_id',
 'submitter_company_name' ,'cpr_country', 'primary_contact_email' ,'first_closed_on', 'duplicate_of', 'duplicate_on', 'incident_number', 'notes', 'notes_detail']
        print("After running query")
        print(df.shape)
    except Exception as e:
        print(e)
        raise e

    return df

def getUpdatedSRData_for_PF( config, pfStr, yr='2016', stDate=None, enDate=None ):
    print("pfStr: "+pfStr)
    print("yr: "+yr)
    print("stDate: "+stDate)
    print("enDate: "+enDate)
    conn = hive.Connection(host=config.get("SR_Source","hive_host"), port=config.get("SR_Source","hive_port"), username=config.get("SR_Source","hive_user"),password=config.get("SR_Source","hive_pwd"),auth='LDAP',configuration={'hive.auto.convert.join':'false','mapred.mappers.tasks':'25','mapred.job.shuffle.input.buffer.percent':'0.50','mapreduce.map.memory.mb':'2000','mapreduce.reduce.memory.mb':'2000','mapred.reduce.child.java.opts':'-Xmx2000m','mapred.map.child.java.opts':'-Xmx2000m','hive.exec.reducers.bytes.per.reducer':'104857600','hive.optimize.skewjoin':'true'})
    sqlStr = "select details.*, inc.incident_number, notesData.notes, notesData.notes_detail from caaggr.kpi_sr_details details join tsbi_pl.tss_incidents_current_f inc on inc.incident_number= details.sr_number join tsbi_pl.tss_incident_notes_f notesData on inc.incident_id= notesData.source_object_id and details.sr_hw_product_erp_family in ('"+ pfStr + "') "

    df = pd.DataFrame()
    if stDate:
        sqlStr = sqlStr + " and to_date(inc.bl_last_update_date) between to_date('" + stDate + "') and to_date('" + enDate + "') "
    print("Before executing query")
    try:
         df = pd.read_sql(sqlStr, conn)
         df.columns = ["sr_number","caseNumber","bl_customer_key","sr_resolution_code","sr_customer_activity_code","sr_current_sa_number","sr_current_severity","sr_closed_date","sr_hw_product_erp_family","sr_hw_product_erp_platform","sr_hw_product_number","sr_hw_product_platform","sr_underlying_cause_desc","end_date_active","bl_delete_flag",'sr_sw_product_version_number','sr_sw_product_version','sr_sw_platform','sr_sw_product_part_description','sr_sw_product_part_number','sr_initial_severity','sr_max_severity','sr_contact_cco_id', 'sr_contact','sr_problem_code', 'sr_product_serial_number','sr_problem_summary',
 'sr_troubleshooting_description', 'sr_underlying_cause_code','sr_outage', 'sr_incident_type','sr_complexity_level','sr_email_contact_details','sr_create_timestamp',
 'sr_current_entitlement_type','sr_close_date_calendar_key','sr_sub_tech_name', 'sr_defect_number', 'sr_current_status', 'sr_creator_work_group','sr_defect_cnt', 'sr_owner_cec_id','sr_owner_work_group',
 'sr_tac_hw_family','sr_tech_name','sr_customer_activity_desc','c3_bug_severity', 'creation_contract_svc_line_id','hw_version_act_id','sw_version_id','tac_product_sw_key', 'updated_cot_tech_key','bl_contact_key', 'sr_duns_number','sr_cust_mkt_segment',
 'sr_cust_vertical_mkt','customer_country','cust_theater','sr_customer','customer_region_name','item_name','inventory_item_id','line_category','line_number',
 'line_status_code','order_number', 'service_level_code','service_level_desc', 'bug_headline','bug_status' ,'cdets_severity', 'bug_description','badcodefixid','identifier',
 'closed_on','created_on','impact', 'de_manager' ,'found' ,'origin', 'original_found', 'product', 'project', 'regression' ,'resolved_on', 'component',
 'submitted_on' ,'platform', 'sr_closetime_cal_month_name','sr_closetime_cal_quarter_name','assr_closetime_cal_year_name', 'sr_close_calendar_year_name','fiscal_month_id',
 'sr_technology_group_name', 'sr_ent_business_unit', 'sr_ent_product_family' ,'rma_count', 'shipped_qty', 'covered_id', 'product_family' ,'sr_product_id',
 'fn_subscription' ,'sr_survey_opt_out' ,'ext_be', 'ext_sub_be','int_be','int_sub_be', 'sales_territory_name','l1_sales_territory_descr', 'l2_sales_territory_descr', 'l3_sales_territory_descr', 'l4_sales_territory_descr','l5_sales_territory_descr', 'l6_sales_territory_descr','iso_country_code','iso_country_name', 'cco_id', 'access_level','contact_party_name', 'submitter_company_id',
 'submitter_company_name' ,'cpr_country', 'primary_contact_email' ,'first_closed_on', 'duplicate_of', 'duplicate_on', 'incident_number', 'notes', 'notes_detail']
         print("After running query")
         print(df.shape)
    except Exception as e:
         print(e)


    return df


'''
    This is a generic utility package for ingestion of the Engineer data at Query id and Queryid/DE Manager level.
'''

def get_config_object():
    # Read Configuration
    config = configparser.ConfigParser()
    config.read('/data/ingestion/config.ini')
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
        mgrdestination = config.get('cqi_engineering_data_ingestion', 'cqi_engineer_mgr_data_collection')
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
        parameterDic.setdefault('eng_mgr_data_destination', mgrdestination)
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
        env = "stage"
        db_type = "stage"

        hostname = config.get('csap_stage_database', 'host')
        port = config.get('csap_stage_database', 'dbPort')
        username = config.get('csap_stage_database', 'user')
        passwd = config.get('csap_stage_database', 'passwd')
        db = config.get('csap_stage_database', 'db')
        type = config.get('csap_stage_database', 'type')
        mongo_connection_url = "mongodb://" + username + ":" + passwd + "@" + hostname + ":" + port + "/?authSource=" + db
        config_source = config.get('cqi_engineering_data_ingestion', 'cqi_config_source_collection')
        endpoint = config.get('cqi_engineering_data_ingestion', 'cqi_engineer_data_endpoint')
        endpointusername = config.get('cqi_engineering_data_ingestion', 'cqi_generic_username')
        endpointpassword = config.get('cqi_engineering_data_ingestion', 'cqi_generic_password')
        destination = config.get('cqi_engineering_data_ingestion', 'cqi_engineer_data_collection')
        mgrdestination = config.get('cqi_engineering_data_ingestion', 'cqi_engineer_mgr_data_collection')
        number_of_years_of_data = config.get('cqi_engineering_data_ingestion', 'number_of_years_of_data')
        cqi_orcl_db_sid = config.get('cqi_oracle_stage_database', 'db')
        cqi_orcl_db_username = config.get('cqi_oracle_stage_database', 'user')
        cqi_orcl_db_password = config.get('cqi_oracle_stage_database', 'passwd')
        cqi_orcl_db_host = config.get('cqi_oracle_stage_database', 'host')
        cqi_orcl_db_dbPort = config.get('cqi_oracle_stage_database', 'dbPort')
        cqi_orcl_db_dbType = config.get('cqi_oracle_stage_database', 'type')
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
        parameterDic.setdefault('eng_mgr_data_destination', mgrdestination)
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
def get_weekend_dates_for_years(number_of_years, queryidList, env, qrylevel):
    # For every view id configured we check if the collection exist( for new query id we will not have a collection)
    # if it exists then we take the max date avaialble and calculate the number of week ends between current week
    # and max date. We load data only for these week ends.

    #List to store the bu and week end dates for which data needs to be loaded
    weekend_array = []

    #Filter unique list of BU names
    bunames = list(set([queryrow["bu"] for queryrow in queryidList]))

    # Get the env paramters for the specified env
    envParameters = get_env_parameters(env)


    # Get the db connection so that we can connect to the mongo DB
    db = get_db_connection_obj(env, envParameters['db'])

    #get the list of collections
    collections = db.collection_names();

    # Calculate the current week number for current year
    weeks = Week.thisweek().week

    # For previous years
    for i in range(1, number_of_years):
        # calculate the year
        year = date.today().year
        # Check how many weeks are there in the year calculated above and add it to weeks
        weeks = weeks + Week.last_week_of_year(year).week

    # define an empty list to store the week end dates
    week_array = [] # week end dates in date time format
    str_week_array = [] # week end dates in string format

    # Now we will have to loops "weeks" number of times and get the week end date
    dt = date.today()
    for i in range(0, weeks):
        strd = dt - timedelta(days=dt.isoweekday() + 1  + (7 * i))
        week_array.append(strd) #week_array will have week end dates for specified number of years.
        str_week_array.append(strd.strftime('%Y-%m-%d'))

    # Loop through the BU names
    for queryidrow in queryidList:
        # Derive the destination collection name
        if qrylevel == 'MGR':
        	ibdDest = queryidrow["bu"].replace(" ", "_").upper() + envParameters["eng_mgr_data_destination"]#"_CQI_IBD_MGR_ENG";
        else:
                ibdDest = queryidrow["bu"].replace(" ", "_").upper() + envParameters["eng_data_destination"]#"_CQI_IBD_ENG";

        #Loop through all collection and check if the collection exists
        colexists = False
        for col in collections:
            if col == ibdDest:
                colexists = True
                #Collection exist, so we need to fetch the max date from the collection
                weekenddate = db[col].find({"QUERY_ID":int(queryidrow["_id"])}).sort([('week_end_date', -1)]).limit(1)

                if weekenddate.count() == 0:
                    # We find no collection for the specified BU and hence it must be the new BU
                    weekend_array.append([queryidrow["bu"], int(queryidrow["_id"]), str_week_array])
                else:
                    #Loop through the result set. This cursor will always have max of one record.
                    for row in weekenddate:
                        weekarray = []
                        #We will have to filter out all the week end dates for which data has already been loaded in the collection
                        for dte in week_array:
                            #Check if the max date in the collection is less than the week end date in weekenddate collection
                            if dte > datetime.datetime.strptime(row['week_end_date'], "%Y-%m-%d").date():
                                #We have a week end date in the past specified number of years for which data is not present in the collection
                                #So we need to add it to our list of week ends for which we need to load the data.
                                weekarray.append(dte.strftime('%Y-%m-%d'))
                        # Different collections can have different Max dates and hence different sets of dates for which data load is required
                        #hence we append the BU name also
                        if len(weekarray) != 0:
                            weekend_array.append([queryidrow["bu"], int(queryidrow["_id"]), weekarray])
                #We are done, we need to loop through rest as there can only be one collection with specified name. So we break
                break
        if colexists != True:
            weekend_array.append([queryidrow["bu"], int(queryidrow["_id"]), str_week_array])
    return weekend_array


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
