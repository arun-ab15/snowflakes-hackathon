from snowflake.snowpark import *

def create_session():
    connection_parameters = {
    "account": "ASTRAZENECA-GITC_HACKATHON",
    "user": "kgtq746",
    "password": "kathIee@2403",
    "role":"ACCOUNTADMIN",
    "warehouse":"TEAM_FLAKERS",
    "database":"FLAKERS",
    "schema": "DATA_RAW"
    }
    new_session = Session.builder.configs(connection_parameters).create()
    return new_session

#testing snowpark conntectivity
#session = create_session()
#CommonConditonsinPatients = session.sql("SELECT CONDITION, count(*) FROM drugsComTrain_raw GROUP BY CONDITION").to_df("Conditions","Count")

#print(CommonConditonsinPatients.show())

#query = session.sql("SELECT RATING, count(*) FROM drugsComTrain_raw GROUP BY RATING ").to_df("Ratings","Count")
#print(query.show())

#drugAvailabilityForConditions = session.sql("SELECT COUNT(DISTINCT DRUGNAME) as drugavailable, CONDITION FROM drugsComTrain_raw GROUP BY CONDITION HAVING COUNT(DISTINCT DRUGNAME) >1")
#print(drugAvailabilityForConditions.show())

#drugUsage = session.sql("SELECT COUNT(DISTINCT CONDITION) as drugusagecount, DRUGNAME FROM drugsComTrain_raw GROUP BY DRUGNAME HAVING COUNT(DISTINCT CONDITION) >1")
#print(drugUsage.show())

#query = session.sql("SELECT COUNT(*) FROM drugsComTrain_raw where RATING > 1 and RATING <=3")
#print(query.show())