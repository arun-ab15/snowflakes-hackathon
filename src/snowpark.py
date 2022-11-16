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
session = create_session()
#CommonConditonsinPatients = session.sql("SELECT CONDITION, count(*) FROM drugsComTrain_raw GROUP BY CONDITION").to_df("Conditions","Count")

#CommonConditonsinPatients = CommonConditonsinPatients.na.drop()
#print(CommonConditonsinPatients.show())

#query = session.sql("SELECT RATING, count(*) FROM drugsComTrain_raw GROUP BY RATING ").to_df("Ratings","Count")
#print(query.show())

#drugAvailabilityForConditions = session.sql("SELECT COUNT(DISTINCT DRUGNAME) as drugavailable, CONDITION FROM drugsComTrain_raw GROUP BY CONDITION HAVING COUNT(DISTINCT DRUGNAME) >1")
#print(drugAvailabilityForConditions.show())

#drugUsage = session.sql("SELECT COUNT(DISTINCT CONDITION) as drugusagecount, DRUGNAME FROM drugsComTrain_raw GROUP BY DRUGNAME HAVING COUNT(DISTINCT CONDITION) >1")
#print(drugUsage.show())

#prepared_train_table = session.sql("SELECT * FROM drugsComTrain_raw where CONDITION NOT LIKE '%span%'")
#prepared_train_table.write.mode("overwrite").save_as_table("drugsComTrain_prepared")


#prepared_test_table = session.sql("SELECT * FROM drugsComTest_raw where CONDITION NOT LIKE '%span%'")
#prepared_test_table.write.mode("overwrite").save_as_table("drugsComTest_prepared")

#drugNames = session.sql("select DISTINCT(DRUGNAME) FROM drugsComTrain_prepared")
#print(drugNames.show())

#negative_rating = session.sql("SELECT DRUGNAME,COUNT(*) FROM drugsComTrain_prepared where RATING >0 and RATING<=3 and DRUGNAME =")
#print(negative_rating.show())

#neutral_rating = session.sql("SELECT COUNT(*) FROM drugsComTrain_prepared where RATING >3 and RATING<=7 and DRUGNAME ='Harvoni'").collect()
#print(neutral_rating)
#positive_rating = session.sql("SELECT DRUGNAME, COUNT(*) FROM drugsComTrain_prepared where RATING >7 and RATING<=10 and DRUGNAME =")
#print(positive_rating.show())




def get_train_data():
    traindata = session.table("drugsComTrain_prepared")
    return traindata

def get_test_data():
    testdata =  session.table("drugsComTest_prepared")
    return testdata
