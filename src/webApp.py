import streamlit as st
from snowpark import *

def main():
    st.title("SNOWFLAKE HACKATHON - DRUG REVIEW")
    tab1, tab2 = st.tabs(["Exploratory Data Analysis", "Prediction"])

    with tab1:
        session = create_session()
        CommonConditonsinPatients = session.sql("SELECT CONDITION, count(*) FROM drugsComTrain_raw GROUP BY CONDITION ORDER BY count(*) DESC").to_df("Conditions", "Count")
        st.write("Most Common Conditions in Patients")
        st.bar_chart(data=CommonConditonsinPatients, x=CommonConditonsinPatients.columns[0],y=CommonConditonsinPatients.columns[1])
        st.write("Drug Availability for patient conditions")
        drugAvailabilityForConditions = session.sql("SELECT COUNT(DISTINCT DRUGNAME) as drugavailable, CONDITION FROM drugsComTrain_raw GROUP BY CONDITION HAVING COUNT(DISTINCT DRUGNAME) >1").to_df("DrugAvailable", "Conditions")
        st.bar_chart(data=drugAvailabilityForConditions,x=drugAvailabilityForConditions.columns[1],y=drugAvailabilityForConditions.columns[0])
        st.write("Drugs which can be used for many conditions")
        drugUsage = session.sql("SELECT COUNT(DISTINCT CONDITION) as drugusagecount, DRUGNAME FROM drugsComTrain_raw GROUP BY DRUGNAME HAVING COUNT(DISTINCT CONDITION) >1").to_df("drugusagecount","Drugname")
        st.bar_chart(data=drugUsage, x=drugUsage.columns[1],y=drugUsage.columns[0])

    with tab2:
        #pass this patientCondition to a model and get a response
        patientCondition = st.text_input('Patient Condition: ', '')
        st.write("The input for model is ", patientCondition)
        st.write("The drug recommended for ", patientCondition, "are")


if __name__ == '__main__':
    main()