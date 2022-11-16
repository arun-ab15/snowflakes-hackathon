import streamlit as st
from snowpark import *

def main():
    st.title("SNOWFLAKE HACKATHON - DRUG REVIEW")
    tab1, tab2,tab3 = st.tabs(["Drugs & Conditions","Ratings","Recommendation"])

    with tab1:
        session = create_session()
        CommonConditonsinPatients = session.sql("SELECT CONDITION, count(*) FROM drugsComTrain_prepared GROUP BY CONDITION ORDER BY count(*) DESC limit 100").to_df("Conditions", "Count")
        CommonConditonsinPatients = CommonConditonsinPatients.na.drop()
        st.write("Most Common Conditions in Patients")
        st.bar_chart(data=CommonConditonsinPatients, x=CommonConditonsinPatients.columns[0],y=CommonConditonsinPatients.columns[1])
        st.write("Drug Availability for patient conditions")
        drugAvailabilityForConditions = session.sql("SELECT COUNT(DISTINCT DRUGNAME) as drugavailable, CONDITION FROM drugsComTrain_prepared GROUP BY CONDITION HAVING COUNT(DISTINCT DRUGNAME) >1 limit 100").to_df("DrugAvailable", "Conditions")
        drugAvailabilityForConditions = drugAvailabilityForConditions.na.drop()
        st.bar_chart(data=drugAvailabilityForConditions,x=drugAvailabilityForConditions.columns[1],y=drugAvailabilityForConditions.columns[0])
        st.write("Drugs which can be used for many conditions")
        drugUsage = session.sql("SELECT COUNT(DISTINCT CONDITION) as drugusagecount, DRUGNAME FROM drugsComTrain_prepared GROUP BY DRUGNAME HAVING COUNT(DISTINCT CONDITION) >1 limit 100").to_df("drugusagecount","Drugname")
        drugUsage = drugUsage.na.drop()
        st.bar_chart(data=drugUsage, x=drugUsage.columns[1],y=drugUsage.columns[0])

    with tab2:
        drugNames = session.sql("select DISTINCT(DRUGNAME) FROM drugsComTrain_prepared limit 100")
        option1 = st.selectbox(
            'Drug you like to know about ?',
            (drugNames))

        st.write('You selected:', option1)
        negative_rating = session.sql(f"SELECT COUNT(*) as No_of_Negative_ratings FROM drugsComTrain_prepared where RATING >0 and RATING<=3 and DRUGNAME = '{option1}'")

        neutral_rating = session.sql(f"SELECT  COUNT(*) as No_of_Neutral_ratings FROM drugsComTrain_prepared where RATING >3 and RATING<=7 and DRUGNAME ='{option1}'")

        positive_rating = session.sql(f"SELECT COUNT(*) as No_of_Positive_ratings FROM drugsComTrain_prepared where RATING >7 and RATING<=10 and DRUGNAME ='{option1}'")

        st.table(data=positive_rating)
        st.table(data=neutral_rating)
        st.table(data=negative_rating)


    with tab3:
        Conditions = session.sql("select DISTINCT(CONDITION) FROM drugsComTrain_prepared limit 100")
        option2 = st.selectbox(
            'Please choose the patient condition',
            (Conditions))
        topdrugs = session.sql(f"select drugname from drugsComTrain_prepared WHERE CONDITION = '{option2}' group by drugname order by avg(RATING) desc limit 5")
        st.write("The top 5 recommended drugs for ",option2," are ")
        st.table(data=topdrugs)

if __name__ == '__main__':
    main()