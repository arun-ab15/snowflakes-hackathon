#!/usr/bin/env python
import snowflake.connector


def snowflake_login():
    conn = snowflake.connector.connect(
        user='KQKK509',
        password='A_b@150695',
        account='ASTRAZENECA-GITC_HACKATHON',
        role='ACCOUNTADMIN'
        # warehouse=WAREHOUSE,
        # database=DATABASE,
        # schema=SCHEMA
    )
    print("Connection created Successfully")
    return conn


def db_creation(conn):
    """Create warehouse, database and schema if not available already"""
    cursor = conn.cursor()
    try:
        cursor.execute("CREATE WAREHOUSE IF NOT EXISTS TEAM_FLAKERS")
        cursor.execute("CREATE DATABASE IF NOT EXISTS FLAKERS")
        cursor.execute("USE DATABASE FLAKERS")
        cursor.execute("CREATE SCHEMA IF NOT EXISTS DATA_RAW")
        print("Warehouse/Database created Successfully")
    finally:
        cursor.close()


def table_creation(conn):
    """Create the tables under drugreview schema in drugreview_dataset dataset"""
    cursor = conn.cursor()
    try:
        cursor.execute(
            '''CREATE OR REPLACE TABLE drugreview.drugsComTest_raw(uniqueID string, drugName string, condition string,
            review string,rating string, date string, usefulCount string)''')
        cursor.execute(
            '''CREATE OR REPLACE TABLE drugreview.drugsComTrain_raw(uniqueID string, drugName string, condition string,
            review string,rating string, date string, usefulCount string)''')
        print("Tables created Successfully")
    finally:
        cursor.close()


def load_tables(conn):
    """ To load the datasource files to Snowflakes staging table and then copy to original table.
    NOTE: Change the below file location as per your directory structure"""
    cursor = conn.cursor()
    try:
        cursor.execute("PUT 'file:///Users/kqkk509/Box Sync/Arun/Learning/Snowflakes/snowflakes-hackathon/datasource"
                       "/drugsComTest_raw.csv' @DRUGREVIEW.%drugsComTest_raw")
        cursor.execute("COPY INTO DRUGREVIEW.drugsComTest_raw file_format = (type = csv "
                       "field_optionally_enclosed_by='\"')")
        cursor.execute("PUT 'file:///Users/kqkk509/Box Sync/Arun/Learning/Snowflakes/snowflakes-hackathon/datasource"
                       "/drugsComTrain_raw.csv' @DRUGREVIEW.%drugsComTest_raw")
        cursor.execute("COPY INTO DRUGREVIEW.drugsComTrain_raw file_format = (type = csv "
                       "field_optionally_enclosed_by='\"')")
        print("Loaded the tables Successfully")
    finally:
        cursor.close()


def main():
    conn = snowflake_login()
    db_creation(conn)
    table_creation(conn)
    load_tables(conn)
    conn.close()


if __name__ == '__main__':
    main()
