# SNOWFLAKES-HACKATHON

This is a documentation for Snowflakes tutorial

## SNOWSQL TUTORIAL
To run this, I'm using Snowsql which is the CLI tool for running Snowflakes commands, refer to the doc at [snowsql](https://docs.snowflake.com/en/user-guide/snowsql.html)

**NOTE: We won't be using snowsql for out demo, as we will be performing these below steps using Python connector**

<br />

#### INSTALLATION / SETUP

```bash
Run brew install --cask snowflake-snowsql
Open (or create, if missing) the ~/.zshrc file
Add the following lines to the file: 
1) vi ~/.zshrc            --> alias snowsql=/Applications/SnowSQL.app/Contents/MacOS/snowsql
2) vi ~/.snowsql/config   --> change log directory to ~/.snowsql/snowsql_rt.log
To test if the download worked correctrly, type snowsql -v
```

<br />

#### SNOWSQL COMMANDS

Below are the steps to login to snowsql terminal and execute the commands

```bash
LOGIN 						--> snowsql -a "GITYONJ-FZ10313" -o log_level=DEBUG -u ARUNAZ # When prompt for password type "A_b@150695"
CREATE DATABASE 			--> create or replace database sf_tuts;
SELECT DATABASE				--> select current_database(), current_schema();
CREATE TABLE				--> create or replace table emp_basic (
                                       first_name string ,
                                       last_name string ,
                                       email string ,
                                       streetaddress string ,
                                       city string ,
                                       start_date date
                                       );
CREATE WAREHOUSE			--> create or replace warehouse sf_tuts_wh with
                                       warehouse_size='X-SMALL'
                                       auto_suspend = 180
                                       auto_resume = true
                                       initially_suspended=true;
                                       
LOAD DATA FROM FILE			--> put file:///Users/kqkk509/Downloads/getting-started/employees0*.csv @sf_tuts.public.%emp_basic;

VIEW STAGED FILES LIST		--> list @sf_tuts.public.%emp_basic;

COPY DATA INTO TARGET TABLE	--> copy into emp_basic
                                   from @%emp_basic
                                   file_format = (type = csv field_optionally_enclosed_by='"')
                                   pattern = '.*employees0[1-5].csv.gz'
                                   on_error = 'skip_file';
                                   
QUERY DATABASE TABLE		--> select * from emp_basic;

INSERT ADDITIONAL ROWS		--> insert into emp_basic values
                                   ('Clementine','Adamou','cadamou@sf_tuts.com','10510 Sachs Road','Klenak','2017-9-22') ,
                                   ('Marlowe','De Anesy','madamouc@sf_tuts.co.uk','36768 Northfield Plaza','Fangshan','2017-1-26');
                                   
QUERY EXAMPLES				--> select email from emp_basic where email like '%.uk';

DROP DATABASE				--> drop database if exists sf_tuts;

DROP WAREHOUSE				--> drop warehouse if exists sf_tuts_wh;

EXIT THE CONNECTION			--> !exit
```

## SNOWFLAKES PYTHON COMMANDS
<br />

To download python 3.8 click on [python3.8-macOS 64-bit installer](https://www.python.org/ftp/python/3.8.0/python-3.8.0-macosx10.9.pkg)

```bash
Use either virtualenv or venv to isolate the Python runtime environments for our demo purpose. To create venv, execute following command
$ virtualenv -p /usr/local/bin/python3.8/ <virtualenvname> (or) create a virtualenv in pYCHARM WITH 3.8 Interpreter

To download the required packages, run the following
$ pip install -r requirements.txt

To test if the snowpark-python connector is downloaded correctly, run 
$ python validate.py
```