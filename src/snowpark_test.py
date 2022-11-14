import snowflake.connector
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from snowflake.sqlalchemy import URL
from snowflake.connector.pandas_tools import write_pandas
from snowflake.snowpark.functions import udf
from snowflake.snowpark.types import IntegerType, StringType, StructType, FloatType
from snowflake.snowpark.session import Session
from snowflake.snowpark import Session
import snowflake.snowpark.functions as F
from snowflake.snowpark import types as T
from snowflake.snowpark import Window
from snowflake.snowpark.functions import udf, max, min, count, avg, sum, col, lit, listagg
import mlxtend
from mlxtend.feature_selection import ColumnSelector
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, FunctionTransformer
from sklearn.pipeline import make_pipeline, Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score
from sklearn import datasets