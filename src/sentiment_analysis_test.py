import time
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, confusion_matrix, roc_auc_score, recall_score, \
    precision_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from wordcloud import WordCloud, STOPWORDS
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from tqdm.notebook import tqdm as tqdm
from tqdm import trange
from sklearn.preprocessing import Normalizer
import nltk


import re
from bs4 import BeautifulSoup

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')

train = pd.read_csv('../datasource/drugsComTrain_raw.csv')
test = pd.read_csv('../datasource/drugsComTest_raw.csv')
# getting the shapes
print("Shape of train :", train.shape)
print("Shape of test :", test.shape)

train.head()
test.head()

# as both the dataset contains same columns we can combine them for better analysis
data = pd.concat([train, test])

# checking the shape
data.shape

# feature engineering
# let's make a new column review sentiment

data.loc[(data['rating'] >= 7), 'Review_Sentiment'] = 2
data.loc[(data['rating'] == 5), 'Review_Sentiment'] = 1
data.loc[(data['rating'] == 6), 'Review_Sentiment'] = 1
data.loc[(data['rating'] < 5), 'Review_Sentiment'] = 0
data['Review_Sentiment'].value_counts()

### Data Preprocessing
##Basic Data Info

# Check for null values

data.isnull().any()

# we will delete the rows so that the data does not overfits

data = data.dropna(axis=0)

# checking the new shape of the data
data.shape


# removing some stopwords from the list of stopwords as they are important for drug recommendation

stops = set(stopwords.words('english'))

not_stop = ["aren't", "couldn't", "didn't", "doesn't", "don't", "hadn't", "hasn't", "haven't", "isn't", "mightn't",
            "mustn't", "needn't", "no", "nor", "not", "shan't", "shouldn't", "wasn't", "weren't", "wouldn't"]
for i in not_stop:
    stops.remove(i)

data.columns

df_condition = data.groupby(['condition'])['drugName'].nunique().sort_values(ascending=False)
df_condition = pd.DataFrame(df_condition).reset_index()
df_condition.tail(20)

"""Removing medical conditions which have less than 5 drug associated with it in the dataset"""

# setting a df with conditions with less than 5 drugs
df_condition_1 = df_condition[df_condition['drugName'] < 5].reset_index()

all_list = set(data.index)

# deleting them
condition_list = []
for i, j in enumerate(data['condition']):
    for c in list(df_condition_1['condition']):
        if j == c:
            condition_list.append(i)

new_idx = all_list.difference(set(condition_list))
data = data.iloc[list(new_idx)].reset_index()
del data['index']

"""removing the conditions with the word "\span" in it."""

all_list = set(data.index)
span_list = []
for i, j in enumerate(data['condition']):
    if '</span>' in j:
        span_list.append(i)
new_idx = all_list.difference(set(span_list))
data = data.iloc[list(new_idx)].reset_index()
del data['index']

data.shape

"""Applying data cleanup with -"""

stemmer = SnowballStemmer('english')


def review_to_words(raw_review):
    # 1. Delete HTML
    # review_text = BeautifulSoup(raw_review, 'html.parser').get_text()
    # 2. Make a space
    letters_only = re.sub('[^a-zA-Z]', ' ', raw_review)
    # 3. lower letters
    words = letters_only.lower().split()
    # 4. Stopwords
    meaningful_words = [w for w in words if not w in stops]
    # 5. Stemming
    stemming_words = [stemmer.stem(w) for w in meaningful_words]
    # 6. space join words
    return (' '.join(stemming_words))


data['review_clean'] = data['review'].apply(review_to_words)
print("Done with revie clean")

# Analysing Emotions
# Make data directory if it doesn't exist
# !mkdir -p nrcdata
# !wget -nc https://nyc3.digitaloceanspaces.com/ml-files-distro/v1/upshot-trump-emolex/data/NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt -P nrcdata


filepath = "../../Sentiment-Analysis-of-Drug-Reviews/nrcdata/NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt"
emolex_df = pd.read_csv(filepath, names=["word", "emotion", "association"], skiprows=45, sep='\t',
                        keep_default_na=False)
emolex_df.head(12)

emolex_df.emotion.unique()

emolex_df.emotion.value_counts()

emolex_df[emolex_df.association == 1].emotion.value_counts()

emolex_words = emolex_df.pivot(index='word', columns='emotion', values='association').reset_index()
emolex_words.head()
print(emolex_words.head())

def extract_review_emotion(df, column):
    new_df = df.copy()

    emotions = emolex_words.columns.drop('word')
    emo_df = pd.DataFrame(0, index=df.index, columns=emotions)
    stemmer = SnowballStemmer("english")
    with tqdm(total=len(list(new_df.iterrows()))) as pbar:
        for i, row in new_df.iterrows():
            pbar.update(1)
            document = word_tokenize(new_df.loc[i][column])
            for word in document:
                word = stemmer.stem(word.lower())
                emo_score = emolex_words[emolex_words.word == word]
                if not emo_score.empty:
                    for emotion in list(emotions):
                        emo_df.at[i, emotion] += emo_score[emotion]

    new_df = pd.concat([new_df, emo_df], axis=1)
    return new_df


emotion_df = extract_review_emotion(data, 'review_clean')
emo = emotion_df.groupby(['drugName']).sum()
emo.head(10)
print(emo.heaad(10))

emo.to_csv('emotion_groupby11.csv', index=True)
emotion_df.to_csv('emotion_sentiment.csv', index=False)
