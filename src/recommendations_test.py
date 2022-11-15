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
from tqdm import tqdm_notebook as tqdm
from tqdm import trange
from sklearn.preprocessing import Normalizer
import nltk

nltk.download('punkt')

train = pd.read_csv('../datasource/drugsComTrain_raw.csv')
test = pd.read_csv('../datasource/drugsComTest_raw.csv')

print("Shape of train :", train.shape)
print("Shape of test :", test.shape)

train.head()
test.head()

# as both the dataset contains same columns we can combine them for better analysis
data = pd.concat([train, test])
# checking the shape
data.shape


# EXPLORATORY DATA ANALYSIS
def listToString(s):
    str1 = " "
    return (str1.join(s))


# checking the different types of conditions patients

most_common_condition = data['condition'].value_counts().head(40)
print("Most Common Conditions in the Patients \n")
print(most_common_condition)
# data['condition'].value_counts().head(40).plot.bar(figsize=(15, 7), color='#6A1B4D')
# plt.title('Most Common Conditions in the Patients', fontsize=20)
# plt.xlabel('Conditions', fontsize=20)
# plt.ylabel('Count', fontsize=20)
# plt.show()


most_drugs_available_per_conditions = data.groupby(['condition'])['drugName'].nunique().sort_values(
    ascending=False).head(40)
print("Most drugs available per Conditions in the Patients \n")
print(most_drugs_available_per_conditions)
# data.groupby(['condition'])['drugName'].nunique().sort_values(ascending = False).head(40).plot.bar(figsize = (15, 7), color = '#46B2E0')
# plt.title('Most drugs available per Conditions in the Patients', fontsize = 20)
# plt.xlabel('Conditions', fontsize = 20)
# plt.ylabel('Count', fontsize = 20)
# plt.show()


# checking the most popular drugs per conditions
drugs_used_for_many_conditions = data.groupby(['drugName'])['condition'].nunique().sort_values(ascending=False).head(40)
print("Drugs which can be used for many Conditions in the Patients \n")
print(drugs_used_for_many_conditions)
# data.groupby(['drugName'])['condition'].nunique().sort_values(ascending = False).head(40).plot.bar(figsize = (15, 7), color = '#FF6666')
# plt.title('Drugs which can be used for many Conditions in the Patients', fontsize = 20)
# plt.xlabel('Drug Names', fontsize = 20)
# plt.ylabel('Count', fontsize = 20)
# plt.show()


data['rating'].value_counts()

# size = [68005, 46901, 36708, 25046, 12547, 10723, 8462, 6671]
# colors = ['#FF6666', '#35BDD0', '#B9345A',  '#8D3DAF', '#F7CD2E', '#23C4ED', '#50DBB4', '#E07C24']
# labels = "10", "1", "9", "8", "7", "5", "6", "4"
# my_circle = plt.Circle((0, 0), 0.7, color = 'white')
# plt.rcParams['figure.figsize'] = (10, 10)
# plt.pie(size, colors = colors, labels = labels, autopct = '%.2f%%')
# plt.axis('off')
# plt.title('A Pie Chart Representing the Share of Ratings', fontsize = 20)
# p = plt.gcf()
# plt.gca().add_artist(my_circle)
# plt.legend()
# plt.show()


temp = []
for i in range(1, 11):
    temp.append([i, np.sum(data[data.rating == i].usefulCount) / np.sum([data.rating == i])])
temp = np.asarray(temp)

plt.scatter(temp[:, 0], temp[:, 1], c=temp[:, 0], cmap='rocket_r', s=200, edgecolors='k')
plt.title('Average Useful Count vs Rating', fontsize=20)
plt.xlabel('Rating', fontsize=20)
plt.ylabel('Average Useful Count', fontsize=20)
plt.xticks([i for i in range(1, 11)])

# feature engineering
# let's make a new column review sentiment

data.loc[(data['rating'] >= 7), 'Review_Sentiment'] = 2
data.loc[(data['rating'] == 5), 'Review_Sentiment'] = 1
data.loc[(data['rating'] == 6), 'Review_Sentiment'] = 1
data.loc[(data['rating'] < 5), 'Review_Sentiment'] = 0
data['Review_Sentiment'].value_counts()

stopwords = set(STOPWORDS)
textString = listToString(data["drugName"])
wordcloud = WordCloud(max_words=100, width=1200, height=800, random_state=1, background_color='navy', colormap='Paired',
                      collocations=False, stopwords=STOPWORDS).generate(textString)
plt.rcParams['figure.figsize'] = (15, 15)
plt.title('Most Common Drugs among the patients', fontsize=20)
print(wordcloud)
plt.axis('off')
plt.imshow(wordcloud)
plt.show()

wordcloud = WordCloud(background_color='blue', colormap='Pastel1', stopwords=stopwords, width=1200,
                      height=800).generate(str(data['review']))

plt.rcParams['figure.figsize'] = (15, 15)
plt.title('Most popular words in the reviews', fontsize=20)
print(wordcloud)
plt.axis('off')
plt.imshow(wordcloud)
plt.show()

# making Words cloud for the postive sentiments

positive_sentiments = " ".join([text for text in data['review'][data['Review_Sentiment'] == 2]])
wordcloud = WordCloud(background_color='lightgreen', stopwords=stopwords, width=1200, height=800).generate(
    positive_sentiments)
plt.rcParams['figure.figsize'] = (15, 15)
plt.title('Most Common Words in Positive Reviews', fontsize=20)
print(wordcloud)
plt.axis('off')
plt.imshow(wordcloud)
plt.show()

negative_sentiments = " ".join([text for text in data['review'][data['Review_Sentiment'] == 0]])
wordcloud = WordCloud(background_color='grey', stopwords=stopwords, width=1200, height=800).generate(
    negative_sentiments)
plt.rcParams['figure.figsize'] = (15, 15)
plt.title('Most Common Words in Negative Reviews', fontsize=20)
print(wordcloud)
plt.axis('off')
plt.imshow(wordcloud)
plt.show()

neutral_sentiments = " ".join([text for text in data['review'][data['Review_Sentiment'] == 1]])
wordcloud = WordCloud(background_color='lightblue', stopwords=stopwords, width=1200, height=800).generate(
    neutral_sentiments)
plt.rcParams['figure.figsize'] = (15, 15)
plt.title('Most Common Words in Neutral Reviews', fontsize=20)
print(wordcloud)
plt.axis('off')
plt.imshow(wordcloud)
plt.show()

"""Extracting year,month and day from the date column"""
# converting the date into datetime format
data['date'] = pd.to_datetime(data['date'], errors='coerce')

# now extracting year from date
data['Year'] = data['date'].dt.year

# extracting the month from the date
data['month'] = data['date'].dt.month

# extracting the days from the date
data['day'] = data['date'].dt.day

# looking at the no. of reviews in each of the year

plt.rcParams['figure.figsize'] = (19, 8)
sns.countplot(data['Year'], palette='plasma')
plt.title('The No. of Reviews in each year', fontsize=20)
plt.xlabel('Year', fontsize=20)
plt.ylabel('Count of Reviews', fontsize=20)
plt.show()

# looking at the no. of ratings in each of the year

plt.rcParams['figure.figsize'] = (19, 8)
sns.boxplot(x=data['Year'], y=data['rating'], palette='magma')
plt.title('The Distribution of Ratings in each Year', fontsize=20)
plt.xlabel('Year', fontsize=20)
plt.ylabel('Count of Reviews', fontsize=20)
plt.show()

# looking at the no. of ratings in each of the year

plt.rcParams['figure.figsize'] = (19, 8)
sns.violinplot(x=data['Year'], y=data['Review_Sentiment'])
plt.title('The Distribution of Sentiments in each Year', fontsize=20)
plt.xlabel('Year', fontsize=20)
plt.ylabel('Variation of Sentimens', fontsize=20)
plt.show()

# looking at the no. of reviews in each of the months

plt.rcParams['figure.figsize'] = (19, 8)
sns.countplot(data['month'], palette='Paired')
plt.title('The No. of Reviews in each Month', fontsize=20)
plt.xlabel('Months', fontsize=20)
plt.ylabel('Ratings', fontsize=20)
plt.show()

# looking at the no. of ratings in each of the month

plt.rcParams['figure.figsize'] = (19, 8)
sns.boxplot(x=data['month'], y=data['rating'], palette='magma')
plt.title('The Distribution of Ratings in each month', fontsize=20)
plt.xlabel('Months', fontsize=20)
plt.ylabel('Ratings', fontsize=20)
plt.show()

# looking at the no. of ratings in each of the month

plt.rcParams['figure.figsize'] = (19, 8)
sns.violinplot(x=data['month'], y=data['rating'], palette='inferno')
plt.title('The Distribution of Sentiments in each month', fontsize=20)
plt.xlabel('Months', fontsize=20)
plt.ylabel('Sentiments', fontsize=20)
plt.show()

plt.rcParams['figure.figsize'] = (15, 8)
sns.distplot(data['usefulCount'], color='#E21717')
plt.title('The Distribution of Useful Counts for each of the Reviews', fontsize=20)
plt.xlabel('Range of Useful Counts', fontsize=20)
plt.ylabel('No. of Useful Counts', fontsize=20)
plt.show()

# plotting a stacked bar to see in which year what were the sentiments

df = pd.crosstab(data['month'], data['Review_Sentiment'])
df.div(df.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True, figsize=(19, 8),
                                             color=['#8D3DAF', '#E03B8B', '#35BDD0'])
plt.title('The Distribution of Sentiments for each of the Reviews month-wise', fontsize=20)
plt.xlabel('Month', fontsize=20)
plt.show()

# plotting a stacked bar to see in which year what were the sentiments

df = pd.crosstab(data['day'], data['Review_Sentiment'])
df.div(df.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True, figsize=(19, 8),
                                             color=['lightblue', 'yellow', 'lightgreen'])
plt.title('The Distribution of Sentiments for each of the Reviews day-wise', fontsize=20)
plt.xlabel('Days', fontsize=20)
plt.legend(loc=2)
plt.show()

# plotting a stacked bar to see in which year what were the sentiments

df = pd.crosstab(data['Year'], data['Review_Sentiment'])
df.div(df.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True, figsize=(19, 8),
                                             color=['#B4161B', '#D9D55B', '#FF6263'])
plt.title('The Distribution of Sentiments for each of the Reviews Year-wise', fontsize=20)
plt.xlabel(' Year', fontsize=20)
plt.show()

"""Data Preprocessing"""

data.describe()
data.info()
data.isnull().any()
data['condition'].isnull().sum()

# we will delete the rows so that the data does not overfits

data = data.dropna(axis=0)

# checking the new shape of the data
data.shape


# importing the important libraries

import re
from bs4 import BeautifulSoup

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import PorterStemmer


# removing some stopwords from the list of stopwords as they are important for drug recommendation

stops = set(stopwords.words('english'))

not_stop = ["aren't","couldn't","didn't","doesn't","don't","hadn't","hasn't","haven't","isn't","mightn't",
            "mustn't","needn't","no","nor","not","shan't","shouldn't","wasn't","weren't","wouldn't"]
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
for i,j in enumerate(data['condition']):
    if '</span>' in j:
        span_list.append(i)
new_idx = all_list.difference(set(span_list))
data = data.iloc[list(new_idx)].reset_index()
del data['index']
data.shape

"""Applying data cleanup"""

stemmer = SnowballStemmer('english')

def review_to_words(raw_review):
    # 1. Delete HTML
    review_text = BeautifulSoup(raw_review, 'html.parser').get_text()
    # 2. Make a space
    letters_only = re.sub('[^a-zA-Z]', ' ', review_text)
    # 3. lower letters
    words = letters_only.lower().split()
    # 4. Stopwords
    meaningful_words = [w for w in words if not w in stops]
    # 5. Stemming
    stemming_words = [stemmer.stem(w) for w in meaningful_words]
    # 6. space join words
    return( ' '.join(stemming_words))


# %time data['review_clean'] = data['review'].apply(review_to_words)


"""Dividing into train test and applying count vectorise"""
df_train, df_test = train_test_split(data, test_size = 0.25, random_state = 0)
# checking the shape
print("Shape of train:", df_train.shape)
print("Shape of test: ", df_test.shape)

"""Making a bag of words using CountVectorise"""
df_train = pd.read_csv('/content/df_training_data.csv')
df_test = pd.read_csv('/content/df_testing_data.csv')


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

cv = CountVectorizer(max_features = 5000, lowercase=True, binary=True)
# cv = CountVectorizer()
pipeline = Pipeline([('vect',cv)])

# df_train_features = pipeline.fit_transform(df_train['review_clean'])
# df_train_features = cv.fit_transform(df_train['review_clean']).toarray()
df_train_features = cv.fit_transform(df_train['review_clean'].values.astype('U'))
df_test_features = pipeline.fit_transform(df_test['review_clean'])

print("Performing Bag of Words - CountVectorise\n")
print("df_train_features :", df_train_features.shape)
print("df_test_features :", df_test_features.shape)


df_train_features = df_train_features.toarray()
print(df_train_features)
df_train.columns

y_train = df_train['Review_Sentiment']
y_test = df_test['Review_Sentiment']


from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_test.shape
df_train_features, df_val_features, y_train, y_val = train_test_split(df_train_features,y_train, test_size=0.1)


#Applying ML Models


"""SVM"""
svmClassifier = SVC(kernel="linear", class_weight="balanced", C=0.003)
t0 = time.time()

svmClassifier.fit(df_train_features, y_train)
t1 = time.time()

svmPredictions = svmClassifier.predict(df_test_features)
t2 = time.time()

time_linear_train = t1-t0
time_linear_predict = t2-t1

# results
print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
report = classification_report(y_test, svmPredictions, output_dict=True)
acc_svm=report['accuracy']
print('0.0: ', report['0.0'])
print('1.0: ', report['1.0'])
print('accuracy: ', report['accuracy'])
print('macro avg: ', report['macro avg'])
print('weighted avg: ', report['weighted avg'])


"""Random Forest"""
start = time.time()
rfc = RandomForestClassifier(n_estimators=80)
rfc.fit(df_train_features, df_train['Review_Sentiment'])
end = time.time()
print("Training time: %s" % str(end-start))

# Evaluates model on test set
pred_rf = rfc.predict(df_test_features)

acc_rf=rfc.score(df_test_features, df_test['Review_Sentiment'])
print("Accuracy: %s" % str(acc_rf))



"""Bayes Classification"""
start = time.time()
multiNB = MultinomialNB().fit(df_train_features, df_train['Review_Sentiment'])
end = time.time()
print("Training time: %s" % str(end-start))

# Evaluates model on test set
pred_nb = multiNB.predict(df_test_features)

acc_nb=multiNB.score(df_test_features, df_test['Review_Sentiment'])
print("Accuracy: %s" % str(acc_nb))



##LGBM Classifier


from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, average_precision_score
from sklearn.model_selection import KFold
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix

# folds = KFold(n_splits=5, shuffle=True, random_state=546789)
target = df_train['Review_Sentiment']
feats = ['usefulCount']

sub_preds = np.zeros(df_test.shape[0])

trn_x, val_x, trn_y, val_y = train_test_split(df_train[feats], target, test_size=0.2, random_state=42)
feature_importance_df = pd.DataFrame()



solution = df_test['Review_Sentiment']
print("Accuracy: %s" % str(accuracy_score(solution, sub_preds)))
print(confusion_matrix(y_pred = sub_preds, y_true = solution))

clf = LGBMClassifier(
    n_estimators=2000,
    learning_rate=0.05,
    num_leaves=30,
    # colsample_bytree=.9,
    subsample=.9,
    max_depth=7,
    reg_alpha=.1,
    reg_lambda=.1,
    min_split_gain=.01,
    min_child_weight=2,
    silent=-1,
    verbose=-1,
)

clf.fit(trn_x, trn_y,
        eval_set=[(trn_x, trn_y), (val_x, val_y)],
        verbose=100, early_stopping_rounds=100  # 30
        )

sub_preds = clf.predict(df_test[feats])

fold_importance_df = pd.DataFrame()
fold_importance_df["feature"] = feats
fold_importance_df["importance"] = clf.feature_importances_
feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)


solution = df_test['Review_Sentiment']
print("Accuracy: %s" % str(accuracy_score(solution, sub_preds)))
print(confusion_matrix(y_pred = sub_preds, y_true = solution))



from textblob import TextBlob
from tqdm import tqdm
reviews = data['review_clean']

Predict_Sentiment = []
for review in tqdm(reviews):
    blob = TextBlob(review)
    Predict_Sentiment += [blob.sentiment.polarity]
data["Predict_Sentiment"] = Predict_Sentiment
data.head()

np.corrcoef(data["Predict_Sentiment"], data["rating"])
np.corrcoef(data["Predict_Sentiment"], data["Review_Sentiment"])
reviews = data['review']

Predict_Sentiment = []
for review in tqdm(reviews):
    blob = TextBlob(review)
    Predict_Sentiment += [blob.sentiment.polarity]
data["Predict_Sentiment2"] = Predict_Sentiment

np.corrcoef(data["Predict_Sentiment2"], data["rating"])
np.corrcoef(data["Predict_Sentiment2"], data["Review_Sentiment"])



### Performing feature engineering

# word count in each unclean comment
data['count_sent'] = data["review"].apply(lambda x: len(re.findall("\n",str(x)))+1)

# Word count in each comment:
data['count_word'] = data["review_clean"].apply(lambda x: len(str(x).split()))

# Unique word count
data['count_unique_word'] = data["review_clean"].apply(lambda x: len(set(str(x).split())))

# Letter count
data['count_letters'] = data["review_clean"].apply(lambda x: len(str(x)))

# punctuation count
import string
data["count_punctuations"] = data["review"].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))

# upper case words count
data["count_words_upper"] = data["review"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

# title case words count
data["count_words_title"] = data["review"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))

# Number of stopwords
data["count_stopwords"] = data["review"].apply(lambda x: len([w for w in str(x).lower().split() if w in stops]))

# Average length of the words
data["mean_word_len"] = data["review_clean"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

# adding season
data['season'] = data["month"].apply(lambda x: 1 if ((x>2) & (x<6)) else(2 if (x>5) & (x<9) else (3 if (x>8) & (x<12) else 4)))




len_train = df_train.shape[0]
print(len_train)


df_train = data[:len_train]
df_test = data[len_train:]
df_train.columns



###Training LGBM Classifier on all the features other than the review_clean, to get how all the other features affect the prediction.
target = df_train['Review_Sentiment']

feats = ['usefulCount', 'day', 'Year', 'month', 'Predict_Sentiment', 'Predict_Sentiment2', 'count_sent',
         'count_word', 'count_unique_word', 'count_letters', 'count_punctuations',
         'count_words_upper', 'count_words_title', 'count_stopwords', 'mean_word_len', 'season']

sub_preds = np.zeros(df_test.shape[0])

trn_x, val_x, trn_y, val_y = train_test_split(df_train[feats], target, test_size=0.2, random_state=42)
feature_importance_df = pd.DataFrame()

clf = LGBMClassifier(
    n_estimators=10000,
    learning_rate=0.10,
    num_leaves=30,
    # colsample_bytree=.9,
    subsample=.9,
    max_depth=7,
    reg_alpha=.1,
    reg_lambda=.1,
    min_split_gain=.01,
    min_child_weight=2,
    silent=-1,
    verbose=-1,
)
clf.fit(trn_x, trn_y,
        eval_set=[(trn_x, trn_y), (val_x, val_y)],
        verbose=100, early_stopping_rounds=100)

pred_lgbm = clf.predict(df_test[feats])

fold_importance_df = pd.DataFrame()
fold_importance_df["feature"] = feats
fold_importance_df["importance"] = clf.feature_importances_
feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)


solution = df_test['Review_Sentiment']

acc_lgbm = accuracy_score(solution, pred_lgbm)
print("Accuracy: %s" % str(acc_lgbm))
confusion_matrix(y_pred=pred_lgbm, y_true=solution)



"""Plotting the feature importance"""
cols = feature_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending = False)[:50].index

best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]

plt.figure(figsize=(14,10))
sns.barplot(x="importance", y="feature", data = best_features.sort_values(by="importance", ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()



##Logistic Regression

lr_tfidf = LogisticRegression(C=0.00007, max_iter=200)
lr_tfidf.fit(df_train_features, df_train['Review_Sentiment'])

lr_pred_tfidf = lr_tfidf.predict(df_test_features)

acc_lr = accuracy_score(df_test['Review_Sentiment'],lr_pred_tfidf)
print(acc_lr)
lr_cm = confusion_matrix(df_test['Review_Sentiment'], lr_pred_tfidf)

lr_pred_tfidf=pd.DataFrame(lr_pred_tfidf)
prediction_lr = lr_pred_tfidf.to_csv('prediction_lr.csv')



### Applying DL modules

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Bidirectional
from keras.layers import BatchNormalization
from keras.layers import LSTM
from keras.preprocessing.sequence import pad_sequences

# making our dependent variable
solution = y_test.copy()

# Model Structure
model = Sequential()
model.add(Input(shape=(df_train_features.shape[1],)))
model.add(Dense(300))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(400))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(100, activation = 'relu'))
model.add(Dense(3, activation = 'sigmoid'))

#  Model compile
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# inputs = Input(shape=(df_train_features.shape[1],), sparse=True)
# L1 = (Dense(200))(inputs)
# L1_N = (BatchNormalization())(L1)
# L1_Act = (Activation('relu'))(L1_N)
# L1_Drop = (Dropout(0.5))(L1_Act)
# L2 = (Dense(200))(L1_Drop)
# L2_N = (BatchNormalization())(L2)
# L2_Act = (Activation('relu'))(L2_N)
# L2_Drop = Dropout(0.5)(L2_Act)
# L3 = (Dense(100, activation = 'relu'))(L2_Drop)
# outputs = (Dense(1, activation = 'sigmoid'))(L3)
# model = Model(inputs=inputs, outputs=outputs)
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



model.summary()
print(df_train_features)

df_train['review_clean']


# 4. Train model
hist = model.fit(df_train_features, y_train, epochs=10, batch_size=64, validation_data=(df_val_features, y_val))

# %matplotlib inline
import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.set_ylim([0.0, 1.0])
acc_ax.set_ylim([0.0, 1.0])

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()


y_pred_DL1 = model.predict(df_test_features.toarray())
y_pred_DL1 = y_pred_DL1>0.5
acc_DL1 = accuracy_score(y_test, y_pred_DL1)
print(acc_DL1)

y_pred_DL1=pd.DataFrame(y_pred_DL1)
y_pred_DL1=y_pred_DL1.idxmax(axis=1)
prediction_DL1 = y_pred_DL1.to_csv('prediction_DL1.csv')




##Neural Network 2

df_tr, df_val, y_tr, y_val = train_test_split(df_train['review_clean'].values.astype('U'),y_train, test_size=0.1)
df_tr.shape


from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Dropout
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
from keras.utils import to_categorical
from gensim.models import Word2Vec
from nltk.cluster import KMeansClusterer

import tensorflow_hub as hub
import tensorflow as tf
from numpy.random import seed
hub_layer = hub.KerasLayer("https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1", output_shape=[50],
                           input_shape=[], dtype=tf.string, name='input', trainable=False)
np.random.seed(1)
model = Sequential()
model.add(hub_layer)
model.add(Dense(units=1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=3, activation='softmax', name='output'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
hist = model.fit(df_tr, y_tr, epochs=50, batch_size=128,validation_data=(df_val, y_val))

# %matplotlib inline
import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.set_ylim([0.0, 1.0])
acc_ax.set_ylim([0.0, 1.0])

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()

y_pred = model.predict(df_test['review_clean'])
y_pred_DL2 = (y_pred > 0.5)

acc_DL2 = accuracy_score(y_test,y_pred_DL2)
print(acc_DL2)

y_pred_DL2=pd.DataFrame(y_pred_DL2)
y_pred_DL2=y_pred_DL2.idxmax(axis=1)
prediction_DL2 = y_pred_DL2.to_csv('prediction_DL2.csv')



##LSTM


import tensorflow as tf
import tensorflow

#from tensorflow import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPool1D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Bidirectional

# fix random seed for reproducibility
MAX_SEQUENCE_LENGTH = 100
MAX_NB_WORDS = 3000
EMBEDDING_DIM = 160
# MAX_NB_WORDS = 500
# max_review_length = 500
# EMBEDDING_DIM = 160
tokenizer = Tokenizer(num_words = MAX_NB_WORDS,
                      filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~',
                      lower=True, split=' ', char_level=False,
                      oov_token=None)
tokenizer.fit_on_texts(df_train['review_clean'].values.astype('U'))
train_sequences = tokenizer.texts_to_sequences(df_train['review_clean'].values.astype('U'))
test_sequences = tokenizer.texts_to_sequences(df_test['review_clean'])
X_train = sequence.pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH)
X_test = sequence.pad_sequences(test_sequences, maxlen = MAX_SEQUENCE_LENGTH)
word_index = tokenizer.word_index
print(len(tokenizer.word_index))


nb_words  = min(MAX_NB_WORDS, len(word_index))
lstm_out = MAX_SEQUENCE_LENGTH

model = Sequential()
model.add(Embedding(nb_words,EMBEDDING_DIM,input_length=MAX_SEQUENCE_LENGTH))
model.add(LSTM(50))
#model.add(Attention(MAX_SEQUENCE_LENGTH))
model.add(Dense(3, activation = 'softmax'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


# Split Training & Validation Data
from sklearn.model_selection import train_test_split


print('creating train and validation data by dividing train_data in 80:20 ratio')
######################################################

X_train_t, X_train_val, Y_train_t, y_train_val = train_test_split(X_train, y_train, test_size = 0.15)

######################################################
print('train data shape:', X_train_t.shape)
print('validation data shape:', X_train_val.shape)
print('Data is ready for training!!')



from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Run LSTM Model
batch = 32
epoch = 30

## set name for the mdoel
training_cycle = 1
notebookname = "Drug_Data_"
variant = "LSTM_w_stopwords_"
version = "1.0_"
title = notebookname + variant + version

stamp = '{}training_cycle{}batchsize_{}'.format(title,training_cycle,batch)
print(stamp)

## save the best model
best_model_path = title + stamp + 'best.h5'
model_checkpoint = ModelCheckpoint(best_model_path, save_best_only = True) ## save only best model

## if 4 steps without decreasing of loss in valid set, stop the trainning
early_stopping = EarlyStopping(patience = 4)

LSTM_model = model.fit(X_train_t, Y_train_t, batch_size=batch, epochs=epoch,
                       validation_data=(X_train_val, y_train_val),callbacks=[model_checkpoint], shuffle = True)

best_score = min(LSTM_model.history['val_loss'])


y_pred_lstm = model.predict(X_test)
y_pred_lstm = (y_pred_lstm > 0.5)

acc_lstm = accuracy_score(y_test,y_pred_lstm)
print(acc_lstm)

y_pred_lstm=pd.DataFrame(y_pred_lstm)
y_pred_lstm=y_pred_lstm.idxmax(axis=1)
prediction_lstm = y_pred_lstm.to_csv('prediction_lstm.csv')


##CNN

import random
import gc
import re
import torch

#import spacy
from tqdm import tqdm_notebook, tnrange
from tqdm.auto import tqdm

tqdm.pandas(desc='Progress')
from collections import Counter

from nltk import word_tokenize

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from sklearn.metrics import f1_score
import os

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# cross validation and metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from torch.optim.optimizer import Optimizer

from sklearn.preprocessing import StandardScaler
from multiprocessing import  Pool
from functools import partial
import numpy as np
from sklearn.decomposition import PCA
import torch as t
import torch.nn as nn
import torch.nn.functional as F



embed_size = 300 # how big is each word vector
max_features = 120000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 750 # max number of words in a question to use
batch_size = 512 # how many samples to process at once
n_epochs = 5 # how many times to iterate over all samples
n_splits = 5 # Number of K-fold Splits
SEED = 10
debug = 0


## Tokenize the sentences
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(df_train['review_clean'].values.astype('U'))
train_X = tokenizer.texts_to_sequences(df_train['review_clean'].values.astype('U'))
test_X = tokenizer.texts_to_sequences(df_test['review_clean'].values.astype('U'))

## Pad the sentences
train_X = pad_sequences(train_X, maxlen=maxlen)
test_X = pad_sequences(test_X, maxlen=maxlen)


class CNN_Text(nn.Module):

    def __init__(self):
        super(CNN_Text, self).__init__()
        filter_sizes = [1, 2, 3, 5]
        num_filters = 36
        n_classes = 3
        self.embedding = nn.Embedding(max_features, embed_size)
        self.convs1 = nn.ModuleList([nn.Conv2d(1, num_filters, (K, embed_size)) for K in filter_sizes])
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(len(filter_sizes) * num_filters, n_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        logit = self.fc1(x)
        return logit


n_epochs = 6
model = CNN_Text()
loss_fn = nn.CrossEntropyLoss(reduction='sum')
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
model.cuda()

# Load train and test in CUDA Memory
x_train = torch.tensor(train_X, dtype=torch.long).cuda()
y_train = torch.tensor(df_train['Review_Sentiment'], dtype=torch.long).cuda()
x_cv = torch.tensor(test_X, dtype=torch.long).cuda()
y_cv = torch.tensor(df_test['Review_Sentiment'], dtype=torch.long).cuda()

# Create Torch datasets
train = torch.utils.data.TensorDataset(x_train, y_train)
valid = torch.utils.data.TensorDataset(x_cv, y_cv)

# Create Data Loaders
train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)

train_loss = []
valid_loss = []

for epoch in range(n_epochs):
    start_time = time.time()
    # Set model to train configuration
    model.train()
    avg_loss = 0.
    for i, (x_batch, y_batch) in enumerate(train_loader):
        # Predict/Forward Pass
        y_pred = model(x_batch)
        # Compute loss
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss.item() / len(train_loader)

    # Set model to validation configuration -Doesn't get trained here
    model.eval()
    avg_val_loss = 0.
    val_preds = np.zeros((len(x_cv), 3))

    for i, (x_batch, y_batch) in enumerate(valid_loader):
        y_pred = model(x_batch).detach()
        avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)
        # keep/store predictions
        val_preds[i * batch_size:(i + 1) * batch_size] = F.softmax(y_pred).cpu().numpy()

    # Check Accuracy
    val_accuracy = sum(val_preds.argmax(axis=1) == df_test['Review_Sentiment']) / len(df_test['Review_Sentiment'])
    train_loss.append(avg_loss)
    valid_loss.append(avg_val_loss)
    elapsed_time = time.time() - start_time
    print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f}  \t val_acc={:.4f}  \t time={:.2f}s'.format(
        epoch + 1, n_epochs, avg_loss, avg_val_loss, val_accuracy, elapsed_time))



def plot_graph(epochs):
    fig = plt.figure(figsize=(12,12))
    plt.title("Train/Validation Loss")
    plt.plot(list(np.arange(epochs) + 1) , train_loss, label='train')
    plt.plot(list(np.arange(epochs) + 1), valid_loss, label='validation')
    plt.xlabel('num_epochs', fontsize=12)
    plt.ylabel('loss', fontsize=12)
    plt.legend(loc='best')


plot_graph(n_epochs)

classes = [0,1,2]
y_pred_cnn = [classes[x] for x in val_preds.argmax(axis=1)]
y_pred_cnn = [classes[x] for x in val_preds.argmax(axis=1)]

y_pred_cnn = [classes[x] for x in val_preds.argmax(axis=1)]


#Harvard emotional dictionary

harvardDict = pd.read_excel('drive/My Drive/drugsCom_raw/inquirerbasic.xls')
harvardDict

harvardDict["Entry"] = harvardDict["Entry"].astype(str)


positiveWords = []
for i in range(0,len(harvardDict.Positiv)):
    if harvardDict.iloc[i,2] == "Positiv":
        temp = harvardDict.iloc[i,0].lower()
        temp1 = re.sub('\d+', '', temp)
        temp2 = re.sub('#', '', temp1)
        positiveWords.append(temp2)

positiveWords = list(set(positiveWords))
print("Number of positive words:", len(positiveWords))



negativeWords = []
for i in range(0,len(harvardDict.Positiv)):
    if (harvardDict.iloc[i,3] == "Negativ"):
        temp = harvardDict.iloc[i,0].lower()
        temp1 = re.sub('\d+', '', temp)
        temp2 = re.sub('#', '', temp1)
        negativeWords.append(temp2)

negativeWords = list(set(negativeWords))
print("Number of negative words:", len(negativeWords))



vectorizer1 = CountVectorizer(vocabulary = positiveWords)
content = df_test['review_clean']
X1 = vectorizer1.fit_transform(content)
f1 = pd.DataFrame(X1.toarray())
f1.columns = positiveWords
df_test["Num Positive Words"] = f1.sum(axis=1)

vectorizer2 = CountVectorizer(vocabulary = negativeWords)
content = df_test['review_clean']
X2 = vectorizer2.fit_transform(content)
f2 = pd.DataFrame(X2.toarray())
f2.columns = negativeWords
df_test["Num Negative Words"] = f2.sum(axis=1)

f1
f2


df_test["Positiv Ratio"] = df_test["Num Positive Words"]/(df_test["Num Positive Words"]+df_test["Num Negative Words"])
df_test["Sentiment Harvard list"] = df_test["Positiv Ratio"].apply(lambda x: 2 if (x>=0.5) else (0 if (x<0.5) else 1))
df_test.head()

df_test.shape

def userful_count(data):
    grouped = data.groupby(['condition']).size().reset_index(name='user_size')
    data = pd.merge(data, grouped, on='condition', how='left')
    return data

df_test =  userful_count(df_test)
df_test['usefulCount'] = df_test['usefulCount']/df_test['user_size']


all_preds= pd.read_csv('/content/final_collected_preds.csv')
print(all_preds.columns)
dict_all_models_pred =[all_preds['NB'],all_preds['LR'],all_preds['DL1'],all_preds['DL2'],all_preds['LSTM']]


# order:  NB,Lr,LGBM,DL1, DL2, LSTM,Harvard
def voting():
    ML_acc_list = [0.57949, 0.663]
    DL_acc_list = [0.6370, 0.7449, 0.84]
    final = [0] * 39108
    total = [0] * 39108
    i = 0

    for i in range(len(ML_acc_list)):
        inter = (ML_acc_list[i] / sum(ML_acc_list)) * dict_all_models_pred[i]
        final = [sum(x) for x in zip(inter, final)]
        i = i + 1

    final_nn = [0] * 39108
    for i in range(len(DL_acc_list)):
        inter_nn = (DL_acc_list[i] / sum(DL_acc_list)) * dict_all_models_pred[i]
        final_nn = [x + y for x, y in zip(inter_nn, final_nn)]
        i = i + 1

    total = [sum(x) for x in zip(final, final_nn, all_preds['LGBM'], all_preds['Sentiment Harvard list'])]

    total = total * df_test['usefulCount']
    return total



arr = voting()
df_test['total_pred']= arr
df_test = df_test.groupby(['condition','drugName']).agg({'total_pred' : ['mean']})
df_test=df_test.reset_index()
df_test


df_test.columns = df_test.columns.map('_'.join)
df_test = df_test.reset_index()
df_test.columns

df_test=df_test.sort_values(['condition_','total_pred_mean'],ascending=False).groupby('condition_').head(6)
df_test


df_test.to_csv('final_prediction_values.csv')