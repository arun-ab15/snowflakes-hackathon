from bs4 import BeautifulSoup
from nltk import SnowballStemmer, word_tokenize
from nltk.corpus import stopwords
from snowpark import *
from tqdm.notebook import tqdm as tqdm

import re
import pandas as pd
import nltk

stemmer = SnowballStemmer('english')

def feature_engineering(data):
    # feature engineering
    # let's make a new column review sentiment
    data.loc[(data['rating'] >= 7), 'Review_Sentiment'] = 2
    data.loc[(data['rating'] == 5), 'Review_Sentiment'] = 1
    data.loc[(data['rating'] == 6), 'Review_Sentiment'] = 1
    data.loc[(data['rating'] < 5), 'Review_Sentiment'] = 0
    data['Review_Sentiment'].value_counts()
    print("Created new column sentiment using feature engineering")
    return data


def remove_null(data):
    # Check for null values
    data.isnull().any()
    # we will delete the rows so that the data does not overfits
    data = data.dropna(axis=0)
    data.shape
    print("Removed null values")
    return data


def remove_stop_words(stops):
    # removing some stopwords from the list of stopwords as they are important for drug recommendation
    stops = set(stopwords.words('english'))
    not_stop = ["aren't", "couldn't", "didn't", "doesn't", "don't", "hadn't", "hasn't", "haven't", "isn't", "mightn't",
                "mustn't", "needn't", "no", "nor", "not", "shan't", "shouldn't", "wasn't", "weren't", "wouldn't"]
    for i in not_stop:
        stops.remove(i)

    print("Removed non stop values")
    return stops


def cleanup_medical_condition(df_condition, data):
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
    print("Cleaned up conditions with less than 5 drugs")
    return data


def remove_condition_with_span(data):
    all_list = set(data.index)
    span_list = []
    for i, j in enumerate(data['condition']):
        if '</span>' in j:
            span_list.append(i)
    new_idx = all_list.difference(set(span_list))
    data = data.iloc[list(new_idx)].reset_index()
    del data['index']
    print("Cleaned up conditions with span data")
    return data


def write_preprocessed_data_to_table(data):
    table_name = 'prepared_data'
    conn = create_session()
    data.to_sql(
        name=table_name.lower(),
        con=conn,
        if_exists="replace"
    )
    print("Loaded the table")


def review_to_words(raw_review, stops):
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
    return ' '.join(stemming_words)


def analyse_emotions():
    filepath = "nrcdata/NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt"
    emolex_df = pd.read_csv(filepath, names=["word", "emotion", "association"], skiprows=45, sep='\t',
                            keep_default_na=False)
    emolex_df.head(12)
    emolex_df.emotion.unique()
    emolex_df.emotion.value_counts()
    emolex_df[emolex_df.association == 1].emotion.value_counts()
    emolex_words = emolex_df.pivot(index='word', columns='emotion', values='association').reset_index()
    emolex_words.head()
    print(emolex_words.head())
    return emolex_words


def extract_review_emotion(df, emolex_words, column):
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


def main():
    nltk.download('stopwords')
    stops = set(stopwords.words('english'))
    train_data = get_train_data().to_pandas()
    test_data = get_test_data().to_pandas()
    data = pd.concat([train_data, test_data])
    print(data)
    data = feature_engineering(data)
    data = remove_null(data)
    stops = remove_stop_words(stops)
    df_condition = data.groupby(['condition'])['drugName'].nunique().sort_values(ascending=False)
    df_condition = pd.DataFrame(df_condition).reset_index()
    data = cleanup_medical_condition(df_condition, data)
    data = remove_condition_with_span(data)
    write_preprocessed_data_to_table(data)

    # data['review_clean'] = data['review'].apply(review_to_words(stops))
    # emolex_words = analyse_emotions()
    # emotion_df = extract_review_emotion(data, emolex_words, 'review_clean')
    #
    # emo = emotion_df.groupby(['drugName']).sum()
    # emo.head(10)
    # print(emo.head(10))
    # emo.to_csv('emotion_groupby11.csv', index=True)
    # emotion_df.to_csv('emotion_sentiment.csv', index=False)


if __name__ == '__main__':
    main()
