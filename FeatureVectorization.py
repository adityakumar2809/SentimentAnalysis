import math
import pickle
import pathlib
import collections
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from DataPrepocessing import preprocessText
from ReadData import readDataFileMethod


def saveDataAsCSV(vectorized_data, filepath, columns=None):
    df = pd.DataFrame(vectorized_data, columns=columns)
    df.to_csv(filepath)


def saveVectorizer(vectorizer, filepath):
    '''Save the vectorizer object using pickle'''
    pickle.dump(vectorizer, open(filepath, 'wb'))


def countVectorizeTextData(data):
    '''Vectorize text DataFrame using CountVectorizer'''
    vectorizer = CountVectorizer(analyzer=preprocessText)
    vectorizer_fit = vectorizer.fit(data)
    vectorizer_transform_data = vectorizer_fit.transform(data)

    return vectorizer_fit, vectorizer_transform_data


def tfidfVectorizeTextData(data):
    '''Vectorize text DataFrame using TfidfVectorizer'''
    vectorizer = TfidfVectorizer(analyzer=preprocessText)
    vectorizer_fit = vectorizer.fit(data)
    vectorizer_transform_data = vectorizer_fit.transform(data)

    return vectorizer_fit, vectorizer_transform_data


def main():
    pwd_path = pathlib.Path(__file__).parent.absolute()
    dir_path = str(pwd_path) + '/data/review_data/train/small'
    corpus = readDataFileMethod(dir_path)

    vectorizer, vectorized_data = countVectorizeTextData(corpus)
    saveDataAsCSV(
        vectorized_data.toarray(),
        'result/vectorized_data/CountVectorized.csv',
        columns=vectorizer.get_feature_names()
    )
    filepath = 'result/vectorizer/CountVectorizer.pkl'
    saveVectorizer(vectorizer, filepath)
    

    vectorizer, vectorized_data = tfidfVectorizeTextData(corpus)
    saveDataAsCSV(
        vectorized_data.toarray(),
        'result/vectorized_data/TFIDFVectorized.csv',
        columns=vectorizer.get_feature_names()
    )
    filepath = 'result/vectorizer/TFIDFVectorizer.pkl'
    saveVectorizer(vectorizer, filepath)


if __name__ == '__main__':
    main()
