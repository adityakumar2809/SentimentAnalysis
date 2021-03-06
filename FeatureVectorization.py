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


def saveDataInPickleFile(data, filepath):
    '''Save the data using pickle'''
    pickle.dump(data, open(filepath, 'wb'))


def retrieveDataFromPickleFile(filepath):
    '''Retrieve the saved data using pickle'''
    data = pickle.load(open(filepath, 'rb'))
    return data


def fitTransformCountVectorizer(data):
    '''Vectorize text DataFrame using CountVectorizer'''
    vectorizer = CountVectorizer(analyzer=preprocessText)
    vectorizer_fit = vectorizer.fit(data)
    vectorizer_transform_data = vectorizer_fit.transform(data)

    return vectorizer_fit, vectorizer_transform_data


def fitTransformTFIDFVectorizer(data):
    '''Vectorize text DataFrame using TfidfVectorizer'''
    vectorizer = TfidfVectorizer(analyzer=preprocessText)
    vectorizer_fit = vectorizer.fit(data)
    vectorizer_transform_data = vectorizer_fit.transform(data)

    return vectorizer_fit, vectorizer_transform_data


def transformUsingSavedVectorizer(vectorizer, data):
    '''Vectorize text DataFrame using CountVectorizer'''
    vectorizer_transform_data = vectorizer.transform(data)
    return vectorizer_transform_data


def main():
    pwd_path = pathlib.Path(__file__).parent.absolute()
    
    # FIT TRANSFORM VECTORIZERS
    dir_path = str(pwd_path) + '/data/review_data/train/custom/small'
    corpus = readDataFileMethod(dir_path)

    vectorizer, vectorized_data = fitTransformCountVectorizer(corpus)
    saveDataAsCSV(
        vectorized_data.toarray(),
        'result/vectorized_data/CountVectorized.csv',
        columns=vectorizer.get_feature_names()
    )
    filepath = 'result/vectorizer/CountVectorizer.pkl'
    saveDataInPickleFile(vectorizer, filepath)
    
    vectorizer, vectorized_data = fitTransformTFIDFVectorizer(corpus)
    saveDataAsCSV(
        vectorized_data.toarray(),
        'result/vectorized_data/TFIDFVectorized.csv',
        columns=vectorizer.get_feature_names()
    )
    filepath = 'result/vectorizer/TFIDFVectorizer.pkl'
    saveDataInPickleFile(vectorizer, filepath)


    # TRANSFORM USING SAVED VECTORIZER
    dir_path = str(pwd_path) + '/data/review_data/train/custom/ultra_small'
    corpus = readDataFileMethod(dir_path)
    
    filepath = 'result/vectorizer/CountVectorizer.pkl'
    vectorizer = retrieveDataFromPickleFile(filepath)
    vectorized_data = transformUsingSavedVectorizer(vectorizer, corpus)
    saveDataAsCSV(
        vectorized_data.toarray(),
        'result/vectorized_data/SavedCountVectorized.csv',
        columns=vectorizer.get_feature_names()
    )
    
    filepath = 'result/vectorizer/TFIDFVectorizer.pkl'
    vectorizer = retrieveDataFromPickleFile(filepath)
    vectorized_data = transformUsingSavedVectorizer(vectorizer, corpus)
    saveDataAsCSV(
        vectorized_data.toarray(),
        'result/vectorized_data/SavedTFIDFVectorized.csv',
        columns=vectorizer.get_feature_names()
    )



if __name__ == '__main__':
    main()
