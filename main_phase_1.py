import numpy as np
from FeatureVectorization import *
from ReadData import readDataFileMethod


def getCorpus():
    pwd_path = pathlib.Path(__file__).parent.absolute()

    dir_path = str(pwd_path) + '/data/review_data/test/pos'
    corpus_pos = readDataFileMethod(dir_path)
    pos_length = len(corpus_pos)
    print('Length of POS:', pos_length)

    
    dir_path = str(pwd_path) + '/data/review_data/test/neg'
    corpus_neg = readDataFileMethod(dir_path)
    neg_length = len(corpus_neg)
    print('Length of NEG:', neg_length)

    corpus = corpus_pos + corpus_neg
    print('Length of Corpus:', len(corpus))

    return corpus


def saveVectorizerAndVectorizedData(corpus):
    vectorizer, vectorized_data = fitTransformCountVectorizer(corpus)

    saveDataAsCSV(
        vectorizer.get_feature_names(),
        'result/vectorized_data/sample/FeatureList.csv',
        columns=['Word']
    )

    filepath = 'result/vectorized_data/sample/CountVectorizedData.pkl'
    saveDataInPickleFile(vectorized_data, filepath)
    filepath = 'result/vectorizer/sample/CountVectorizer.pkl'
    saveDataInPickleFile(vectorizer, filepath)


    vectorizer, vectorized_data = fitTransformTFIDFVectorizer(corpus)
    filepath = 'result/vectorized_data/sample/TFIDFVectorizedData.pkl'
    saveDataInPickleFile(vectorized_data, filepath)
    filepath = 'result/vectorizer/sample/TFIDFVectorizer.pkl'
    saveDataInPickleFile(vectorizer, filepath)


    filepath = 'result/vectorized_data/sample/CountVectorizedData.pkl'
    vectorized_data = retrieveDataFromPickleFile(filepath)


def main():
    print('Initiating process...')
    corpus = getCorpus()

    saveVectorizerAndVectorizedData(corpus)

    filepath = 'result/vectorizer/train/CountVectorizer.pkl'
    vectorizer = retrieveDataFromPickleFile(filepath)
    vectorized_data = transformUsingSavedVectorizer(vectorizer, corpus)
    filepath = 'result/vectorized_data/test/CountVectorizedData.pkl'
    saveDataInPickleFile(vectorized_data, filepath)
    
    filepath = 'result/vectorizer/train/TFIDFVectorizer.pkl'
    vectorizer = retrieveDataFromPickleFile(filepath)
    vectorized_data = transformUsingSavedVectorizer(vectorizer, corpus)
    filepath = 'result/vectorized_data/test/TFIDFVectorizedData.pkl'
    saveDataInPickleFile(vectorized_data, filepath)
    

if __name__ == '__main__':
    main()