import numpy as np
from FeatureVectorization import *
from ReadData import readDataFileMethod


def getCorpus():
    pwd_path = pathlib.Path(__file__).parent.absolute()

    dir_path = str(pwd_path) + '/data/review_data/train/sample/pos'
    corpus_pos = readDataFileMethod(dir_path)
    pos_length = len(corpus_pos)
    print('Length of POS:', pos_length)

    
    dir_path = str(pwd_path) + '/data/review_data/train/sample/neg'
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

    # freq_list = []
    # for i in range(vectorized_data.shape[0]):
    #     non_zero_columns = (vectorized_data[i, :].nonzero()[1])
    #     for j in non_zero_columns:
    #         freq_list.append([i, j, vectorized_data[i, j]])

    # saveDataAsCSV(
    #     freq_list,
    #     'result/vectorized_data/final/CountVectorized.csv',
    #     columns=['Row', 'Column', 'Frequency']
    # )

    # saveDataAsCSV(
    #     vectorized_data.toarray(),
    #     'result/vectorized_data/final/CountVectorized.csv',
    #     columns=vectorizer.get_feature_names()
    # )

    filepath = 'result/vectorized_data/sample/CountVectorizedData.pkl'
    saveDataInPickleFile(vectorized_data, filepath)
    filepath = 'result/vectorizer/sample/CountVectorizer.pkl'
    saveDataInPickleFile(vectorizer, filepath)


    vectorizer, vectorized_data = fitTransformTFIDFVectorizer(corpus)

    # freq_list = []
    # for i in range(vectorized_data.shape[0]):
    #     non_zero_columns = (vectorized_data[i, :].nonzero()[1])
    #     for j in non_zero_columns:
    #         freq_list.append([i, j, vectorized_data[i, j]])

    # saveDataAsCSV(
    #     freq_list,
    #     'result/vectorized_data/final/TFIDFVectorized.csv',
    #     columns=['Row', 'Column', 'Frequency']
    # )

    # saveDataAsCSV(
    #     vectorized_data.toarray(),
    #     'result/vectorized_data/final/TFIDFVectorized.csv',
    #     columns=vectorizer.get_feature_names()
    # )
    filepath = 'result/vectorized_data/sample/TFIDFVectorizedData.pkl'
    saveDataInPickleFile(vectorized_data, filepath)
    filepath = 'result/vectorizer/sample/TFIDFVectorizer.pkl'
    saveDataInPickleFile(vectorizer, filepath)


    filepath = 'result/vectorized_data/sample/CountVectorizedData.pkl'
    vectorized_data = retrieveDataFromPickleFile(filepath)
    print(vectorized_data)


def main():
    print('Initiating process...')
    corpus = getCorpus()
    saveVectorizerAndVectorizedData(corpus)
    

if __name__ == '__main__':
    main()