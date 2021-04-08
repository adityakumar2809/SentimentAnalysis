from FeatureVectorization import *
from ReadData import readDataFileMethod


def getCorpus():
    pwd_path = pathlib.Path(__file__).parent.absolute()

    dir_path = str(pwd_path) + '/data/review_data/train/pos'
    corpus_pos = readDataFileMethod(dir_path)
    pos_length = len(corpus_pos)

    
    dir_path = str(pwd_path) + '/data/review_data/train/neg'
    corpus_neg = readDataFileMethod(dir_path)
    neg_length = len(corpus_neg)

    corpus = corpus_pos.extend(corpus_neg)

    return corpus


def saveVectorizerAndVectorizedData(corpus):
    vectorizer, vectorized_data = fitTransformCountVectorizer(corpus)
    saveDataAsCSV(
        vectorized_data.toarray(),
        'result/vectorized_data/final/CountVectorized.csv',
        columns=vectorizer.get_feature_names()
    )
    filepath = 'result/vectorizer/final/CountVectorizer.pkl'
    saveVectorizer(vectorizer, filepath)

    vectorizer, vectorized_data = fitTransformTFIDFVectorizer(corpus)
    saveDataAsCSV(
        vectorized_data.toarray(),
        'result/vectorized_data/final/TFIDFVectorized.csv',
        columns=vectorizer.get_feature_names()
    )
    filepath = 'result/vectorizer/final/TFIDFVectorizer.pkl'
    saveVectorizer(vectorizer, filepath)


def main():
    corpus = getCorpus()
    saveVectorizerAndVectorizedData(corpus)
    

if __name__ == '__main__':
    main()