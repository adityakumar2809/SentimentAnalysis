import os
import pathlib
from nltk.corpus.reader.plaintext import PlaintextCorpusReader


def readDataFileMethod(dir_path):
    '''Use trivial file access method to create a corpus'''
    files = os.listdir(dir_path)
    corpus = []
    for filename in files:
        file_object = open(dir_path + '/' + filename, 'r', encoding='utf-8')
        file_data = file_object.read()
        file_object.close()
        corpus.append(file_data)
    return corpus


def readDataPlaintextCorpusReader(dir_path):
    '''Use built-in feature of PlaintextCorpusReader to read corpus'''
    corpus = PlaintextCorpusReader(dir_path, '.*')
    return corpus


def main():
    pwd_path = pathlib.Path(__file__).parent.absolute()
    dir_path = str(pwd_path) + '/data/review_data/train/pos'
    
    # print('\n\nUsing File Method:')
    # corpus = readDataFileMethod(dir_path)
    # print(corpus)
    # for i, x in enumerate(corpus):
    #     print(f'For File index {i}, the corpus length is {len(x)}')
    

    print('\n\nUsing PlaintextCorpusReader:')
    corpus = readDataPlaintextCorpusReader(dir_path)
    print(corpus)
    # files = os.listdir(dir_path)
    # for i, filename in enumerate(files):
    #     print(f'For File index {i}, the corpus length is {len(corpus.words(filename))}')
    


if __name__ == '__main__':
    main()
