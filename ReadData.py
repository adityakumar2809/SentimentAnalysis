import os
import pathlib


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


def main():
    pwd_path = pathlib.Path(__file__).parent.absolute()
    dir_path = str(pwd_path) + '/data/review_data/train/pos'
    
    print('\n\nUsing File Method:')
    corpus = readDataFileMethod(dir_path)
    print(corpus)
    for i, x in enumerate(corpus):
        print(f'For File index {i}, the corpus length is {len(x)}')
    

if __name__ == '__main__':
    main()
