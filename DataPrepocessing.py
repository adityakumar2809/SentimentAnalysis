import string
import nltk
import re

nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


def normalizeText(text):
    '''Normalize the input text'''
    text = text.lower() # Convert to lowercase
    text = re.sub(r'\d+', '', text) # Remove numbers
    text = text.translate(
        text.maketrans('', '', string.punctuation)
    ) # Remove Punctuations
    text = text.strip()
    return text


def tokenizeText(text):
    '''Split normalized text into tokens'''
    text_tokens = word_tokenize(text)
    return text_tokens


def removeStopwords(text_list):
    '''Remove all stopwords in a piece of tokenized text'''
    clean_text_list = [
        x for x in text_list if x not in stopwords.words('english')
    ]
    return clean_text_list


def stemText(token_list):
    '''Perform stemming on tokenized text'''
    stemmer = PorterStemmer()
    stemmed_token_list = []
    for token in token_list:
        stemmed_token_list.append(stemmer.stem(token))
    return stemmed_token_list


def preprocessText(text):
    '''Apply preprocessing on text'''
    text = normalizeText(text)
    text = tokenizeText(text)
    text = removeStopwords(text)
    text = stemText(text)
    text.sort()
    return text


def main():
    text = 'There are 6754 several,  types of stemming algorithms to stem words in stemmed stuff.'
    print(f'Original Text: {text}\n')
    text = normalizeText(text)
    print(f'Normalized Text: {text}\n')
    tokenized_text_list = tokenizeText(text)
    print(f'Tokenized Text: {tokenized_text_list}\n')
    tokenized_text_list = removeStopwords(tokenized_text_list)
    print(f'Text without Stopwords: {tokenized_text_list}\n')
    stem_text_list = stemText(tokenized_text_list)
    print(f'Stemmed Text: {stem_text_list}\n')    


if __name__ == '__main__':
    main()