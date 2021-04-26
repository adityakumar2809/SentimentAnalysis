from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from FeatureVectorization import *


def trainAndSaveModels(model, X_train, y_train):
    '''Train different models on dataset'''
    model_name = str(model)[ : str(model).index('(')]
    model.fit(X_train, y_train)
    path = f'result/models/{model_name}.sav'
    pickle.dump(model, open(path, 'wb'))
    print(f'{model_name} trained and saved successfully!')


def main():    
    filepath = 'result/vectorized_data/train/TFIDFVectorizedData.pkl'
    X_train = retrieveDataFromPickleFile(filepath)

    y_pos = 0.5
    y_train_pos = [1 for x in range(int((X_train.shape[0]) * y_pos))]
    y_train_neg = [0 for x in range(int(X_train.shape[0] - len(y_train_pos)))]
    y_train = y_train_pos + y_train_neg

    model_list = [
        DecisionTreeClassifier(criterion='entropy', random_state=0),
        KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2),
        LogisticRegression(random_state = 0),
        GaussianNB(),
        RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0),
        SVC(kernel='linear', random_state=0)
    ]

    for model in model_list:
        trainAndSaveModels(model, X_train, y_train)


if __name__ == '__main__':
    main()