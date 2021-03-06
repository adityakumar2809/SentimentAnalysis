import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, \
                            recall_score, precision_score, roc_auc_score

def prediction(classifier, X_test, y_test):
    '''Predicting the Test set results'''
    y_pred = classifier.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_acc = roc_auc_score(y_test, y_pred)
    
    return [tn, fp, fn, tp, acc, precision, recall, f1, roc_acc]


def main():
    with open('result/vectorized_data/test/TFIDFVectorizedData.pkl', 'rb') as f:
        X_test = pickle.load(f)
    
    y_pos = 0.5
    y_test_pos = [1 for x in range(int((X_test.shape[0]) * y_pos))]
    y_test_neg = [0 for x in range(int(X_test.shape[0] - len(y_test_pos)))]
    y_test = y_test_pos + y_test_neg

    X_test, y_test = shuffle(X_test, y_test)
    print('Testing Set Made Successfully')

    model_list=[]
    test_results = []
    model_files = os.listdir('result/models')
    for model_file in model_files:
        with open(f'result/models/{model_file}', 'rb') as f:
            model = pickle.load(f)
            model_list.append(model)
            test_results.append([str(model)[ : str(model).index('(')]])
    
    for index, classifier in enumerate(model_list):
        scores = prediction(classifier, X_test, y_test)
        test_results[index] = test_results[index] + scores

    test_results = pd.DataFrame(test_results, columns=[
        'Classifier',
        'True Negative',
        'False Positive',
        'False Negative',
        'True Positive',
        'Accuracy',
        'Precision',
        'Recall',
        'F1 Score',
        'ROC'
    ])
    test_results.to_csv('result/model_evaluation/ModelEvaluationScores.csv')

    # Graph the results
    test_results = pd.read_csv('result/model_evaluation/ModelEvaluationScores.csv')
    
    classifier_abbr = []
    for classifier_name in test_results['Classifier']:
        classifier_abbr.append(('').join([
            char for char in classifier_name if char.isupper()
        ]))
    
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(classifier_abbr, test_results['Accuracy'], label='Accuracy')
    plt.plot(classifier_abbr, test_results['Precision'], label='Precision')
    plt.plot(classifier_abbr, test_results['Recall'], label='Recall')
    plt.plot(classifier_abbr, test_results['F1 Score'], label='F1 Score')
    plt.plot(classifier_abbr, test_results['ROC'], label='ROC Score')
    plt.xlabel('Classifier Used')
    plt.ylabel('Evaluation Score')
    plt.legend()

    bar_width = 0.20
    bar_tn = np.arange(len(classifier_abbr))
    bar_fp = [x + bar_width for x in bar_tn]
    bar_fn = [x + bar_width for x in bar_fp]
    bar_tp = [x + bar_width for x in bar_fn]
    plt.subplot(1, 2, 2)
    plt.bar(bar_tn, test_results['True Negative'], width=bar_width, color='r', label='TN')
    plt.bar(bar_fp, test_results['False Positive'], width=bar_width, color='b', label='FP')
    plt.bar(bar_fn, test_results['False Negative'], width=bar_width, color='y', label='FN')
    plt.bar(bar_tp, test_results['True Positive'], width=bar_width, color='g', label='TP')
    plt.xlabel('Classifier Used')
    plt.ylabel('Number of Samples')
    plt.xticks(
        [r + bar_width for r in range(len(classifier_abbr))],
        classifier_abbr
    )
    plt.legend()
    plt.show()


def analyzeReview(test_review):
    '''Classify the given review as positive or negative'''
    with open('result/vectorizer/train/TFIDFVectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
        X_test = vectorizer.transform([test_review])

    model_list=[]
    test_results = []
    model_files = os.listdir('result/models')
    for model_file in model_files:
        with open(f'result/models/{model_file}', 'rb') as f:
            model = pickle.load(f)
            model_list.append(model)
            test_results.append([str(model)[ : str(model).index('(')]])
    
    for index, classifier in enumerate(model_list):
        y_pred = classifier.predict(X_test)
        y_pred = 'Positive' if y_pred == 1 else 'Negative'
        test_results[index].append(y_pred)
    
    print('\n\nResults :-')
    for test_result in test_results:
        print(test_result[0], ':', test_result[1])
    print('\n\n')
    


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    choice = input('Do you wish to check your own review? (Y/N)')
    if choice in ['N', 'n']:
        main()
    else:
        test_review = input('Enter your review: ')
        analyzeReview(test_review)