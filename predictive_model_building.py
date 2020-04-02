import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import pickle
from os import path
import os

def preprocess(positive, negative):

    target_ds = pd.read_csv(positive).assign(target=1)
    non_target_ds = pd.read_csv(negative).assign(target=0)

    # target_ds = pd.read_csv('raise_up_both_hands_data.csv').assign(target=1)
    # non_target_ds = pd.read_csv('random_pose_data.csv').assign(target=0)

    ds = target_ds.append(non_target_ds, ignore_index=True)

    return ds


def model_building(ds):

    # Feature Engineering-----------------
    X = ds.iloc[:, 0:51].values
    y = ds.iloc[:, 51].values

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

    # Fitting Random Forest Classification to the Training set
    from sklearn.ensemble import RandomForestClassifier
    RFC = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
    RFC.fit(X_train, y_train)

    os.mkdir('saved_model')
    with open('saved_model/model.txt', 'wb') as f:
        pickle.dump(RFC, f)


    ''' For testing purpose'''
    # # Fitting Models to the training set
    # classifier = []
    # # Fitting Logistic Regression to the Training set with weighting
    # from sklearn.linear_model import LogisticRegression
    # classifier.append(LogisticRegression(random_state=0))
    # classifier[0].fit(X_train, y_train)
    #
    # import matplotlib.pyplot as plt
    # from sklearn.metrics import roc_auc_score
    # from sklearn.metrics import roc_curve
    # logit_roc_auc = roc_auc_score(y_test, classifier[0].predict(X_test))
    # fpr, tpr, thresholds = roc_curve(y_test, classifier[0].predict_proba(X_test)[:, 1])
    # plt.figure()
    # plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    # plt.plot([0, 1], [0, 1], 'r--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic')
    # plt.legend(loc="lower right")
    # plt.savefig('Log_ROC')
    # plt.show()
    #
    # # Fitting Decision Tree Classification to the Training set
    # from sklearn.tree import DecisionTreeClassifier
    # classifier.append(DecisionTreeClassifier(criterion='entropy', random_state=0))
    # classifier[1].fit(X_train, y_train)
    #
    # # %%
    # from sklearn.metrics import roc_auc_score
    # from sklearn.metrics import roc_curve
    # logit_roc_auc = roc_auc_score(y_test, classifier[1].predict(X_test))
    # fpr, tpr, thresholds = roc_curve(y_test, classifier[1].predict_proba(X_test)[:, 1])
    # plt.figure()
    # plt.plot(fpr, tpr, label='Decision Tree Classification (area = %0.2f)' % logit_roc_auc)
    # plt.plot([0, 1], [0, 1], 'r--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic')
    # plt.legend(loc="lower right")
    # plt.savefig('Log_ROC')
    # plt.show()
    #
    # # %%
    # # Fitting Naive Bayes to the Training set
    # from sklearn.naive_bayes import GaussianNB
    # classifier.append(GaussianNB())
    # classifier[2].fit(X_train, y_train)
    #
    # # %%
    # from sklearn.metrics import roc_auc_score
    # from sklearn.metrics import roc_curve
    # logit_roc_auc = roc_auc_score(y_test, classifier[2].predict(X_test))
    # fpr, tpr, thresholds = roc_curve(y_test, classifier[2].predict_proba(X_test)[:, 1])
    # plt.figure()
    # plt.plot(fpr, tpr, label='Naive Bayes Classification (area = %0.2f)' % logit_roc_auc)
    # plt.plot([0, 1], [0, 1], 'r--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic')
    # plt.legend(loc="lower right")
    # plt.savefig('Log_ROC')
    # plt.show()
    # # %%
    # # Fitting K-NN to the Training set
    # from sklearn.neighbors import KNeighborsClassifier
    # classifier.append(KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2))
    # classifier[3].fit(X_train, y_train)
    #
    # # %%
    # from sklearn.metrics import roc_auc_score
    # from sklearn.metrics import roc_curve
    # logit_roc_auc = roc_auc_score(y_test, classifier[3].predict(X_test))
    # fpr, tpr, thresholds = roc_curve(y_test, classifier[3].predict_proba(X_test)[:, 1])
    # plt.figure()
    # plt.plot(fpr, tpr, label='K Nearest Neighbours Classification (area = %0.2f)' % logit_roc_auc)
    # plt.plot([0, 1], [0, 1], 'r--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic')
    # plt.legend(loc="lower right")
    # plt.savefig('Log_ROC')
    # plt.show()
    # # %%
    # # Fitting SVM to the Training set
    # from sklearn.svm import SVC
    # classifier.append(SVC(kernel='linear', random_state=0, probability=True))
    # classifier[4].fit(X_train, y_train)
    #
    # # %%
    # from sklearn.metrics import roc_auc_score
    # from sklearn.metrics import roc_curve
    # logit_roc_auc = roc_auc_score(y_test, classifier[4].predict(X_test))
    # fpr, tpr, thresholds = roc_curve(y_test, classifier[4].predict_proba(X_test)[:, 1])
    # plt.figure()
    # plt.plot(fpr, tpr, label='Support Vector Machine (area = %0.2f)' % logit_roc_auc)
    # plt.plot([0, 1], [0, 1], 'r--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic')
    # plt.legend(loc="lower right")
    # plt.savefig('Log_ROC')
    # plt.show()
    #
    # # %%
    # # Fitting Kernel SVM to the Training set
    # from sklearn.svm import SVC
    # classifier.append(SVC(kernel='rbf', random_state=0, probability=True))
    # classifier[5].fit(X_train, y_train)
    #
    # # %%
    # from sklearn.metrics import roc_auc_score
    # from sklearn.metrics import roc_curve
    # logit_roc_auc = roc_auc_score(y_test, classifier[5].predict(X_test))
    # fpr, tpr, thresholds = roc_curve(y_test, classifier[5].predict_proba(X_test)[:, 1])
    # plt.figure()
    # plt.plot(fpr, tpr, label='Kernel SVM (area = %0.2f)' % logit_roc_auc)
    # plt.plot([0, 1], [0, 1], 'r--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic')
    # plt.legend(loc="lower right")
    # plt.savefig('Log_ROC')
    # plt.show()
    #
    # # %%
    # # Fitting Random Forest Classification to the Training set
    # from sklearn.ensemble import RandomForestClassifier
    # classifier.append(RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0))
    # classifier[6].fit(X_train, y_train)
    #
    # # %%
    # from sklearn.metrics import roc_auc_score
    # from sklearn.metrics import roc_curve
    # logit_roc_auc = roc_auc_score(y_test, classifier[6].predict(X_test))
    # fpr, tpr, thresholds = roc_curve(y_test, classifier[6].predict_proba(X_test)[:, 1])
    # plt.figure()
    # plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    # plt.plot([0, 1], [0, 1], 'r--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic')
    # plt.legend(loc="lower right")
    # plt.savefig('Log_ROC')
    # plt.show()
    #
    # y_pred = []
    # y_pred.append(classifier[0].predict(X_test))
    # y_pred.append(classifier[1].predict_proba(X_test)[:, 1]>0.5)
    # y_pred.append(classifier[2].predict(X_test))
    # y_pred.append(classifier[3].predict(X_test))
    # y_pred.append(classifier[4].predict(X_test))
    # y_pred.append(classifier[5].predict(X_test))
    # y_pred.append(classifier[6].predict_proba(X_test)[:, 1]>0.5)
    #
    # from sklearn.metrics import confusion_matrix
    # cm = []
    # for x in range(0, len(y_pred)):
    #     cm.append(confusion_matrix(y_test, y_pred[x]))
    #
    # # Making classification_report
    # from sklearn.metrics import classification_report
    # for x in range(0, len(y_pred)):
    #     print(x)
    #     print(classification_report(y_test, y_pred[x]))
    #
    # return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predictive model building for classifying customised posture")
    parser.add_argument("positive", help="Path of target dataset")
    parser.add_argument("negative",  help="Path of non-target dataset")
    args = parser.parse_args()

    target, non_target = args.positive, args.negative
    ds = preprocess(target, non_target)
    model_building(ds)

    if path.isfile('saved_model/model.txt'):
        print('model is saved.')













