# Using trained doc2vec model


import pandas as pd
import numpy as np
from gensim.models.doc2vec import Doc2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

def run_cross_validation_kfold(clf, X_cv, y_cv, n_fold=5):
    kf = KFold(n_splits=n_fold)
    roc_auc_scores = []
    fold = 0
    for train_index, test_index in kf.split(X_cv):
        fold += 1
        X_train, X_test = X_cv[train_index], X_cv[test_index]
        y_train, y_test = y_cv[train_index], y_cv[test_index]
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        area = roc_auc_score(y_test, predictions)
        roc_auc_scores.append(area)
        print("Fold {0} area: {1}".format(fold, area))
    mean_area = np.mean(roc_auc_scores)
    print("Mean area: {0}".format(mean_area))

if __name__ == '__main__':
    train = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
    test = pd.read_csv("testData.tsv", header=0, delimiter="\t", quoting=3)

    PvDmModel = Doc2Vec.load("pv_dm_300features_15minwords_10context.d2vmodel")   

    trainVectors = []
    for review_id in train["id"]:
        recovered_review = PvDmModel.docvecs[review_id]
        trainVectors.append(recovered_review)

    X = np.asarray(trainVectors)

    testVectors = []
    for review_id in test["id"]:
        recovered_review = PvDmModel.docvecs[review_id]
        testVectors.append(recovered_review)       

    test_2Darray = np.asarray(testVectors)
    y = np.asarray(train["sentiment"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    print("-------------Run logistic regression---------------")
    logReg = LogisticRegression(solver='lbfgs', max_iter=450, n_jobs=-1, verbose=1)
    logReg.fit(X_train, y_train)
    run_cross_validation_kfold(logReg, X_train, y_train)
    y_predicted_lr = logReg.predict(X_test)
    print("logistic regression roc_auc_score ", roc_auc_score(y_test, y_predicted_lr))
    logRegres_result = logReg.predict(test_2Darray)

    outputLR = pd.DataFrame(data={"id": test["id"], "sentiment": logRegres_result})
    outputLR.to_csv("Doc2vec_LogR.csv", index=False, quoting=3)

    print("-------------Run RandomForestClassifier---------------")
    forest = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    forest.fit(X_train, y_train)

    run_cross_validation_kfold(forest, X_train, y_train)
    y_predicted = forest.predict(X_test)
    print("RandomForestClassifier roc_auc_score ", roc_auc_score(y_test, y_predicted))
    forest_clf_result = forest.predict(test_2Darray)


    output = pd.DataFrame(data={"id": test["id"], "sentiment": forest_clf_result})
    output.to_csv("Doc2vec_RFC.csv", index=False, quoting=3)