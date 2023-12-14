#!/usr/bin/env python

import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score


def main():
    data = "data/promoters.data.txt"
    names = ['Class', 'id', 'Sequence']
    data = pd.read_csv(data, names = names)

    classes = data.loc[:, 'Class']

    sequences = list(data.loc[:, 'Sequence'])
    dataset = {}

    # loop through sequences and split into individual nucleotides
    for i, seq in enumerate(sequences):
        # split into nucleotides, remove tab characters
        nucleotides = list(seq)
        nucleotides = [x for x in nucleotides if x != '\t']
        # append class assignment
        nucleotides.append(classes[i])
        # add to dataset
        dataset[i] = nucleotides
        
    dframe = pd.DataFrame(dataset)
    df = dframe.transpose()
    df.rename(columns = {57: 'Class'}, inplace = True)

    series = []
    for name in df.columns:
        series.append(df[name].value_counts())
        
    info = pd.DataFrame(series)
    details = info.transpose()
    #print(details)

    numerical_df = pd.get_dummies(df)
    #print(numerical_df.iloc[:5])

    df = numerical_df.drop(columns=['Class_-'])

    df.rename(columns = {'Class_+': 'Class'}, inplace = True)
    #print(df.iloc[:5])

    X = np.array(df.drop(['Class'], axis=1))
    print(X)
    y = np.array(df['Class'])
    seed = 1
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=seed)

    scoring='accuracy'

    model = SVC(kernel = 'linear')
    name = 'SVM Linear'
    #results = []
    #names = []

    # kfold = model_selection.KFold(n_splits=10, random_state = seed)
    kfold = model_selection.KFold(n_splits=10)
    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    #results.append(cv_results)
    #names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print('Test-- ',name,': ',accuracy_score(y_test, predictions))
    print()
    print(classification_report(y_test, predictions))


if __name__ == '__main__':
    main()