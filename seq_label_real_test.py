#!/usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from random import randint
from Bio import SeqIO


def read_fasta(fasta):
    return ((record.id, str(record.seq)) for record in SeqIO.parse(open(fasta, 'r'), "fasta"))


def get_kmers(seq, size=6):
    """Create a list of k-mers of `size`.

    This function takes a sequence and converts it into k-mers of a specified
    size (default 6). K-mers are all possible overlapping substrings, of the
    same length, from a genomic 'read'. 

    Parameters:
    seq (string): an input genomic sequence
    size   (int): the length of the k-mers (default 6)

    Returns:
    list: a list of all k-mers from the input seq
    """
    return [seq[x:x+size].lower() for x in range(len(seq) - size + 1)]


def get_words_and_classes(df):
    """
    """
    df['words'] = df.apply(lambda x: get_kmers(x['sequence']), axis=1)
    df = df.drop('sequence', axis=1)
    return df


def sk_data(df):
    """
    """
    df_texts = list(df['words'])

    # flattening each list into a single list
    # each entry being all the k-mers seperated by a space
    for item in range(len(df_texts)):
        df_texts[item] = ' '.join(df_texts[item])
    
    # grabbing the class information from the pd table
    y_data = df.iloc[:, 0].values

    return (df_texts, y_data)


def get_metrics(y_test, y_predicted):
        accuracy = accuracy_score(y_test, y_predicted)
        precision = precision_score(y_test, y_predicted, average='weighted')
        recall = recall_score(y_test, y_predicted, average='weighted')
        f1 = f1_score(y_test, y_predicted, average='weighted')
        return accuracy, precision, recall, f1


def get_colors(number):
    """
    """
    return ['#%06X' % randint(0, 0xFFFFFF) for i in range(number)]
    

def main():
    # loading in the data
    data_dir = 'data/'
    #human_data = pd.read_table(f'{data_dir}human_data.txt')

    fasta = read_fasta(f'{data_dir}megares_modified_database_v2.00.fasta')

    megares = pd.DataFrame(columns=['sequence', 'class'])

    for read in fasta:
        megares.loc[-1] = [read[1], read[0].split('|')[1]]
        megares.index = megares.index + 1
        megares = megares.sort_index()

    print(megares.head())

    #print(human_data.head())

    # Creating a words entry in the pd table from the k-mers and then
    # removing the sequence from the table to save space
    #human_data = get_words_and_classes(human_data)
    megares = get_words_and_classes(megares)

    print(megares.head())

    # print(human_data.head())

    # converting the column into a list of lists containing the k-mers
    #human_texts, y_data = sk_data(human_data)
    megares_texts, y_data = sk_data(megares)

    #print(y_data)
    cv = CountVectorizer(ngram_range=(4,4))
    # X = cv.fit_transform(human_texts)
    X = cv.fit_transform(megares_texts)

    # print(X.shape)

    # Making a plot graph of the class counts
    # human_class_counts = human_data['class'].value_counts().sort_index()
    # plt.bar(
    #     human_class_counts.index,
    #     human_class_counts,
    #     color=get_colors(len(human_class_counts.index)))
    # plt.show()

    megares_class_counts = megares['class'].value_counts().sort_index()
    plt.bar(
        megares_class_counts.index,
        megares_class_counts,
        color=get_colors(len(megares_class_counts.index)))
    plt.show()

    # # # splitting the data into test and training data
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y_data, 
        test_size = 0.20, 
        random_state=42)


    classifier = MultinomialNB(alpha=0.1)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    print("Confusion matrix\n")
    print(pd.crosstab(
        pd.Series(y_test, name='Actual'),
        pd.Series(y_pred, name='Predicted')))
    accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
    print(
        f'accuracy = {accuracy}\n'\
        f'precision = {precision}\n'\
        f'recall = {recall}\n'\
        f'f1 = {f1}')
#

if __name__ == '__main__':
    main()