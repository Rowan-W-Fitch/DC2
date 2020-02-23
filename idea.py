"""
get top words from spams
get top words from hams
get semetric difference of spam and ham words (union - intersection)

sf = number of words in spam feats (semetric difference w/ ham feats)
hf = number of words in ham feats (semetric difference w/ spam feats)
add sf and hf to csv file
"""

import pandas as pd
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords as stp
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from skrules import SkopeRules
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve

def read_csv():
    #creates panda table
    csv = sys.argv[1]
    mail = pd.read_csv(csv, usecols = [0,1], na_filter = False)
    mail.rename(columns = {'v1':'label', 'v2':'message'}, inplace = True)
    mail.replace("spam", 1, inplace = True)
    mail.replace('ham', 0, inplace = True)
    mail.insert(loc = 2, column = 'sf', value = 0) #add sf value to panda
    mail.insert(loc = 3, column = 'hf', value = 0) #add hf value to panda
    #return panda table
    return mail

def get_features(mail, label):
    list = [] #going to be a list of emails
    for index,row in mail.iterrows():
        if row['label'] == label:
            list.append(row['message'].lower()) #add the email to the list (all lowercase)
    #get random sample of list
    random.shuffle(list)
    cv_list = list[0: int(len(list)/3)]
    cv = CountVectorizer(input = 'content', stop_words = stp.words('english'), ngram_range = (1,2))
    cv.fit(cv_list)
    #return list of features
    return cv.get_feature_names()

def get_feat_scores():
    #get disjoint union of spam and ham features
    mail = read_csv()
    spam_feat = get_features(mail, 1)
    ham_feat = get_features(mail, 0)
    sf = list(set(spam_feat) - (set(spam_feat) and set(ham_feat)))
    hf = list(set(ham_feat) - (set(ham_feat) and set(spam_feat)))
    #update the sf and hf columns in panda sheet
    for index, row in mail.iterrows():
        num_sf, num_hf = 0,0
        txt = row['message'].lower()
        for w in word_tokenize(txt):
            if w in sf:
                num_sf += 1
            elif w in hf:
                num_hf += 1
        mail.at[index, 'sf'] = num_sf + 1
        mail.at[index, 'hf'] = num_hf + 1
    #return updated panda
    return mail

def main():
    mail = get_feat_scores() #panda table
    train, test = train_test_split(mail, test_size=0.3) #split up data
    x_train = train.drop(columns = ['label']) #remove labels from test x
    y_train = train.drop(columns = ['message','sf','hf'])
    cv = CountVectorizer(input = 'content', stop_words = stp.words('english'), ngram_range = (1,2))
    x_tr = cv.fit_transform(x_train.message) #vectorize x_train text for algorithm
    skr = SkopeRules(n_estimators = 30, feature_names = ['sf','hf']) #algorithm
    y_train = y_train.to_numpy().ravel() #turn y_train into a 1d array for algorithm
    y_train = y_train.astype('int')
    skr.fit(x_tr.toarray(),y_train)
    #test data
    x_test = train.drop(columns = ['label'])
    y_test = train.drop(columns = ['message','sf','hf'])
    x_tst = cv.transform(x_test.message)
    y_test = y_test.to_numpy().ravel()
    y_test = y_test.astype('int')
    y_score = skr.score_top_rules(x_tst.toarray())
    #metrics
    recall_scr = recall_score(y_test, y_score, average = 'micro' )
    f1_scr = f1_score(y_test,y_score, average = 'micro')
    pr_score = precision_score(y_test, y_score, average = 'micro')
    print("recall: " + str(recall_scr))
    print("f1: " + str(f1_scr))
    print("precision: " + str(pr_score))
    #plot
    precision, recall, r = precision_recall_curve(y_test, y_score)
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision Recall curve')
    plt.show()

main()
