#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 10:57:31 2019

@author: qwang
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May 24 12:55:56 2019

@author: s1515896
"""

import gc
import os
import csv
import random
import re

import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer #, TfidfVectorizer
from sklearn.metrics import confusion_matrix, make_scorer, fbeta_score, recall_score, accuracy_score, precision_score
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

from argparse import Namespace
import collections

os.chdir('/home/qwang/rob')


#%% ==================================== Setting ====================================
# Set Numpy and PyTorch seeds
def set_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
 
rob_name = 'randomisation'
#rob_name = 'blinded'
#rob_name = 'ssz'
       
# Arguments
args = Namespace(
    seed=1234,
    shuffle=True,
    train_size=0.80,
    val_size=0.10,
    test_size=0.10
)

# Set seeds
set_seeds(seed=args.seed)


#%% ==================================== Data ====================================
csv.field_size_limit(100000000)
#dat = pd.read_csv("datafile/dataWithFullText_utf8.txt", sep='\t', engine="python", encoding="utf-8")   
#dat['text'] = dat['CleanFullText']
#dat['label_random'] = dat['RandomizationTreatmentControl'] 
#dat['label_blind'] = dat['BlindedOutcomeAssessment'] 
#dat['label_ssz'] = dat['SampleSizeCalculation'] 
#dat = dat[-dat["ID"].isin([8, 608, 647, 703, 807, 903, 960, 1446, 1707, 1707, 1714, 1716, 1754, 2994, 
#                           2995, 2996, 2997, 3943, 4045, 4064, 4066, 4076, 4077, 4083, 3804, 4035])]
#dat.set_index(pd.Series(range(0, len(dat))), inplace=True)
#dat.to_csv("datafile/fulldata.csv", sep='\t', encoding='utf-8', index=False)


# Final raw data
df = pd.read_csv("datafile/fulldata.csv", usecols=['text', 'label_random', 'label_blind', 'label_ssz'], sep = '\t', engine = 'python', encoding='utf-8')
df.loc[df.label_random==1, 'label'] = 'random'
df.loc[df.label_random==0, 'label'] = 'non-random'
#df.loc[df.label_blind==1, 'label'] = 'blinded'
#df.loc[df.label_blind==0, 'label'] = 'non-blinded'
#df.loc[df.label_ssz==1, 'label'] = 'ssz'
#df.loc[df.label_ssz==0, 'label'] = 'non-ssz'
df.label.value_counts()

# Split by label
by_label = collections.defaultdict(list)
for _, row in df.iterrows():
    by_label[row.label].append(row.to_dict())
for label in by_label:
    print("{0}: {1}".format(label, len(by_label[label])))
    
# Create split data
final_list = []
for _, item_list in sorted(by_label.items()):
    if args.shuffle:
        np.random.shuffle(item_list)
    n = len(item_list)
    n_train = int(args.train_size*n)
    n_val = int(args.val_size*n)
    n_test = int(args.test_size*n)

    # Give data point a split attribute
    for item in item_list[:n_train]:
        item['split'] = 'train'
    for item in item_list[n_train:n_train+n_val]:
        item['split'] = 'val'
    for item in item_list[n_train+n_val:]:
        item['split'] = 'test'  
    # Add to final list
    final_list.extend(item_list)
   
    
# df with split datasets
split_df = pd.DataFrame(final_list)
split_df["split"].value_counts()

# Preprocessing
def preprocess_text(text):
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = text.encode("ascii", errors="ignore").decode()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r"[!%^&*()=_+{};:$£€@~#|/,.<>?\`\'\"\[\]\\]", " ", text)  # [!%^&*()=_+{};:$£€@~#|/<>?\`\'\"\[\]\\]
    text = re.sub(r'\b(\w{1})\b', '', text) 
    text = re.sub(r'\s+', ' ', text)
    return text.lower()
    
split_df.text = split_df.text.apply(preprocess_text)


#%% Split data
#X_train = split_df.loc[split_df['split']=='train', 'text']
#y_train = split_df.loc[split_df['split']=='train', 'label_random']
#
#X_valid = split_df.loc[split_df['split']=='val', 'text']
#y_valid = split_df.loc[split_df['split']=='val', 'label_random']
#
#X_test = split_df.loc[split_df['split']=='test', 'text']
#y_test = split_df.loc[split_df['split']=='test', 'label_random']




# or...
X_train = split_df.loc[(split_df['split']=='train') | (split_df['split']=='val'), 'text']
y_train = split_df.loc[(split_df['split']=='train') | (split_df['split']=='val'), 'label_random']

X_test = split_df.loc[split_df['split']=='test', 'text']
y_test = split_df.loc[split_df['split']=='test', 'label_random']



# %% Grid search (optimal output) (correct version)
gc.collect()
if __name__ == "__main__":
    
    def spec(y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()      
        return tn / (tn+fp)
    
    recall_scorer = make_scorer(recall_score, greater_is_better=True) 
    acc_scorer = make_scorer(accuracy_score, greater_is_better=True)
    prec_scorer = make_scorer(precision_score, greater_is_better=True) 
    f1_scorer = make_scorer(fbeta_score, beta=1)   
    spec_scorer = make_scorer(spec, greater_is_better=True)
    
    

    scorer = {'Accuracy': acc_scorer,
              'F1': f1_scorer,
              'Recall': recall_scorer,
              'Precision': prec_scorer,
              'Specificity': spec_scorer
             }
          
    pipeline = Pipeline([
        ('vect', CountVectorizer(stop_words='english', lowercase=True, token_pattern=r'\w{1,}')),
        ('tfidf', TfidfTransformer(use_idf=True, sublinear_tf=False)),
        ('sgd', SGDClassifier(random_state=66, max_iter=1000, early_stopping=True, validation_fraction=0.1, tol=0.001)),
    ])   
        
    parameters = {  'vect__min_df': (20,24,28,36,40,44,48,52,56,60),
                    'vect__ngram_range': ((1,1),), #((1,1), (1,2)),   
                    'vect__max_features': (3000,), #(1000,2000,3000,4000,5000,6000,7000,8000,9000,10000),
                    'tfidf__norm': ('l2',), #('l1', 'l2', None),
                    'sgd__alpha': (0.001,), #(0.01, 0.001, 0.0001),
                    'sgd__penalty': ('l2',), #('l1', 'l2', 'elasticnet'),
                 }
    
    clf = GridSearchCV(pipeline, parameters, scoring=scorer, refit='F1', cv=5, n_jobs=6, pre_dispatch=2*6, verbose=1, return_train_score=False)      
    clf.fit(X_train, y_train) 
    
    ### Test ###
#    pred = pd.DataFrame(np.zeros((len(y_test), 1)))        
#    y_true, y_pred = y_test, clf.predict(X_test)
#    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
#    report = pd.DataFrame(np.zeros((1,5)), columns=['test_sens','test_spec','test_acc','test_prec','test_f1'])       
#    report['test_sens'] = tp / (tp+fn)
#    report['test_spec'] = tn / (tn+fp)
#    report['test_acc'] = (tp+tn) / (tp+fp+fn+tn)
#    report['test_prec'] = tp / (tp+fp)
#    report['test_f1'] = 2*tp / (2*tp + fp + fn)  
#    print(report)

grid_report = pd.DataFrame({'pars': clf.cv_results_['params'],
                            'val_sens': clf.cv_results_['mean_test_Recall'],
                            'val_spec': clf.cv_results_['mean_test_Specificity'],
                            'val_acc': clf.cv_results_['mean_test_Accuracy'],
                            'val_prec': clf.cv_results_['mean_test_Precision'],
                            'val_f1': clf.cv_results_['mean_test_F1'],
                            'val_sens_std': clf.cv_results_['std_test_Recall'],
                            'val_spec_std': clf.cv_results_['std_test_Specificity'],
                            'val_acc_std': clf.cv_results_['std_test_Accuracy'],
                            'val_prec_std': clf.cv_results_['std_test_Precision'],
                            'val_f1_std': clf.cv_results_['std_test_F1']} 
    )
grid_report.to_csv('report_sgd_temp.csv')






# Cross-validation report
clf.cv_results_
clf.cv_results_['mean_test_Accuracy']
clf.cv_results_['mean_test_F1']
clf.cv_results_['mean_test_Recall']
clf.cv_results_['mean_test_Precision']
clf.cv_results_['mean_test_Specificity']


clf.cv_results_['params'][0]

clf.cv_results_['mean_test_Accuracy']

clf.cv_results_['params']

clf.best_score_ # output the optimal score of pre-defined 'refit' method
clf.best_params_

grid_report = pd.DataFrame({'pars': clf.cv_results_['params'],
                            'val_sens': clf.cv_results_['mean_test_Recall'],
                            'val_spec': clf.cv_results_['mean_test_Specificity'],
                            'val_acc': clf.cv_results_['mean_test_Accuracy'],
                            'val_prec': clf.cv_results_['mean_test_Precision'],
                            'val_f1': clf.cv_results_['mean_test_F1']})




# %% Normal (non grid search cv)
if __name__ == "__main__":
    
    pars = [1000, 2000]#,2000,3000,4000,5000,6000,7000,8000,9000,10000] # tunning parameters
       
    recall_scorer = make_scorer(recall_score, greater_is_better=True)       
    f1_scorer = make_scorer(fbeta_score, beta=1)
    
    report = pd.DataFrame(np.zeros((len(pars),10)), index=pars, 
                          columns=['val_sens','val_spec','val_acc','val_prec','val_f1','test_sens','test_spec','test_acc','test_prec','test_f1'])   
          
    for ele in pars:            
        ## CountVectorizer    
        count_vect = CountVectorizer(stop_words='english', lowercase=True, token_pattern=r'\w{1,}', 
                                     min_df=2, ngram_range=(1,1), max_features=ele)
        X_train_count = count_vect.fit_transform(X_train)
        X_valid_count = count_vect.fit_transform(X_valid)
        X_test_count = count_vect.fit_transform(X_test)
        
        ## TfidfTransformer  
        tfidf = TfidfTransformer(use_idf=True, norm='l2', sublinear_tf=False)
        X_train_tfidf = tfidf.fit_transform(X_train_count)
        X_valid_tfidf = tfidf.fit_transform(X_valid_count)
        X_test_tfidf = tfidf.fit_transform(X_test_count)
        
        ## SGDClassifier
        sgd = SGDClassifier(random_state=66, alpha=0.0001, penalty='l2')
        sgd.fit(X_train_tfidf, y_train) 
        
        

        ### Validation ###
        y_valid_predict = sgd.predict(X_valid_tfidf)
        pred = pd.DataFrame(np.zeros((len(y_valid_predict), 1)), columns=list([ele]))  
        y_true, y_pred = y_valid, y_valid_predict
        pred.loc[:, ele] = y_pred

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()      
        report.loc[ele, 'val_sens'] = tp / (tp+fn)
        report.loc[ele, 'val_spec'] = tn / (tn+fp)
        report.loc[ele, 'val_acc'] = (tp+tn) / (tp+fp+fn+tn)
        report.loc[ele, 'val_prec'] = tp / (tp+fp)
        report.loc[ele, 'val_f1'] = 2*tp / (2*tp + fp + fn)

        ### Test ###
        y_test_predict = sgd.predict(X_test_tfidf)
        pred = pd.DataFrame(np.zeros((len(y_test_predict), 1)), columns=list([ele]))        
        y_true, y_pred = y_test, y_test_predict
        pred.loc[:, ele] = y_pred

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()      
        report.loc[ele, 'test_sens'] = tp / (tp+fn)
        report.loc[ele, 'test_spec'] = tn / (tn+fp)
        report.loc[ele, 'test_acc'] = (tp+tn) / (tp+fp+fn+tn)
        report.loc[ele, 'test_prec'] = tp / (tp+fp)
        report.loc[ele, 'test_f1'] = 2*tp / (2*tp + fp + fn)
    
    print(report)
        

gc.collect()


# %% Grid search (Multiple output)
if __name__ == "__main__":
    
    m_features = [1000,2000]#,2000,3000,4000,5000,6000,7000,8000,9000,10000] # tunning parameters
       
    recall_scorer = make_scorer(recall_score, greater_is_better=True)       
    f1_scorer = make_scorer(fbeta_score, beta=1)
    
    report = pd.DataFrame(np.zeros((len(m_features),10)), index=m_features, 
                          columns=['val_sens','val_spec','val_acc','val_prec','val_f1','test_sens','test_spec','test_acc','test_prec','test_f1'])   
          
    for ele in m_features:
        pipeline = Pipeline([
            ('vect', CountVectorizer(stop_words='english', lowercase=True, token_pattern=r'\w{1,}', min_df=2,
                                     max_features=ele)),
            ('tfidf', TfidfTransformer(norm='l2', use_idf=True, sublinear_tf=False)),
            ('clf_sgd', SGDClassifier(random_state=66, 
                                      #max_iter=50, 
                                      alpha=0.0001, penalty='l2')),
        ])            
        parameters = { 
                        'vect__ngram_range': ((1,1), (1,2)),                      
                     }
    
        clf = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1, verbose=1, scoring=f1_scorer)      
        clf.fit(X_train, y_train) 
        
        ### Validation ###
        pred = pd.DataFrame(np.zeros((len(y_valid), 1)), columns=list([ele]))  
        y_true, y_pred = y_valid, clf.predict(X_valid)
        pred.loc[:, ele] = y_pred

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()      
        report.loc[ele, 'val_sens'] = tp / (tp+fn)
        report.loc[ele, 'val_spec'] = tn / (tn+fp)
        report.loc[ele, 'val_acc'] = (tp+tn) / (tp+fp+fn+tn)
        report.loc[ele, 'val_prec'] = tp / (tp+fp)
        report.loc[ele, 'val_f1'] = 2*tp / (2*tp + fp + fn)

        ### Test ###
        pred = pd.DataFrame(np.zeros((len(y_test), 1)), columns=list([ele]))        
        y_true, y_pred = y_test, clf.predict(X_test)
        pred.loc[:, ele] = y_pred

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()      
        report.loc[ele, 'test_sens'] = tp / (tp+fn)
        report.loc[ele, 'test_spec'] = tn / (tn+fp)
        report.loc[ele, 'test_acc'] = (tp+tn) / (tp+fp+fn+tn)
        report.loc[ele, 'test_prec'] = tp / (tp+fp)
        report.loc[ele, 'test_f1'] = 2*tp / (2*tp + fp + fn)
    
    print(report)
        
clf.best_estimator_.get_params()  

clf.cv_results_

report.to_csv('report_sgd_numfeatures.csv')

# Save best estimators
# joblib.dump(clf.best_estimator_, 'SGD_seed10.pkl')
gc.collect()

#%% Pre
#SGDClassifier(loss='hinge',              # 'hinge' gives a linear SVM.
#              penalty='l2',              # l2 is the standard regularizer for linear SVM models
#              class_weight=None,
#              
#              learning_rate='optimal',   # eta = 1.0 / (alpha * (t + t0))
#              alpha=0.0001,              # Used to compute learning_rate when set to 'optimal'
#        
#              random_state=None,         # Seed for shuffling the data
#              shuffle=True,              # Shuffle the training data after each epoch 
#              
#              early_stopping=False,   
#              max_iter=50,               # The maximum number of epochs over the training data
#              tol=0.001,                 # The stopping criterion
#              n_iter_no_change=5,        # Number of iterations with no improvement to wait before early stopping
#              validation_fraction=0.1,   # The proportion of training data to set aside as validation set for early stopping
#                            
#              n_jobs=4, 
#              verbose=0)













