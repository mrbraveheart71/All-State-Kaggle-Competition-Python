#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
modified from Vladimir Iglovikov tilii7 and extremin
"""

import pandas as pd
import subprocess
import numpy as np
import xgboost as xgb
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold

def logregobj(preds, dtrain):
    labels = dtrain.get_label()
    con = 2
    x = preds - labels
    grad = con * x / (np.abs(x) + con)
    hess = con ** 2 / (np.abs(x) + con) ** 2
    return grad, hess


def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'mae', mean_absolute_error(np.exp(preds), np.exp(labels))

## read data
train = pd.read_csv('../train.csv')
test = pd.read_csv('../test.csv')

index = list(train.index)
print
index[0:10]
np.random.shuffle(index)
print
index[0:10]
train = train.iloc[index]
'train = train.iloc[np.random.permutation(len(train))]'

## set test loss to NaN
test['loss'] = np.nan

## response and IDs
shift = 200
y = np.log(train['loss'].values + shift)
id_train = train['id'].values
id_test = test['id'].values

## stack train test
ntrain = train.shape[0]
tr_te = pd.concat((train, test), axis=0)

## Preprocessing and transforming to sparse data
sparse_data = []

f_cat = [f for f in tr_te.columns if 'cat' in f]
for f in f_cat:
    series_tr_te = tr_te[f]
    counts = pd.value_counts(series_tr_te)
    mask = series_tr_te.isin(counts[counts > 5].index)
    series_tr_te[~mask] = "-"
    dummy = pd.get_dummies(series_tr_te.astype('category'))
    tmp = csr_matrix(dummy)
    sparse_data.append(tmp)

f_num = [f for f in tr_te.columns if 'cont' in f]
scaler = StandardScaler()
tmp = csr_matrix(scaler.fit_transform(tr_te[f_num]))
sparse_data.append(tmp)

del (tr_te, train, test)

## sparse train and test data
xtr_te = hstack(sparse_data, format='csr')
xtrain = xtr_te[:ntrain, :]
xtest = xtr_te[ntrain:, :]

print('Dim train', xtrain.shape)
print('Dim test', xtest.shape)

del (xtr_te, sparse_data, tmp)

n_folds = 5
kf = KFold(xtrain.shape[0], n_folds=n_folds)
prediction = np.zeros(id_train.shape)

final_fold_prediction = []
final_fold_real = []

partial_evalutaion = open('temp_scores.txt', 'w')

for i, (train_index, test_index) in enumerate(kf):
    print('\n Fold %d' % (i + 1))
    X_train, X_val = xtrain[train_index,:], xtrain[test_index,:]
    y_train, y_val = y[train_index], y[test_index]

    RANDOM_STATE = 2016
    params = {
        'min_child_weight': 1,
        'eta': 0.01,
        'colsample_bytree': 0.5,
        'max_depth': 12,
        'subsample': 0.8,
        'alpha': 1,
        'gamma': 1,
        'silent': 1,
        'verbose_eval': True,
        'seed': RANDOM_STATE
    }

    xgtrain = xgb.DMatrix(X_train, label=y_train)
    xgtrain_2 = xgb.DMatrix(X_val, label=y_val)

    xgtest = xgb.DMatrix(xtest)

    watchlist = [(xgtrain, 'train'), (xgtrain_2, 'eval')]

    model = xgb.train(params, xgtrain, 100000, watchlist, obj=logregobj, feval=evalerror, early_stopping_rounds=100)
    prediction += np.exp(model.predict(xgtest)) - shift

    X_val = xgb.DMatrix(X_val)
    temp_serises = pd.Series(np.exp(model.predict(X_val)) - shift)
    final_fold_prediction.append(temp_serises)
    temp_serises = np.exp(y_val) - shift
    final_fold_real.append(temp_serises)

    temp_cv_score = mean_absolute_error(np.exp(model.predict(X_val)) - shift, np.exp(y_val) - shift)

    partial_evalutaion.write('fold ' + str(i) + ' ' + str(temp_cv_score) + '\n')
    partial_evalutaion.flush()

prediction = prediction / n_folds
submission = pd.DataFrame()
submission['id'] = id_test
submission['loss'] = prediction

submission.to_csv('sub_v_5_long2_python.csv', index=False)

final_fold_prediction = pd.concat(final_fold_prediction, ignore_index=True)
final_fold_real = pd.concat(final_fold_real, ignore_index=True)

cv_score = mean_absolute_error(final_fold_prediction, final_fold_real)
print(cv_score)
