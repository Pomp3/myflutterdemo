
#!/usr/bin/python2
# -*- coding: utf-8 -*-
# $File: learner.py
# $Date: Sun May 25 19:09:33 2014 +0800
# $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>
#
# TODO:
#   generalize metrics, see:
#       http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics

import sklearn
from sklearn.neighbors import * # KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import * # SVC, SVR, NuSVR, LinearSVC

# LinearRegression, LogisticRegression
# SGDClassifier, Perceptron, Ridge, Lasso, ElasticNet
from sklearn.linear_model import *

# MultinomialNB, BernoulliNB
from sklearn.naive_bayes import *

from sklearn.tree import * # DecisionTreeClassifier, DecisionTreeRegressor

from scipy import sparse

# RandomForestClassifier, RandomForestRegressor
# ExtraTreesClassifier, ExtraTreesRegressor
# AdaBoostClassifier, AdaBoostRegressor
# GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import *

# save model
from sklearn.externals import joblib
import pickle

# metrics
from sklearn import metrics

import numpy as np
from collections import defaultdict
import argparse
import sys
import types


models = {
        'linearr': LinearRegression,
        'logisticr': LogisticRegression,
        'knnc': KNeighborsClassifier,
        'knnr': KNeighborsRegressor,
        'svc': SVC,
        'svr': SVR,
        'nusvr': NuSVR,
        'lsvc': LinearSVC,
        'sgdc': SGDClassifier,
        'dtc': DecisionTreeClassifier,
        'dtr': DecisionTreeRegressor,
        'rfc': RandomForestClassifier,
        'rfr': RandomForestRegressor,
        'etc': ExtraTreesClassifier,
        'etr': ExtraTreesRegressor,
        'abc': AdaBoostClassifier,
        'abr': AdaBoostRegressor,
        'gbc': GradientBoostingClassifier,
        'gbr': GradientBoostingRegressor,
        'perceptron': Perceptron,
        'ridge': Ridge,
        'lasso': Lasso,
        'elasticnet': ElasticNet,
        'mnb': MultinomialNB,
        'bnb': BernoulliNB,
        }

sparse_models = set([
    SVR,
    NuSVR,
    LinearSVC,
    KNeighborsClassifier,
    KNeighborsRegressor,
    SGDClassifier,
    Perceptron,
    Ridge,
    LogisticRegression,
    LinearRegression,
    ])

args = []

# table is a 2-D string list
# return a printed string
def format_table(table):
    if len(table) == 0:
        return ''
    col_length = defaultdict(int)
    for row in table:
        for ind, item in enumerate(row):
            col_length[ind] = max(col_length[ind], len(item))

    # WARNING: low efficiency, use string buffer instead
    ret = ''
    for row in table:
        for ind, item in enumerate(row):
            fmtstr = '{{:<{}}}' . format(col_length[ind])
            ret += fmtstr.format(item) + " "
        ret += "\n"
    return ret

def get_model_abbr_help():
    lines = format_table([['Abbreviation', 'Model']] + map(lambda item: [item[0], item[1].__name__], \
        sorted(models.items()))).split('\n')
    return "\n".join(map(lambda x: ' ' * 8 + x, lines))

class VerboseAction(argparse.Action):
    def __call__(self, parser, args, values, option_string=None):
        # print 'values: {v!r}'.format(v=values)
        if values==None:
            values='1'
        try:
            values=int(values)
        except ValueError:
            values=values.count('v')+1
        setattr(args, self.dest, values)

def get_args():
    description = 'command line wrapper for some models in scikit-learn'

    tasks = ['fit', 'predict', 'fitpredict',
            'f', 'p', 'fp',
            'doc']

    # (task_names, required_params, optional_params)
    task_arg_setting = [
            (['fit', 'f'],
                ['training_file', 'model', 'model_output'],
                ['model_options']),
            (['predict', 'p'],
                ['test_file', 'model_input', 'prediction_file'],
                []),
            (['fitpredict', 'fp'],
                ['training_file', 'model', 'test_file', 'prediction_file'],
                ['model_options', 'model_output']),
            (['doc'],
                ['model'],
                [])
            ]
