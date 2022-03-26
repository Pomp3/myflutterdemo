
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

    epilog = "task specification:\n{}" . format(
            "\n" . join([
                '    task name: {}\n        required arguments: {}\n        optional arguments: {}' . format(
                    *map(lambda item: ", " . join(item), setting)) \
                            for setting in task_arg_setting]))
    epilog += "\n"

    epilog += "Notes:\n"
    epilog += "    1. model abbreviation correspondence:\n"
    epilog += get_model_abbr_help() + '\n'
    epilog += "    2. model compatible with sparse matrix:\n"
    epilog += ' ' * 8 + ", " . join(map(lambda x: x.__name__, sparse_models))
    epilog += '\n'
    epilog += '\n'

    epilog += 'Examples:\n'
    epilog += """\
    1. fit(train) a SVR model with sigmoid kernel:
        ./learner.py -t f --training-file training-data --model svr \\
                --model-output model.svr kernel:s:sigmoid

    2. predict using precomputed model:
        ./learner.py -t p --test-file test --model-input model.svr
            --prediction-file pred-result

    3. fit and predict, model saved, verbose output, and show metrics:
        ./learner.py -t fp --training-file training-data --model svr \\
            --model-output model.svr --test-file test-data \\
            --prediction-file pred-result -v --show-metrics

    4. pass parameters for svc model, specify linear kernel:
        ./learner.py --task fp --training-file training-data --model svc \\
            --test-file test-data --prediction-file pred-result \\
            --show-metrics kernel:s:linear

    5. show documents:
        ./learner.py -t doc --model svc
"""


    parser = argparse.ArgumentParser(
            description = description, epilog = epilog,