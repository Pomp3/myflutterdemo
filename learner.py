
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
            formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-t', '--task',
            choices = tasks,
            help = 'task to process, see help for detailed information',
            required = True)
    parser.add_argument('--training-file',
            help = 'input: training file, svm format by default')
    parser.add_argument('--test-file',
            help = 'input: test file, svm format by default')
    parser.add_argument('--model-input',
            help = 'input: model input file, used in prediction')
    parser.add_argument('--model-output',
            help = 'output: model output file, used in fitting')
    parser.add_argument('-m', '--model',
            help = 'model, specified in fitting',
            choices = models)
    parser.add_argument('--prediction-file',
            help = 'output: prediction file')

    parser.add_argument('--model-format',
            choices = ['pickle', 'joblib'],
            default = 'pickle',
            help = 'model format, pickle(default) or joblib')

    parser.add_argument('--show-metrics',
            action = 'store_true',
            help = 'show metric after prediction')

    parser.add_argument('-v', '--verbose',
            help = 'verbose level, -v <level> or multiple -v\'s or something like -vvv',
            nargs = '?',
            default = 0,
            action = VerboseAction)

    parser.add_argument('model_options',
            nargs = '*',
            help = """\
additional paramters for specific model of format "name:type:val", \
effective only when training is needed. type is either int, float or str, \
which abbreviates as i, f and s.""")

    args = parser.parse_args()

    model_options = dict()
    for opt in args.model_options:
        opt = opt.split(':')
        assert len(opt) == 3, 'model option format error'
        key, t, val = opt
        if t == 'i':
            t = 'int'
        elif t == 'f':
            t = 'float'
        elif t == 's':
            t = 'str'
        elif t == 'b':
            t = 'bool'

        model_options[key] = eval(t)(val)

    args.model_options = model_options

    # make task name a full name
    for setting in task_arg_setting:
        if args.task in setting[0]:
            args.task = setting[0][0]

    # check whether paramters for specific task is met
    def check_params(task, argnames):
        if args.task in task:
            for name in argnames:
                if not(name in args.__dict__ and args.__dict__[name]):
                    info = 'argument `{}\' must present in `{}\' task' .\
                            format("--" + name.replace('_', '-'), task[0])
                    raise Exception(info)

    try:
        for setting in task_arg_setting:
            check_params(setting[0], setting[1])
    except Exception as e:
        sys.stderr.write(str(e) + '\n')
        sys.exit(1)

    # add a verbose print method to args
    def verbose_print(self, msg, vb = 1): # vb = verbose_level
        if vb <= self.verbose:
            print(msg)

    args.vprint = types.MethodType(verbose_print, args, args.__class__)


    return args

def read_svmformat_data(fname):
    from sklearn.datasets import load_svmlight_file
    X, y = load_svmlight_file(fname)
    return X, y


def write_labels(fname, y_pred):
    count_types = defaultdict(int)
    for y in y_pred:
        count_types[type(y)] += 1
    most_prevalent_type = sorted(map(lambda x: (x[1], x[0]), count_types.iteritems()))[0][1]
    if most_prevalent_type == float:
        typefmt = '{:f}'
    else:
        typefmt = '{}'

    with open(fname, 'w') as fout:
        for y in y_pred:
            fout.write(typefmt.format(y) + '\n')


def get_model(args):
    model = models[args.model](**args.model_options)
    return model

def get_dim(X):
    dim = -1
    for x in X:
        for ind, val in x:
            if ind > dim:
                dim = ind
    return dim + 1

def preprocess_data(model, X):
    assert type(X) == sparse.csr_matrix
    if model in sparse_models: