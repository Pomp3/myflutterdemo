
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
