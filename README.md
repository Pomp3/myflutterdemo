
sklearn-cmdline-wrapper
=======================

access sklearn (scikit-learn) machine learning library via command line.

Introduction
------------
### What is sklearn?
__sklearn__ is a python machine learning library comprised of various
machine learning algorithms. See [here](http://scikit-learn.org/stable/) for detail.

### Why command line?
Although you can write a python script every time you like to use functionalities
provided by sklearn, it is still annoying to write duplicated script again and again.
It is both error prone and inefficient, especially when you want to perform comparisons
on different learning algorithms.

This off-the-shell command line tool is here to make your life easier

Prerequisite
------------
Of course You have to install sklearn :) , move steps to the [official website of sklearn](http://scikit-learn.org/stable/)

This script is only tested on version 0.14.1.
For lower versions, something like ```AdaBoostClassifier``` will not work.


Features
--------
- only ONE script, you can copy and paste it as you like.
- only supervised learning tasks are supported currently
- parameters(limited, but sufficient) of model can be passed via command-line,
	you can even make the model utilize mult-cores.
- automatically use sparse matrix if model supports
- libsvm format input

For detailed information: ```./learner.py -h```

Todo (let's make it a better script)
------------------------------------
- compatibility between different versions of sklearn
- support for unsupervised learning tasks
- data visualization for unsupervised learning tasks
- more metrics
- more input file type (like csv, or just space separated columns,
better if there's automatically detection).

Known Issue
-----------
Due to internal data structures transition, this script is not
memory-efficient. It may eat up more memory that you expected.
For this reason, it is not recommend to use this script on "Big Data"

Contact
-------
Xinyu Zhou <zxytim[at]gmail[dot]com>

License
-------
GPL v3

Help
----
output of ```./learner.py -h```

	usage: learner.py [-h] -t {fit,predict,fitpredict,f,p,fp,doc}
					  [--training-file TRAINING_FILE] [--test-file TEST_FILE]
					  [--model-input MODEL_INPUT] [--model-output MODEL_OUTPUT]
					  [-m {logisticr,knnc,mnb,perceptron,lsvc,lasso,abc,ridge,abr,elasticnet,bnb,knnr,sgdc,etr,rfr,nusvr,gbc,dtc,linearr,svc,rfc,etc,gbr,dtr,svr}]