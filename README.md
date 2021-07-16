
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