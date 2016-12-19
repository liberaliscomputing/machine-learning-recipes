#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
Hello World - Machine Learning Recipies #1
December 13, 2016
Coded by Meen Chul Kim

Note:
Supervised learning is a process of creating a classifier 
	by finding patterns in examples
'''

from sklearn import tree


# Declare features consisting of weight and smoothness
#		Smooth: 1, bumpy: 0
features = [
	[140, 1], [130, 1], [150, 0], [170, 0]
]

# Define lables
# 	Apple: 0, orange: 1
labels = [0, 0, 1, 1]

# Build a Decision Tree Classifier
clf = tree.DecisionTreeClassifier()

# Train the classifier
clf.fit(features, labels)

# Predict with a new example
data = [[150, 0]]
print clf.predict(data) #returns 1 for orange