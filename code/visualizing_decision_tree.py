#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
Visualizing a Decision Tree - Machine Learning Recipies #2
December 13, 2016
Coded by Meen Chul Kim

Goals:
1. Import dataset
2. Train a classifier
3. Predict label for new flower
4. Visualize the tree

Note:
Sample datasets at http://scikit-learn.org/stable/datasets/
Graphi viualization at http://graphviz.org/
'''

import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.externals.six import StringIO
import pydot


# Import dataset
iris = load_iris()

'''
Explore data

1. Column names
print iris.feature_names

2. Labels
print iris.target_names

3. Keys
print iris.keys()

4. Input values of the 1st sample
print iris.data[0]

5. Output of the 1st sample
print iris.target[0]

for i in range(len(iris.target)):
	print 'Example {}: lable {}, features {}'.format(
		i, iris.target[i], iris.data[i]
	)
'''

# Prepare testing data
#		Examples to be used to test the classifier's accuracy
#		Not part of the traning data
test_idx = np.random.randint(0, 100, 3)
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

# Remove testing data from training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# Build and train a Decision Tree classifier
clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

# Test the classifier
# print test_target 
# print clf.predict(test_data)

# Predict label for new flower
# 	and visualize the tree
dot_data = StringIO()
tree.export_graphviz(clf,
	out_file=dot_data,
	feature_names=iris.feature_names,
	class_names=iris.target_names,
	filled=True,
	rounded=True,
	impurity=False
)

graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph[-1].write_pdf('iris.pdf')