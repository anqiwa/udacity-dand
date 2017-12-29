#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier
import operator
import collections
import matplotlib.pyplot as plt

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

def sep_1():
    print '\n\n', '*' * 100

def sep_2():
    print '*' * 100

# What features are there?
sep_1()    
print 'Features:'
sep_2() 
for i in data_dict.keys():
    for item in data_dict[i]:
        print item
    break

features_list = ['poi','salary', 'deferral_payments','total_payments','exercised_stock_options','bonus',
                 'restricted_stock','restricted_stock_deferred','total_stock_value','expenses', 'loan_advances',
                 'director_fees','deferred_income','long_term_incentive', 'to_messages','shared_receipt_with_poi',
                 'from_messages','from_this_person_to_poi', 'from_poi_to_this_person'] # You will need to use more features

### Task 2: Remove outliers
sep_1()
features = ['bonus', 'long_term_incentive']
data = featureFormat(data_dict, features)
for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter(salary, bonus)
plt.xlabel('bonus')
plt.ylabel('long term incentive')
plt.show()

# One outlier appears in the graph, what is it?
bonus = {}
for i in data_dict.keys():
    if data_dict[i]['bonus'] != 'NaN':
        bonus[i] = data_dict[i]['bonus']
 
print '\nOutlier: ', max(bonus, key=bonus.get)
sep_2()

# Obviously TOTAL is an outlier, so remove it.
data_dict.pop('TOTAL', 0)

# Plot again
data = featureFormat(data_dict, features)
for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter(salary, bonus)
plt.xlabel('bonus')
plt.ylabel('long term incentive')
plt.show()

# Number of data points in this dataset
sep_1() 
print 'Number of data points after removing outlier:', len(data_dict)
sep_2() 

# Number of data points with NaN values
people_NaN = {} 
for i in data_dict.keys():
    count_item = []
    for item in data_dict[i]:     
        if data_dict[i][item] == 'NaN':
            count_item.append(item)
    l = len(count_item)
    if l in people_NaN.keys():
        people_NaN[l].update([i])
    else:
        people_NaN[len(count_item)] = {i}

od = collections.OrderedDict(reversed(list(people_NaN.items())))
sep_1()
print 'Data points with NaN values sorted in descending order:'
sep_2()
for k, v in od.iteritems(): 
    print k, '\n', list(v)

# LOCKHART EUGENE E has most NaN values. 
# How many features does each observation have?
for i in data_dict.keys():
    l = len(data_dict[i].items())    

sep_1()
print 'Number of features:', l
sep_2()

# Since 20 out of 21 features of LOCKHART EUGENE E are NaN,
# What is the non-NaN feature of him? 
sep_1() 
print 'LOCKHART EUGENE E:\n', 
sep_2() 
for i in data_dict['LOCKHART EUGENE E'].items():
    if 'NaN' not in i:
        print i

# Since LOCKHART EUGENE E has no valuable info,
# it is resonable to remove it though it's not an outlier.
data_dict.pop('LOCKHART EUGENE E', 0)

# Number of data points in this dataset
sep_1() 
print 'Number of data points after removing valueless data point:', len(data_dict)
sep_2() 

# Plot again
data = featureFormat(data_dict, features)
for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter(salary, bonus)
plt.xlabel('bonus')
plt.ylabel('long term incentive')
plt.show()
    
# How many data points with what features have NaN values?    
features_NaN = {}
for i in data_dict.keys():   
    for item in data_dict[i]:
        if data_dict[i][item] == 'NaN':
            if item in features_NaN.keys():
                features_NaN[item] += 1
            else:
                features_NaN[item] = 1

sorted_features_NaN = sorted(features_NaN.items(), key=operator.itemgetter(1), reverse=True)

sep_1() 
print 'Features with NaN values sorted in descending order:'
sep_2()
for i in sorted_features_NaN:
    print i

# What percentages of features are missing values?
missing = 0
total = 0
for i in data_dict.keys():
    for item in data_dict[i]:
        if data_dict[i][item] == 'NaN':
            missing += 1
        total += 1

sep_1()
print 'Percentage of features with missing values: ', float(missing)/float(total)
sep_2()

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

# I create three new features: 
def computeFraction(poi_messages, all_messages):
    if poi_messages != 'NaN' and all_messages != 'NaN':
        fraction = float(poi_messages)/float(all_messages)
    else:
        fraction = 0.
    return fraction

for i in my_dataset:
    to_poi = my_dataset[i]['from_this_person_to_poi']
    from_poi = my_dataset[i]['from_poi_to_this_person']
    to_messages = my_dataset[i]['to_messages']
    from_messages = my_dataset[i]['from_messages']
    shared_receipt_with_poi = my_dataset[i]['shared_receipt_with_poi']
    my_dataset[i]['fraction_from_this_person_to_poi'] = computeFraction(to_poi, from_messages)
    my_dataset[i]['fraction_to_this_person_from_poi'] = computeFraction(from_poi, to_messages)
    my_dataset[i]['fraction_shared_receipt_with_poi'] = computeFraction(shared_receipt_with_poi, from_messages)

# Plot new features
features = ['fraction_from_this_person_to_poi', 'fraction_to_this_person_from_poi']
data = featureFormat(my_dataset, features)
for point in data:
    fraction_from_this_person_to_poi = point[0]
    fraction_to_this_person_from_poi = point[1]
    plt.scatter(fraction_from_this_person_to_poi, fraction_to_this_person_from_poi)
plt.xlabel("fraction of emails from this person to POI's")
plt.ylabel("fraction of emails from POI's to this person")
plt.show()

# Add new features to features_list
features_list.extend(['fraction_from_this_person_to_poi', 'fraction_to_this_person_from_poi', 'fraction_shared_receipt_with_poi'])

sep_1()    
print 'Features list updated with new features: \n', 
for f in features_list:
    print '-', f
sep_2()

### Extract features and labels from dataset for local testing
# Convert dictionary to numpy array
data = featureFormat(my_dataset, features_list, sort_keys = True)

# Separate 'poi' from other features
labels, features = targetFeatureSplit(data)

#from sklearn.preprocessing import Imputer
#features = Imputer().fit_transform(features)

# K-best features
from sklearn.feature_selection import SelectKBest
k_best = SelectKBest(k=10)
k_best.fit(features, labels)

sep_1()
results_list = zip(k_best.get_support(), features_list[1:], k_best.scores_)
feature_score = sorted(results_list, key=lambda x: x[2], reverse=True)
top_features = feature_score[:10]
print 'K-best features: ', top_features
sep_2()

features_list = ['poi']
for i in top_features:
    features_list.append(i[1])
sep_1()   
print 'Updated features list with only 10 features: '
for i in features_list:
    print '-', i
sep_2()

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(features, labels)
sep_1()
print 'Classifier: Naive Bayes '
test_classifier(clf, my_dataset, features_list)
sep_2()

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(features, labels)
print '\nClassifier: Decision Tree '
test_classifier(clf, my_dataset, features_list)
sep_2()

# feature importances
import numpy as np
feature_importances = clf.feature_importances_
features_ranking = (-np.array(feature_importances)).argsort()
sep_1()
print "Features ranking"
sep_2()
for i in features_ranking:
    print feature_importances[i], features_list[i+1]


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
#from sklearn.cross_validation import train_test_split
#features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# StratifiedShuffleSplit
from sklearn.cross_validation import StratifiedShuffleSplit
ss = StratifiedShuffleSplit(labels, 100, random_state=42)

# GridSearchCV
from sklearn.model_selection import GridSearchCV
params = {'criterion': ['gini','entropy'],
          'splitter': ['best','random'],
          'max_depth': [None,1,10,20],
          'min_samples_split':[2,5,10],
          'min_samples_leaf': [1,5,10],
          'max_leaf_nodes': [None,2,5,10],
          'min_weight_fraction_leaf': [0,0.25,0.5]
          }
dt = DecisionTreeClassifier()
clf = GridSearchCV(dt, params, cv=ss, scoring='f1')
clf.fit(features, labels)
sep_1()
print '\nBest parameters: ', clf.best_params_
sep_2()

# Best parameters
clf = DecisionTreeClassifier(splitter='best', max_leaf_nodes=5, min_samples_leaf=5, 
                             min_weight_fraction_leaf=0, criterion='entropy', min_samples_split=2,
                             max_depth=1)
print 'Tuned Decision Tree'
test_classifier(clf, my_dataset, features_list)
sep_2()

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)