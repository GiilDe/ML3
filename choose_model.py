from sklearn.model_selection import cross_val_score
from get_prepared_data import get_prepared_data
import numpy as np
from sklearn.model_selection import ShuffleSplit

train_X, train_Y, _, _, _, _ = get_prepared_data(load=True)


from sklearn import svm
'''____________tuning parameters for SVR with cross K validation____________'''
max_score = 0
max_hyper_parameters = 'None'
cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
    if kernel == 'poly':
        for degree in range(10):
            clf = svm.SVR(kernel=kernel, degree=degree)
            score = np.average(cross_val_score(clf, train_X, train_Y, cv=cv))
            if score > max_score:
                max_score = score
                max_hyper_parameters = 'kernel = ' + str(kernel) + ', degree = ' + str(degree)
    else:
        clf = svm.SVR(kernel=kernel)
        score = np.average(cross_val_score(clf, train_X, train_Y, cv=cv))
        if score > max_score:
            max_score = score
            max_hyper_parameters = str(kernel)

print('best hyper parameters for svm.SVR are: ', max_hyper_parameters, '.\n score is: ', max_score)
with open('best hyper parameters for svm.SVR', 'w') as f:
    f.write('best hyper parameters for svm.SVR are: ' + max_hyper_parameters + '. score is: ' + str(max_score))

from sklearn.svm import LinearSVC
'''____________tuning parameters for LinearSVC with cross K validation____________'''
max_score = 0
max_hyper_parameters = 'None'
for penalty in ['l2']:
    for C in np.arange(0.5, 4, 0.5):
        clf = LinearSVC(penalty=penalty, C=C)
        score = np.average(cross_val_score(clf, train_X, train_Y,cv=cv))
        if score > max_score:
            max_score = score
            max_hyper_parameters = 'penalty = ' + penalty + ', C = ' + str(C)

print('best hyper parameters for LinearSVC are: ', max_hyper_parameters, '.\n score is: ', max_score)
with open('best hyper parameters for LinearSVC', 'w') as f:
    f.write('best hyper parameters for LinearSVC are: ' + max_hyper_parameters)

from sklearn import neighbors
'''____________tuning parameters for LinearSVC with cross K validation____________'''
max_score = 0
max_hyper_parameters = 'None'
for weights in ['uniform', 'distance']:
    for n_neighbors in [1, 3, 5, 7, 9]:
        clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
        score = np.average(cross_val_score(clf, train_X, train_Y, cv=cv))
        if score > max_score:
            max_score = score
            max_hyper_parameters = 'n_neighbors = ' + str(n_neighbors) + ', weights = ' + weights

print('best hyper parameters for KNN are: ', max_hyper_parameters, '.\n score is: ', max_score)
with open('best hyper parameters for KNN', 'w') as f:
    f.write('best hyper parameters for KNN are: ' + max_hyper_parameters)

'''____________tuning parameters for RandomForestClassifier with cross K validation____________'''
from sklearn.ensemble import RandomForestClassifier
max_score = 0
max_hyper_parameters = 'None'
for n_estimators in range(2, 15):
    clf = RandomForestClassifier(n_estimators=n_estimators)
    score = np.average(cross_val_score(clf, train_X, train_Y, cv=cv))
    if score > max_score:
        max_score = score
        max_hyper_parameters = 'n_estimators = ' + str(n_estimators)

print('best hyper parameters for RandomForestClassifier are: ', max_hyper_parameters, '.\n score is: ', max_score)
with open('best hyper parameters for RandomForestClassifier', 'w') as f:
    f.write('best hyper parameters for RandomForestClassifier are: ' + max_hyper_parameters)


'''____________tuning parameters for ExtraTreeClassifier with cross K validation____________'''
from sklearn.ensemble import ExtraTreesClassifier
max_score = 0
max_hyper_parameters = 'None'
for n_estimators in range(2, 15):
    for criterion in ['gini', 'entropy']:
        clf = ExtraTreesClassifier(n_estimators=n_estimators, criterion=criterion)
        score = np.average(cross_val_score(clf, train_X, train_Y, cv=cv))
        if score > max_score:
            max_score = score
            max_hyper_parameters = 'n_estimators = ' + str(n_estimators) + ', criterion = ' + criterion

print('best hyper parameters for ExtraTreesClassifier are: ', max_hyper_parameters, '.\n score is: ', max_score)
with open('best hyper parameters for ExtraTreesClassifier', 'w') as f:
    f.write('best hyper parameters for ExtraTreesClassifier are: ' + max_hyper_parameters)
