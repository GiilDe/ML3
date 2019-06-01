from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
from collections import Counter
import ast
from get_prepared_data import get_prepared_data


print('finished importing')


class Model_Chooser:
    def __init__(self, classifiers_dict=None, score_measurement_for_find_best_params='accuracy', predict_proba=False):
        if classifiers_dict is None:
            if not predict_proba:
                self.classifiers_dict = {  # key is classifier name, value is a list of tuples, where first val is parameter
                                      # name for the model and second val is a list of possible values for the parameter
                    'KNeighborsClassifier': [('n_neighbors', list(range(1, 15, 2)))],  # predict_proba
                    'SVC': [('kernel', ['linear', 'poly', 'rbf', 'sigmoid']), ('gamma', ['scale'])],
                     'DecisionTreeClassifier': [('min_samples_split ', list(range(2, 52, 5)))],  # predict_proba
                    'RandomForestClassifier': [('n_estimators', list(range(5, 20, 5))),  # predict_proba
                                               ('min_samples_split', list(range(2, 52, 5)))],
                    'GaussianNB': [],  # predict_proba
                    'LogisticRegression': [('solver', ['lbfgs'])],  # predict_proba
                     'QuadraticDiscriminantAnalysis': [],  # predict_proba
                    }
            else:
                self.classifiers_dict = {  # key is classifier name, value is a list of tuples, where first val is parameter
                    # name for the model and second val is a list of possible values for the parameter
                    'KNeighborsClassifier': [('n_neighbors', list(range(1, 15, 2)))],  # predict_proba
                    'DecisionTreeClassifier': [('min_samples_split ', list(range(2, 52, 5)))],  # predict_proba
                    'RandomForestClassifier': [('n_estimators', list(range(5, 20, 5))),  # predict_proba
                                               ('min_samples_split', list(range(2, 52, 5)))],
                    'GaussianNB': [],  # predict_proba
                    'LogisticRegression': [('solver', ['lbfgs'])],  # predict_proba
                    'QuadraticDiscriminantAnalysis': [],  # predict_proba
                }
            self.classifiers_params_dict = dict()

    def create_classifier(self, classifier, param_list):
        if len(param_list) == 0:
            clf = eval(classifier + '()')
        elif len(param_list) == 1:
            param_name = param_list[0][0]
            param_val = param_list[0][1]
            clf = eval(classifier + '(' + param_name + '=' + str(param_val) + ')')
        elif len(param_list) == 2:
            first_param_name = param_list[0][0]
            first_param_val = param_list[0][1]
            second_param_name = param_list[1][0]
            second_param_val = param_list[1][1]
            clf = eval(classifier + '(' + first_param_name + '=' + str(first_param_val) + ', ' +
                       second_param_name + '=' + str(second_param_val) + ')')
        else:
            print('Param number greater than 2 is unsupported at the moment'.capitalize())
            exit()
        return clf

    def find_classifiers_best_params(self, X, Y, score_measure='accuracy'):
        for classifier, param_list in self.classifiers_dict.items():
            if len(param_list) == 0:
                self.classifiers_params_dict[classifier] = []
            elif len(param_list) == 1:
                best_score = 0
                best_param = None
                param_name = param_list[0][0]
                for param_val in param_list[0][1]:
                    if isinstance(param_val, str):
                        param_val = '\'' + param_val + '\''
                    clf = eval(classifier + '(' + param_name + '=' + str(param_val) + ')')
                    score = np.average(cross_validate(clf, X, Y, scoring=score_measure, cv='warn')['test_score'])
                    if score > best_score:
                        best_score = score
                        best_param = param_val
                self.classifiers_params_dict[classifier] = [(param_name, best_param)]
            elif len(param_list) == 2:
                best_score = 0
                best_first_param = None
                best_second_param = None
                first_param_name = param_list[0][0]
                second_param_name = param_list[1][0]
                for first_param_val in param_list[0][1]:
                    if isinstance(first_param_val, str):
                        first_param_val = '\'' + first_param_val + '\''
                    for second_param_val in param_list[1][1]:
                        if isinstance(second_param_val, str):
                            second_param_val = '\'' + second_param_val + '\''
                        clf = eval(classifier + '(' + first_param_name + '=' + str(first_param_val) + ', ' +
                                   second_param_name + '=' + str(second_param_val) + ')')
                        score = np.average(cross_validate(clf, X, Y, cv='warn')['test_score'])
                        if score > best_score:
                            best_score = score
                            best_first_param = first_param_val
                            best_second_param = second_param_val
                self.classifiers_params_dict[classifier] = [(first_param_name, best_first_param),
                                                            (second_param_name, best_second_param)]
            else:
                print('Param number greater than 2 is unsupported at the moment'.capitalize())
        return self.classifiers_params_dict

    def get_winner(self, train_X, train_Y, test_X, test_Y, score_measure_func=None):
        best_score = float('-inf')
        best_classifier = None
        best_param_list = []
        for classifier, param_list in self.classifiers_params_dict.items():
            clf = self.create_classifier(classifier, param_list)
            clf.fit(train_X, train_Y)
            prediction = clf.predict(test_X)
            if score_measure_func is not None:
                score = score_measure_func(prediction, test_Y)
            else:
                score = accuracy_score(prediction, test_Y)
            if score > best_score:
                best_score = score
                best_classifier = classifier
                best_param_list = param_list
        return best_classifier, best_param_list, best_score








# ______________________________________________________________________________________________________________________
# FIRST TASK - Predict which party will win the majority of votes

# this is the score measurer fot this task
def party_win_score(prediction, test_Y):
    prediction_counter = list(Counter(prediction).items())
    test_Y_counter = list(Counter(test_Y).items())
    predicted_winner = max(prediction_counter, key=lambda x: x[1])[0]
    real_winner = max(test_Y_counter, key=lambda x: x[1])[0]
    if real_winner == predicted_winner:
        return 1
    else:
        return 0


# now choosing the model that preform best according to this measure
def task_1_____get_winner(train_X, train_Y, X_to_split, Y_to_split, test_X):
    scores_dict = dict()
    for score_measure in ['accuracy', 'f1_micro', 'recall_micro', 'precision_micro']:
        print('starting loop for', score_measure)
        mc = Model_Chooser()
        mc.find_classifiers_best_params(train_X.to_numpy(), train_Y.to_numpy(), score_measure=score_measure)
        for i in range(50):
            print('i = ', i)
            train_X, validation_X, train_Y, validation_Y = train_test_split(X_to_split, Y_to_split, test_size=0.2)
            best_classifier, best_param_list, __ = mc.get_winner(train_X, train_Y, validation_X, validation_Y,
                                                                score_measure_func=party_win_score)
            if scores_dict.get((best_classifier, str(best_param_list))) is None:
                scores_dict[(best_classifier, str(best_param_list))] = 1
            else:
                scores_dict[(best_classifier, str(best_param_list))] += 1
        print('this is the scores_dict so far:\n', scores_dict)

    best_classifier, best_param_list = max(scores_dict.items(), key=lambda x: x[1])[0]
    best_classifier = mc.create_classifier(best_classifier, ast.literal_eval(best_param_list))
    best_classifier.fit(X_to_split, Y_to_split)
    prediction = list(best_classifier.predict(test_X))
    print('the predicted winner party is: '.capitalize(), Counter(prediction).most_common(1)[0])
    return Counter(prediction).most_common(1)[0]


# ______________________________________________________________________________________________________________________
# SECOND TASK - Predict distribution of voters

def voters_distribution_score(prediction, test_Y):
    prediction_counter = list(Counter(prediction).items())
    test_Y_counter = list(Counter(test_Y).items())
    prediction_counter.sort(key=lambda x: x[0])
    test_Y_counter.sort(key=lambda x: x[0])
    score = 0
    for prediction, actual in zip(prediction_counter, test_Y_counter):
        score -= abs(prediction[1] - actual[1])
    return score


def task_2_____get_distributions(train_X, train_Y, X_to_split, Y_to_split, test_X):
    scores_dict = dict()
    for score_measure in ['accuracy', 'f1_micro', 'recall_micro', 'precision_micro']:
        print('starting loop for', score_measure)
        mc = Model_Chooser()
        mc.find_classifiers_best_params(train_X.to_numpy(), train_Y.to_numpy(), score_measure=score_measure)
        for _ in range(50):
            train_X, validation_X, train_Y, validation_Y = train_test_split(X_to_split, Y_to_split, test_size=0.2)
            best_classifier, best_param_list, __ = mc.get_winner(train_X, train_Y, validation_X, validation_Y,
                                                                score_measure_func=voters_distribution_score)
            if scores_dict.get((best_classifier, str(best_param_list))) is None:
                scores_dict[(best_classifier, str(best_param_list))] = 1
            else:
                scores_dict[(best_classifier, str(best_param_list))] += 1
        print('this is the scores_dict so far:\n', scores_dict)

    best_classifier, best_param_list = max(scores_dict.items(), key=lambda x: x[1])[0]
    best_classifier = mc.create_classifier(best_classifier, ast.literal_eval(best_param_list))
    best_classifier.fit(X_to_split, Y_to_split)
    prediction = list(best_classifier.predict(test_X))

    prediction_counter = Counter(prediction).items()
    sum_ = sum([x[1] for x in prediction_counter])

    distribution_list = [(x[0], x[1]/sum_) for x in prediction_counter]
    distribution_list.sort(key=lambda x: x[1], reverse=True)
    print('the predicted distribution is: '.capitalize(), distribution_list)
    return distribution_list

# ______________________________________________________________________________________________________________________
# THIRD TASK - Predict distribution of voters


def task_3_____get_most_likely_voters(train_X, train_Y, X_to_split, Y_to_split, test_X):
    scores_dict = dict()
    for score_measure in ['accuracy', 'f1_micro', 'recall_micro', 'precision_micro']:
        print('starting loop for', score_measure)
        mc = Model_Chooser(predict_proba=True)
        mc.find_classifiers_best_params(train_X.to_numpy(), train_Y.to_numpy(), score_measure=score_measure)
        for _ in range(50):
            train_X, validation_X, train_Y, validation_Y = train_test_split(X_to_split, Y_to_split, test_size=0.2)
            best_classifier, best_param_list, __ = mc.get_winner(train_X, train_Y, validation_X, validation_Y)
            if scores_dict.get((best_classifier, str(best_param_list))) is None:
                scores_dict[(best_classifier, str(best_param_list))] = 1
            else:
                scores_dict[(best_classifier, str(best_param_list))] += 1
        print('this is the scores_dict so far:\n', scores_dict)

    best_classifier, best_param_list = max(scores_dict.items(), key=lambda x: x[1])[0]
    best_classifier = mc.create_classifier(best_classifier, ast.literal_eval(best_param_list))
    best_classifier.fit(X_to_split, Y_to_split)

    probabilities = best_classifier.predict_proba(test_X)
    probabile_voters = dict([(party, []) for party in range(13)])
    for p_list, voter_index in zip(probabilities, range(len(probabilities))):
        good_proba = [(party, voter_index) for party in range(len(p_list)) if p_list[party] >= 0.5]
        if len(p_list) == 0:
            continue
        for party, voter in good_proba:
            probabile_voters[party].append(voter)

    print(probabile_voters)
    return probabile_voters


    # prediction = list(best_classifier.predict(test_X))
    #
    # prediction_counter = Counter(prediction).items()
    # sum_ = sum([x[1] for x in prediction_counter])
    #
    # distribution_list = [(x[0], x[1]/sum_) for x in prediction_counter]
    # distribution_list.sort(key=lambda x: x[1], reverse=True)
    # print('the predicted distribution is: '.capitalize(), distribution_list)
    # return distribution_list

if __name__ == '__main__':
    train_X, train_Y, validation_X, validation_Y, test_X, test_Y = get_prepared_data(load=True)
    X_to_split = pd.concat([train_X, validation_X])
    Y_to_split = pd.concat([train_Y, validation_Y])

    # task_1_res = task_1_____get_winner(train_X, train_Y, X_to_split, Y_to_split, test_X)
    # task_2_res = task_2_____get_distributions(train_X, train_Y, X_to_split, Y_to_split, test_X)
    task_3_res = task_3_____get_most_likely_voters(train_X, train_Y, X_to_split, Y_to_split, test_X)

    with open('task_3.csv', 'w') as f:
        for key in task_3_res.keys():
            f.write("%s,%s\n"%(key, task_3_res[key]))

    # print('task_1_res: \n', task_1_res)
    # print('task_2_res: \n', task_2_res)
    print('task_3_res: \n', task_3_res)
