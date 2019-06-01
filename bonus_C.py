from get_prepared_data import *
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from Moodel_Chooser import task_1_____get_winner
from sklearn.ensemble import RandomForestClassifier
from collections import Counter


# (2.0, 0.265), (0.0, 0.217), (9.0, 0.081), (7.0, 0.056), (3.0, 0.053),
# (5.0, 0.051), (6.0, 0.051), (12.0, 0.046), (10.0, 0.044), (4.0, 0.037),
# (8.0, 0.036), (1.0, 0.034), (11.0, 0.029)


def plot_vote_to_features_colored(data: DataFrame):
    names = data.columns.values
    for i in range(1, len(names)):
        sns.pairplot(data.iloc[:, [0, i]], hue='Vote')
        name = 'Vote to ' + str(names[i])
        plt.title(name)
        #plt.savefig(name + '.png')
        plt.show()


def plot_vote_to_features(data: DataFrame):
    names = data.columns.values
    n = data['Vote'].values
    M = np.unique(n)

    for i, _ in enumerate(names):
        for j in M:
            data_labeled = data[data.Vote == j]
            sns.pairplot(data_labeled.iloc[:, [i]])
            name = 'Vote labeled ' + str(j) + ' to ' + str(names[i])
            plt.title(name)
            plt.show()
            plt.savefig(name + '.png')


winning = {2, 0, 9, 7}

train_X, train_Y, validation_X, validation_Y, test_X, test_Y = get_prepared_data()

data = test_X.copy()

data.insert(0, 'Vote', test_Y)

data_winning = data.loc[data['Vote'].isin(winning)]
#plot_vote_to_features(data_winning)
#plot_vote_to_features_colored(data_winning)

classifier = RandomForestClassifier(min_samples_split=2, n_estimators=5)
classifier.fit(train_X, train_Y)

#with no change
prediction = list(classifier.predict(test_X))
winner0, _ = Counter(prediction).most_common(1)[0]

#change 1
test_X['Avg_environmental_importance'] = -0.15 #makes 0 the winner

prediction = list(classifier.predict(test_X))
winner1, _ = Counter(prediction).most_common(1)[0]

#change 2
test_X['Avg_education_importance'] = -0.15 #makes 0 the winner

prediction = list(classifier.predict(test_X))
winner2, _ = Counter(prediction).most_common(1)[0]

#change 3
test_X['Weighted_education_rank'] = 0.9 #makes 0 the winner

prediction = list(classifier.predict(test_X))
winner3, _ = Counter(prediction).most_common(1)[0]

#change 3
test_X['Number_of_valued_Kneset_members'] = 0.0005 #makes 0 the winner

prediction = list(classifier.predict(test_X))
winner4, _ = Counter(prediction).most_common(1)[0]

#change 2
test_X['Avg_education_importance'] = -0.25 #makes 0 the winner

prediction = list(classifier.predict(test_X))
winner5, _ = Counter(prediction).most_common(1)[0]




