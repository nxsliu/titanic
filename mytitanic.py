# Data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualisation
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

train_df = train_df.drop(['Name', 'Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Name', 'Ticket', 'Cabin'], axis=1)

train_df['Sex'] = train_df['Sex'].map({'female': 1, 'male': 0}).astype(int)
test_df['Sex'] = test_df['Sex'].map({'female': 1, 'male': 0}).astype(int)

guess_ages = np.zeros((2,3))

for i in range(0, 2):
    for j in range(0,3):
        guess_df = train_df[(train_df['Sex'] == i) & (train_df['Pclass'] == j+1)]['Age'].dropna()

        age_guess = guess_df.median()

        guess_ages[i,j] = int(age_guess/0.5 + 0.5) * 0.5

for i in range(0,2):
    for j in range(0,3):
        train_df.loc[(train_df.Age.isnull()) & (train_df.Sex == i) & (train_df.Pclass == j+1), 'Age'] = guess_ages[i,j]

train_df['Age'] = train_df['Age'].astype(int)

for i in range(0, 2):
    for j in range(0,3):
        guess_df = test_df[(test_df['Sex'] == i) & (test_df['Pclass'] == j+1)]['Age'].dropna()

        age_guess = guess_df.median()

        guess_ages[i,j] = int(age_guess/0.5 + 0.5) * 0.5

for i in range(0,2):
    for j in range(0,3):
        test_df.loc[(test_df.Age.isnull()) & (test_df.Sex == i) & (test_df.Pclass == j+1), 'Age'] = guess_ages[i,j]

test_df['Age'] = test_df['Age'].astype(int)

train_df.loc[train_df['Age'] <= 16, 'Age'] = 0
train_df.loc[(train_df['Age'] > 16) & (train_df['Age'] <= 32), 'Age'] = 1
train_df.loc[(train_df['Age'] > 32) & (train_df['Age'] <= 48), 'Age'] = 2
train_df.loc[(train_df['Age'] > 48) & (train_df['Age'] <= 64), 'Age'] = 3
train_df.loc[train_df['Age'] > 64, 'Age'] = 4

test_df.loc[test_df['Age'] <= 16, 'Age'] = 0
test_df.loc[(test_df['Age'] > 16) & (test_df['Age'] <= 32), 'Age'] = 1
test_df.loc[(test_df['Age'] > 32) & (test_df['Age'] <= 48), 'Age'] = 2
test_df.loc[(test_df['Age'] > 48) & (test_df['Age'] <= 64), 'Age'] = 3
test_df.loc[test_df['Age'] > 64, 'Age'] = 4

train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1
test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch'] + 1

train_df = train_df.drop(['Parch', 'SibSp'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp'], axis=1)

freq_port = train_df.Embarked.dropna().mode()[0]
train_df['Embarked'] = train_df['Embarked'].fillna(freq_port)
test_df['Embarked'] = test_df['Embarked'].fillna(freq_port)
train_df['Embarked'] = train_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
test_df['Embarked'] = test_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

train_df.loc[train_df['Fare'] <= 7.91, 'Fare'] = 0
train_df.loc[(train_df['Fare'] > 7.91) & (train_df['Fare'] <= 14.454), 'Fare'] = 1
train_df.loc[(train_df['Fare'] > 14.454) & (train_df['Fare'] <= 31), 'Fare'] = 2
train_df.loc[train_df['Fare'] > 31, 'Fare'] = 3
train_df['Fare'] = train_df['Fare'].astype(int)

test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
test_df.loc[test_df['Fare'] <= 7.91, 'Fare'] = 0
test_df.loc[(test_df['Fare'] > 7.91) & (test_df['Fare'] <= 14.454), 'Fare'] = 1
test_df.loc[(test_df['Fare'] > 14.454) & (test_df['Fare'] <= 31), 'Fare'] = 2
test_df.loc[test_df['Fare'] > 31, 'Fare'] = 3
test_df['Fare'] = test_df['Fare'].astype(int)


print(train_df.head().to_string())
print(test_df.head().to_string()) 

array = train_df.drop(['PassengerId'], axis=1).values
X = array[:, 1:7]
Y = array[:, 0]
#print(X)
#print(Y)
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

seed = 7
scoring = 'accuracy'

models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('GAUSSIAN', GaussianNB()))
models.append(('PERCEPTRON', Perceptron()))
models.append(('LSVC', LinearSVC()))
models.append(('SVC', SVC()))
models.append(('SGD', SGDClassifier()))
models.append(('RDNFOREST', RandomForestClassifier()))

results = [] 
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)    
    # model.fit(X_train, Y_train)
    # predictions = model.predict(X_validation)
    # print("%s: -------------------------" % (name))
    # print(accuracy_score(Y_validation, predictions))

print('Random Forest-------------------------')
rdnf = RandomForestClassifier()
rdnf.fit(X_train, Y_train)
predictions = rdnf.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

X_test = test_df.drop("PassengerId", axis=1).copy()
print(X_test)
Y_pred = rdnf.predict(X_test)

submission = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    "Survived": Y_pred
})

submission.to_csv('output/submission.csv', index=False)