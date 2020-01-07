# Data set contatins The dataset consists of measurements of fetal heart
# rate and uterine contraction as features and fetal state class code (1=normal, 2=suspect,
# 3=pathologic) as label. There are, in total, 2126 samples with 23 features. Based on the
# numbers of instances and features (2126 is not far more than 23), the RBF kernel is the first
# choice.
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import timeit

data = pd.read_excel('CTG.xls', "Raw Data")
x = data.iloc[1:2126, 3:-2].values
y = data.iloc[1:2126, -1].values
print(Counter(y))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# now we need to tune Svm in term of penalty C and kernel coefficient
svm = SVC(kernel='rbf')
parameters = {'C': (100, 1e3, 1e4, 1e5),
              'gamma': (1e-08, 1e-7, 1e-6, 1e-5)}
grid_search = GridSearchCV(svm, parameters, n_jobs=-1, cv=3)
start_time = timeit.default_timer()
grid_search.fit(x_train, y_train)
print('Time', timeit.default_timer() - start_time)
print(grid_search.best_params_)
print(grid_search.best_score_)
svc_best = grid_search.best_estimator_
accuracy = svc_best.score(x_test, y_test)
print('Accuracy', accuracy*100)
# Also checking performance
prediction = svc_best.predict(x_test)
print('Prediction ',prediction)
report = classification_report(y_test, prediction)
print('Classiffication report', report)
